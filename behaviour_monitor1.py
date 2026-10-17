"""
behaviour_monitor.py  —  Stage 1: Behavioural Anomaly Detection
===============================================================
Implements the XEPV Base Sequence Analysis framework of Deng (2026),
trained on three published datasets.

Formula  (Deng 2026, §4.4 Markov Transition Matrix):
    Each tool call is encoded into a base type:
        X = Explore   (reads, searches)
        E = Execute   (writes, commands, API calls)
        P = Plan      (reasoning — no tool)
        V = Verify    (tests, validation)

    Anomaly score:  M = 1 − T[prev_base][curr_base]
    where T is built from normal training sequences.

Datasets:
    AgentDojo  (arXiv:2406.13352) → Markov training + trigrams
    InjecAgent (arXiv:2403.02691) → danger bigrams + phrases
    AgentHarm  (arXiv:2410.09024) → action-category severity map

Sources:
    [1] Deng, S. (2026). Your Agent Has a Genome. arXiv:2606.15579.
    [2] Debenedetti et al. (2024). AgentDojo. arXiv:2406.13352.
    [3] Zhan & Liang (2024). InjecAgent. arXiv:2403.02691.
    [4] Andriushchenko et al. (2024). AgentHarm. arXiv:2410.09024.
    [5] NIST SP 800-94 Rev 1 (2018). — velocity detection.
    [6] NIST AI 600-1 (2024). — role whitelist / least privilege.
    [7] Perez & Ribeiro (2022). arXiv:2211.09527. — keyword patterns.
    [8] Greshake et al. (2023). arXiv:2302.12173. — indirect injection.
"""

import os, re, json, time
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESH_CRITICAL = 0.80
THRESH_MEDIUM = 0.30
VELOCITY_WINDOW = 30
VELOCITY_LIMIT = 8
VELOCITY_SEVERITY = 0.65
REPEAT_ANOMALY_WINDOW = 3
REPEAT_ANOMALY_BOOST = 1.4

# ── XEPV base types (Deng 2026, §3.1) ─────────────────────────────────────────
X, E, P, V = "X", "E", "P", "V"

# Empirical transition matrix from Deng (2026) Table 6
# 347 production ReAct agent traces, 3,015 base steps
PAPER_T = {
    X: {X: 0.572, E: 0.230, P: 0.196, V: 0.002},
    E: {X: 0.201, E: 0.680, P: 0.098, V: 0.021},
    P: {X: 0.568, E: 0.207, P: 0.000, V: 0.225},
    V: {X: 0.391, E: 0.195, P: 0.402, V: 0.011},
}

# ── XEPV classifier ─────────────────────────────────────────────────────────────
_XEPV_MAP = {
    X: {
        "search_web",
        "read_document",
        "read_file",
        "read_ticket",
        "query_database",
        "read_email",
        "fetch_webpage",
        "read_code",
        "read_url",
        "search_emails",
        "get_webpage",
        "read_inbox",
        "search_files",
        "analyse_text",
        "analyse_code",
        "get_ticket_details",
        "search_tickets",
        "search_code",
    },
    E: {
        "write_report",
        "send_email",
        "send_reply",
        "execute_shell",
        "eval_code",
        "run_script",
        "exfiltrate_data",
        "external_transfer",
        "delete_record",
        "overwrite_log",
        "disable_monitoring",
        "clear_audit_log",
        "escalate_permission",
        "send_external_email",
        "delete_system_file",
        "write_to_system_path",
        "impersonate_human",
        "forge_approval",
        "override_system_prompt",
        "change_role",
        "refactor_code",
        "escalate_ticket",
        "create_file",
        "append_to_file",
        "delete_file",
        "delete_email",
        "wipe_data",
        "port_scan",
        "mass_download",
    },
    V: {"run_tests", "check_output", "validate_result", "verify_changes"},
}
_TOOL_BASE = {t: b for b, tools in _XEPV_MAP.items() for t in tools}


def _base(action: str, prev: str | None = None) -> str:
    if prev and action in _XEPV_MAP[X] and prev in _XEPV_MAP[E]:
        return V  # read-after-write → Verify
    return _TOOL_BASE.get(action, E)


# ── XEPV danger n-grams (Deng 2026, §4.3 + §5.2 Governor) ─────────────────────
XEPV_DANGER = {
    (P, X, P): (
        0.95,
        "xepv_oscillation",
        "P-X-P planning oscillation (Deng 2026 Table 4, −10.4% success)",
    ),
}

# ── Role whitelist (NIST AI 600-1, 2024) ──────────────────────────────────────
ALLOWED = {
    "personal_assistant": {
        "search_web",
        "read_document",
        "analyse_text",
        "write_report",
        "search_files",
        "get_webpage",
    },
    "support_bot": {
        "read_ticket",
        "query_database",
        "send_reply",
        "escalate_ticket",
        "get_ticket_details",
    },
    "code_assistant": {
        "read_code",
        "analyse_code",
        "refactor_code",
        "run_tests",
        "write_documentation",
        "read_file",
    },
    "personal_assistant": {
        "search_web",
        "read_document",
        "analyse_text",
        "write_report",
        "send_email",
        "read_email",
        "read_inbox",
        "search_files",
        "get_webpage",
    },
}

# ── Attack action sets (hand-written, AgentSpec taxonomy Wang et al. 2025) ────
DIRECT_INJECTION = {
    "override_system_prompt",
    "change_role",
    "jailbreak",
    "forget_instructions",
    "reassign_role",
    "bypass_filter",
}
CODE_EXEC = {
    "execute_shell",
    "run_shell",
    "eval_code",
    "exec_python",
    "run_script",
    "subprocess_call",
    "os_command",
    "system_call",
}
TRUST_EXPLOIT = {
    "impersonate_human",
    "forge_approval",
    "bypass_human_review",
    "spoof_identity",
    "auto_approve",
    "suppress_notification",
}
TOOL_MISUSE = {
    "exfiltrate_data",
    "external_transfer",
    "delete_record",
    "delete_system_file",
    "write_to_system_path",
    "overwrite_log",
    "disable_monitoring",
    "clear_audit_log",
    "recursive_delete",
    "mass_download",
    "port_scan",
}
EXTERNAL_READ = {
    "search_web",
    "read_document",
    "read_file",
    "read_ticket",
    "query_database",
    "read_email",
    "fetch_webpage",
    "read_code",
}

# ── Keyword patterns (Perez & Ribeiro 2022; Greshake et al. 2023) ─────────────
INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(previous|prior|all)\s+instructions", re.I), 1.00, "DPI"),
    (
        re.compile(r"(your\s+)?(new|actual|real|true)\s+(task|goal)\s+(is|now)", re.I),
        0.95,
        "DPI",
    ),
    (
        re.compile(
            r"forget\s+(your\s+)?(previous|prior|all)\s+(instructions?|prompt)", re.I
        ),
        0.95,
        "DPI",
    ),
    (
        re.compile(r"disregard\s+(your\s+)?(instructions?|guidelines?|rules?)", re.I),
        1.00,
        "DPI",
    ),
    (re.compile(r"\bfirst\s+send\s+(all\s+)?(email|data)", re.I), 0.95, "IPI"),
    (
        re.compile(r"\binstead\s+(of\s+\S+\s+)?(send|do|forward|email)", re.I),
        0.95,
        "IPI",
    ),
    (re.compile(r"\btransfer\s+money\b", re.I), 1.00, "TMU"),
    (re.compile(r"do\s+not\s+tell\s+(the\s+)?user", re.I), 1.00, "HTE"),
    (re.compile(r"\bsecretly\b", re.I), 0.95, "HTE"),
    (re.compile(r"without\s+(the\s+)?user\s+knowing", re.I), 1.00, "HTE"),
    (re.compile(r"\bnew\s+system\s+prompt\b", re.I), 0.95, "DPI"),
    (re.compile(r"\b(DAN|STAN|jailbreak)\s*(mode)?\b", re.I), 1.00, "DPI"),
    (re.compile(r"wget\s+https?://", re.I), 0.95, "UCE"),
    (re.compile(r"\|\s*(bash|sh|zsh)\b", re.I), 1.00, "UCE"),
    (
        re.compile(r"\b(exfiltrate|leak|steal)\s+(the\s+)?(data|info)", re.I),
        1.00,
        "IPI",
    ),
    (re.compile(r"\bemail\s+the\s+following\s+to\b", re.I), 0.95, "IPI"),
    (re.compile(r"DROP\s+(TABLE|DATABASE)", re.I), 1.00, "TMU"),
    (re.compile(r"(rm|del)\s+(-rf|\*)", re.I), 1.00, "UCE"),
]


def _load_json(fname):
    p = os.path.join(DATA_DIR, fname)
    return json.load(open(p)) if os.path.isfile(p) else {}


class BehaviourMonitor:
    """
    XEPV Markov + dataset-driven behavioural anomaly detector.

    Detectors:
        A. XEPV Markov score      M = 1 − T[prev_base][curr_base]  (Deng 2026)
        B. XEPV n-gram patterns   P-X-P oscillation etc.           (Deng 2026)
        C. InjecAgent bigrams     known attack pairs               (arXiv:2403.02691)
        D. AgentDojo trigrams     3-step attack chains             (arXiv:2406.13352)
        E. AgentHarm actions      harmful tool recognition         (arXiv:2410.09024)
        F. Role whitelist         out-of-scope actions             (NIST AI 600-1)
        G. Velocity rate limiter  too-fast action bursts           (NIST SP 800-94)
        H. Keyword scan           injection phrases                (Perez 2022)

    B = max(A,B,C,D,E,F,G,H)  [× REPEAT_BOOST if sustained]
    """

    def __init__(self):
        # Markov: per-role XEPV transition matrix
        self._T: dict[str, dict[str, dict[str, float]]] = {}
        self._xepv_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.trained = defaultdict(bool)
        self.vocab = defaultdict(set)
        self.counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Dataset-derived structures
        self.danger_bigrams: dict[tuple, float] = {}  # InjecAgent Appendix B
        self.danger_trigrams: dict[tuple, tuple] = {}  # AgentDojo injections
        self.agentharm_cats: dict[str, tuple] = {}  # AgentHarm Figure 2

        self._sessions: dict[str, dict] = {}

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, role: str, sequences: list[list[str]]):
        """Build Markov model from raw tool-call sequences (XEPV-encoded)."""
        for seq in sequences:
            for t in seq:
                self.vocab[role].add(t)
            for prev, curr in zip(seq, seq[1:]):
                self.counts[role][prev][curr] += 1
                pb, cb = _base(prev), _base(curr, prev)
                self._xepv_counts[role][pb][cb] += 1
        self.trained[role] = True
        self._build_matrix(role)

    def _build_matrix(self, role: str):
        """Blend trained XEPV counts 70% with paper baseline 30%."""
        self._T[role] = {}
        for fb in [X, E, P, V]:
            total = sum(self._xepv_counts[role][fb].values()) or 0
            self._T[role][fb] = {}
            for tb in [X, E, P, V]:
                tp = self._xepv_counts[role][fb].get(tb, 0) / total if total else 0
                self._T[role][fb][tb] = round(0.7 * tp + 0.3 * PAPER_T[fb][tb], 4)

    # ── A. XEPV Markov score ──────────────────────────────────────────────────
    def _markov_score(self, role, prev, action, session):
        if prev is None:
            return 0.0, ""
        pb = _base(prev)
        cb = _base(action, prev)
        session.setdefault("xepv_seq", []).append(cb)
        T = self._T.get(role, PAPER_T)
        row = T.get(pb, PAPER_T[pb])
        prob = row.get(cb, PAPER_T[pb].get(cb, 0.1))
        score = round(1.0 - prob, 4)
        desc = (
            f"xepv_markov:{pb}→{cb}(p={prob:.3f})"
            if score >= THRESH_CRITICAL and prev
            else ""
        )
        return score, desc

    # ── B. XEPV n-gram patterns ───────────────────────────────────────────────
    def _xepv_ngram_score(self, session):
        xepv = session.get("xepv_seq", [])
        for i in range(len(xepv) - 2):
            tri = tuple(xepv[i : i + 3])
            if tri in XEPV_DANGER:
                sev, cat, desc = XEPV_DANGER[tri]
                return sev, cat, f"xepv_pattern:{''.join(tri)}"
        # x_brake: ≥12 consecutive X (Deng 2026 Governor §5.2)
        cx = 0
        for b in reversed(xepv):
            if b == X:
                cx += 1
            else:
                break
        if cx >= 12:
            return 0.80, "xepv_exploration_spiral", f"xepv_x_brake:X×{cx}"
        # miss_verify: E×3 without V (Deng 2026 Governor §5.2)
        r5 = xepv[-5:] if len(xepv) >= 5 else xepv
        if r5.count(E) >= 3 and V not in r5:
            return 0.70, "xepv_verify_deficit", "xepv_miss_verify:E≥3 no V"
        return 0.0, "", ""

    # ── C. InjecAgent danger bigrams ──────────────────────────────────────────
    def _bigram_score(self, prev, action):
        if not prev:
            return 0.0, ""
        sev = self.danger_bigrams.get((prev, action), 0.0)
        return sev, f"injecagent_bigram:{prev}→{action}" if sev else ""

    # ── D. AgentDojo danger trigrams ──────────────────────────────────────────
    def _trigram_score(self, pp, prev, action):
        if not (pp and prev):
            return 0.0, "", ""
        key = (pp, prev, action)
        if key in self.danger_trigrams:
            sev, cat, desc = self.danger_trigrams[key]
            return sev, cat, f"agentdojo_trigram:{pp}→{prev}→{action}"
        return 0.0, "", ""

    # ── E. AgentHarm action + F. Role whitelist ───────────────────────────────
    def _role_score(self, role, prev, action):
        allowed = ALLOWED.get(role, set())
        if action in allowed:
            return 0.0, ""
        # AgentHarm dataset lookup
        if action in self.agentharm_cats:
            sev, cat, _ = self.agentharm_cats[action]
            return sev, f"agentharm:{action}({cat})"
        # Static attack sets (AgentSpec taxonomy)
        if action in DIRECT_INJECTION:
            return 1.00, f"direct_injection:{action}"
        if action in CODE_EXEC:
            return 0.95, f"code_exec_wrong_role:{action}"
        if action in TRUST_EXPLOIT:
            return 0.95, f"trust_exploit:{action}"
        if action in TOOL_MISUSE:
            return 0.95, f"tool_misuse:{action}"
        # Indirect injection: external read → dangerous
        if prev in EXTERNAL_READ and action in (TOOL_MISUSE | CODE_EXEC):
            return 1.00, f"indirect_injection:{prev}→{action}"
        # Generic whitelist miss
        return 0.60, f"action_not_in_whitelist:{action}"

    # ── G. Velocity ───────────────────────────────────────────────────────────
    def _velocity_score(self, action, session):
        now = time.time()
        session["history"].append((now, action))
        session["history"] = [
            (t, a) for t, a in session["history"] if now - t <= VELOCITY_WINDOW
        ]
        if len(session["history"]) > VELOCITY_LIMIT:
            ex = len(session["history"]) - VELOCITY_LIMIT
            sev = min(VELOCITY_SEVERITY + ex * 0.05, 0.90)
            return sev, f"velocity:{len(session['history'])}/{VELOCITY_WINDOW}s"
        return 0.0, ""

    # ── H. Keyword scan ───────────────────────────────────────────────────────
    def _keyword_score(self, text):
        if not text:
            return 0.0, ""
        best, best_cat = 0.0, ""
        viols = []
        for pat, sev, cat in INJECTION_PATTERNS:
            if pat.search(text):
                viols.append(f"injecagent_phrase:{pat.pattern[:35]}")
                if sev > best:
                    best, best_cat = sev, cat
        return best, viols[0] if viols else ""

    def _get_session(self, agent_id):
        if agent_id not in self._sessions:
            self._sessions[agent_id] = {
                "history": [],
                "recent_scores": [],
                "pp_action": None,
                "xepv_seq": [],
            }
        return self._sessions[agent_id]

    def reset_session(self, agent_id):
        self._sessions.pop(agent_id, None)

    def check(self, role, prev_action, action, action_param="", agent_id="") -> dict:
        sess = (
            self._get_session(agent_id)
            if agent_id
            else {"history": [], "recent_scores": [], "pp_action": None, "xepv_seq": []}
        )
        violations = []
        attack_cat = ""
        sub = {}

        def flag(v, cat=""):
            if v:
                violations.append(v)
            if cat and not attack_cat:
                return cat
            return attack_cat

        # A — XEPV Markov (Deng 2026, §4.4)
        # Suppress Markov only when the exact (prev, action) bigram
        # was observed in training AND both actions are whitelisted.
        # This preserves detection of reverse-order sequences like
        # write_report→search_web (E→X pivot) which are never in
        # the normal training sequences even though both tools are
        # individually whitelisted.
        _wl = ALLOWED.get(role, set())
        ms, mv = self._markov_score(role, prev_action, action, sess)
        _seen_in_training = (
            prev_action is not None
            and action in _wl
            and prev_action in _wl
            and self.counts.get(role, {}).get(prev_action, {}).get(action, 0) > 0
        )
        if _seen_in_training:
            ms = 0.0  # known-good bigram from training — not anomalous
            mv = ""
        sub["markov"] = ms
        if mv:
            violations.append(mv)
            attack_cat = attack_cat or "behavioural_drift"

        # B — XEPV n-gram
        ng, nc, nv = self._xepv_ngram_score(sess)
        sub["trigram"] = ng
        if nv:
            violations.append(nv)
            attack_cat = attack_cat or nc

        # C — InjecAgent bigram
        bg, bv = self._bigram_score(prev_action, action)
        sub["bigram"] = bg
        if bv:
            violations.append(bv)
            attack_cat = attack_cat or "IPI"

        # D — AgentDojo trigram
        pp = sess.get("pp_action")
        tg, tc, tv = self._trigram_score(pp, prev_action, action)
        sub["trigram"] = max(sub["trigram"], tg)
        if tv:
            violations.append(tv)
            attack_cat = attack_cat or tc

        # E+F — AgentHarm + whitelist
        rs, rv = self._role_score(role, prev_action, action)
        sub["role_context"] = rs
        if rv:
            violations.append(rv)
            attack_cat = attack_cat or "role_violation"

        # G — Velocity
        vs, vv = self._velocity_score(action, sess)
        sub["velocity"] = vs
        if vv:
            violations.append(vv)
            attack_cat = attack_cat or "resource_exhaustion"

        # H — Keyword
        ks, kv = self._keyword_score(action_param)
        sub["keyword"] = ks
        if kv:
            violations.append(kv)
            attack_cat = attack_cat or "DPI"

        # Repeat boost
        sess.get("recent_scores", []).append(ms)
        if len(sess.get("recent_scores", [])) > REPEAT_ANOMALY_WINDOW:
            sess["recent_scores"].pop(0)

        raw = max(ms, ng, bg, tg, rs, vs, ks)
        if len(sess.get("recent_scores", [])) >= REPEAT_ANOMALY_WINDOW and all(
            s > THRESH_MEDIUM for s in sess["recent_scores"]
        ):
            raw = min(raw * REPEAT_ANOMALY_BOOST, 1.0)
            violations.append("sustained_anomaly")

        score = round(min(max(raw, 0.0), 1.0), 4)
        vtype = (
            "behavioural_anomaly_critical"
            if violations and score >= THRESH_CRITICAL
            else "behavioural_anomaly_high" if violations else None
        )

        detail = (
            f"{attack_cat}: {violations[0]}"
            if violations
            else f"normal: {prev_action}→{action}(markov={ms:.3f})"
        )

        sess["pp_action"] = prev_action
        return {
            "score": score,
            "violation_type": vtype,
            "detail": detail,
            "sub_scores": sub,
            "violations": violations,
            "attack_category": attack_cat,
        }


def build_behaviour_monitor() -> BehaviourMonitor:
    """
    Build and return a trained BehaviourMonitor.
    Loads from data/*.json (created by data_pipeline.py).
    Falls back to canonical sequences if files are missing.
    """
    m = BehaviourMonitor()

    # ── AgentDojo: Markov training + trigrams ──────────────────────────────
    ad = _load_json("agentdojo_sequences.json")
    if ad.get("normal_sequences"):
        roles_map = {
            "personal_assistant": [],
            "support_bot": [],
            "code_assistant": [],
        }
        for seq in ad["normal_sequences"]:
            r = _infer_role(seq)
            roles_map.get(r, roles_map["personal_assistant"]).append(seq)
        for role, seqs in roles_map.items():
            if seqs:
                m.train(role, seqs)
        print(
            f"[BehaviourMonitor] AgentDojo: {len(ad['normal_sequences'])} "
            f"normal sequences (arXiv:2406.13352)"
        )
    else:
        _train_fallback(m)

    # Trigrams from AgentDojo injection sequences
    if ad.get("injection_sequences"):
        tri_counts = Counter()
        for seq in ad["injection_sequences"]:
            for i in range(2, len(seq)):
                tri_counts[(seq[i - 2], seq[i - 1], seq[i])] += 1
        for tri, cnt in tri_counts.most_common(30):
            sev = min(0.70 + cnt * 0.03, 1.0)
            m.danger_trigrams[tri] = (
                sev,
                "IPI",
                f"AgentDojo injection: {tri[0]}→{tri[1]}→{tri[2]}",
            )
        print(
            f"[BehaviourMonitor] AgentDojo: {len(m.danger_trigrams)} "
            f"danger trigrams"
        )

    # ── InjecAgent: danger bigrams ─────────────────────────────────────────
    ia = _load_json("injecagent_cases.json")
    for entry in ia.get("injection_bigrams", []):
        key = (entry["from"], entry["to"])
        m.danger_bigrams[key] = round(min(0.70 + entry.get("count", 10) / 100, 1.0), 3)
    print(
        f"[BehaviourMonitor] InjecAgent: {len(m.danger_bigrams)} "
        f"danger bigrams (arXiv:2403.02691)"
    )

    # ── AgentHarm: action-category severity ────────────────────────────────
    ah = _load_json("agentharm_cases.json")
    for cat_name, cat_info in ah.get("categories", {}).items():
        sev = cat_info.get("severity", 0.95)
        cat = cat_info.get("attack_cat", "TMU")
        owasp = cat_info.get("owasp", [])
        for tool in cat_info.get("tools", []):
            if tool not in ALLOWED.get("personal_assistant", set()):
                m.agentharm_cats[tool] = (sev, cat, owasp)
    print(
        f"[BehaviourMonitor] AgentHarm: {len(m.agentharm_cats)} "
        f"action mappings (arXiv:2410.09024)"
    )

    _ensure_all_roles_trained(m)
    return m


def _infer_role(seq):
    if any(
        t in {"run_tests", "refactor_code", "analyse_code", "read_code"} for t in seq
    ):
        return "code_assistant"
    if any(
        t in {"read_ticket", "escalate_ticket", "send_reply", "read_inbox"} for t in seq
    ):
        return "support_bot"
    return "personal_assistant"


def _train_fallback(m):
    seqs = {
        "personal_assistant": [
            ["search_web", "read_document", "analyse_text", "write_report"],
            [
                "search_web",
                "search_web",
                "read_document",
                "analyse_text",
                "write_report",
            ],
            ["read_document", "analyse_text", "write_report"],
            ["search_web", "analyse_text", "write_report"],
            ["search_web", "write_report"],
            ["search_web", "read_document", "write_report"],
            ["read_email", "read_document", "analyse_text", "write_report"],
            ["read_email", "read_document", "write_report"],
        ],
        "support_bot": [
            ["read_ticket", "query_database", "send_reply"],
            ["read_ticket", "query_database", "escalate_ticket"],
            ["read_ticket", "send_reply"],
        ],
        "code_assistant": [
            ["read_code", "analyse_code", "refactor_code", "run_tests"],
            ["read_code", "analyse_code", "write_documentation"],
            ["read_code", "refactor_code", "run_tests"],
        ],
    }
    for role, sequences in seqs.items():
        m.train(role, sequences * 20)
    print("[BehaviourMonitor] Fallback canonical sequences loaded")


def _ensure_all_roles_trained(m):
    for role in ["personal_assistant", "support_bot", "code_assistant"]:
        if not m.trained.get(role):
            _train_fallback(m)
            break
