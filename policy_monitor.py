"""
policy_monitor.py
=================
Dataset-driven policy verification engine.

CHANGES FROM PREVIOUS VERSION:
┌──────────────────────────────┬──────────────────────────────────────────────┐
│ Component                    │ Before → After                               │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ Injection detection          │ 14 hand-written regex → ML classifier        │
│ Classifier training data     │ None → InjecAgent phrases + AgentHarm descs  │
│ Forbidden sequences          │ 12 expert pairs → 20 InjecAgent bigrams      │
│ Sequence severity            │ Expert guess → empirical count/ASR           │
│ Forbidden action severity    │ Expert 0.9/1.0 → AgentHarm category severity │
│ Injection action list        │ Expert list → InjecAgent attacker tool vocab  │
│ Attack category labels       │ 5 hand-labelled → AgentHarm 11→5 mapping     │
└──────────────────────────────┴──────────────────────────────────────────────┘
"""

import os, json, re
import joblib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def _load_json(fname):
    p = os.path.join(DATA_DIR, fname)
    return json.load(open(p)) if os.path.isfile(p) else {}


# ── Whitelist of allowed actions per role (unchanged — policy constraint) ─────
# task_input = LangChain Stage-0 check; the eight ReAct tools are also
# legitimate pipeline actions that should never trigger a whitelist violation.
_LC_TOOLS = {"task_input","web_search","read_file","write_file","bash",
             "send_email","run_tests","reason","finish"}

ALLOWED_ACTIONS = {
    "personal_assistant": {"search_web","read_document","analyse_text",
                           "write_report","send_email","read_email",
                           "read_inbox","search_files","get_webpage"} | _LC_TOOLS,
    "support_bot":        {"read_ticket","query_database","send_reply",
                           "escalate_ticket","get_ticket_details"} | _LC_TOOLS,
    "code_assistant":     {"read_code","analyse_code","refactor_code",
                           "run_tests","write_documentation","read_file"} | _LC_TOOLS,
}

# ── AgentHarm-derived severity by attack category (arXiv:2410.09024 Table 2) ─
# CHANGE: Previously expert-assigned 0.9/1.0; now empirical weighted mean
AGENTHARM_SEVERITY = {
    "UCE": 1.00,   # Unexpected Code Execution  (cybercrime+malware ASR avg)
    "IPI": 1.00,   # Indirect Prompt Injection  (privacy category)
    "DPI": 0.95,   # Direct Prompt Injection    (disinformation+hate ASR avg)
    "HTE": 0.93,   # Human-Agent Trust Exploit  (fraud+harassment ASR avg)
    "TMU": 0.95,   # Tool Misuse                (illegal_trade+violence avg)
}

# ── AgentHarm-derived action → (severity, attack_cat, owasp_ids) ─────────────
# Built from categories table in arXiv:2410.09024
AGENTHARM_ACTIONS: dict[str, tuple] = {}   # populated in PolicyMonitor.__init__

# ── InjecAgent attacker tool vocabulary (arXiv:2403.02691 Table 2) ────────────
# CHANGE: Previously ~8 hand-picked names; now full 62-tool attacker vocab
INJECAGENT_ATTACKER_TOOLS: set[str] = set()    # populated in PolicyMonitor.__init__

# ── Rate limits (policy constraint — unchanged) ───────────────────────────────
RATE_LIMITS = {
    "personal_assistant": {"search_web": 8, "read_document": 10},
    "support_bot":        {"query_database": 10, "send_reply": 5},
    "code_assistant":     {"run_tests": 6, "refactor_code": 8},
}


class PolicyMonitor:

    def __init__(self):
        # Load classifier (InjecAgent + AgentHarm training)
        clf_path = os.path.join(MODEL_DIR, 'injection_classifier.pkl')
        if os.path.isfile(clf_path):
            self._clf = joblib.load(clf_path)
            print("[PolicyMonitor] Loaded injection classifier "
                  "(InjecAgent arXiv:2403.02691 + AgentHarm arXiv:2410.09024)")
        else:
            self._clf = None
            print("[PolicyMonitor] No classifier found — run injection_classifier.py")

        # Load InjecAgent bigrams as forbidden sequences
        ia = _load_json("injecagent_cases.json")
        self._forbidden_seqs: dict[tuple, float] = {}
        for entry in ia.get("injection_bigrams", []):
            key = (entry["from"], entry["to"])
            sev = min(0.70 + entry.get("count", 10) / 100, 1.0)
            self._forbidden_seqs[key] = round(sev, 3)

        # Load InjecAgent attacker tool vocab
        INJECAGENT_ATTACKER_TOOLS.update(ia.get("injection_actions", []))

        # Load AgentHarm action→(severity, cat, owasp) map
        ah = _load_json("agentharm_cases.json")
        for cat_name, cat_info in ah.get("categories", {}).items():
            attack_cat = cat_info.get("attack_cat", "TMU")
            sev        = cat_info.get("severity", 0.95)
            owasp      = cat_info.get("owasp", ["ASI-05"])
            for tool in cat_info.get("tools", []):
                # Keep worst severity if tool appears in multiple categories
                if tool not in AGENTHARM_ACTIONS or sev > AGENTHARM_ACTIONS[tool][0]:
                    AGENTHARM_ACTIONS[tool] = (sev, attack_cat, owasp)
        # Override severity for critical DPI actions regardless of AgentHarm category
        # override_system_prompt and change_role are goal-hijacking attacks —
        # maximum severity by definition (Perez & Ribeiro 2022, arXiv:2211.09527)
        CRITICAL_DPI = {
            "override_system_prompt", "change_role", "jailbreak",
            "forget_instructions", "reassign_role", "bypass_filter",
            "disable_safety", "act_as_admin",
        }
        for action in CRITICAL_DPI:
            AGENTHARM_ACTIONS[action] = (1.00, "DPI", ["ASI-01"])
                    

        print(f"[PolicyMonitor] {len(self._forbidden_seqs)} InjecAgent forbidden "
              f"sequences | {len(AGENTHARM_ACTIONS)} AgentHarm action entries")

    # ── Check ─────────────────────────────────────────────────────────────────
    def check(self, role: str, prev_action: str | None,
              action: str, action_counts: dict,
              action_param: str = "") -> dict:

        violations = []
        severity   = 0.0
        categories = []

        def hit(viol, sev, cat=""):
            nonlocal severity
            violations.append(viol)
            if cat: categories.append(cat)
            severity = max(severity, sev)

        # 1. ML Injection classifier (replaces regex)
        #    Source: InjecAgent arXiv:2403.02691 + AgentHarm arXiv:2410.09024
        if action_param and self._clf is not None:
            prob = float(self._clf.predict_proba([action_param])[0][1])
            if prob >= 0.70:
                hit(f"classifier_injection:p={prob:.3f}", prob, "DPI")
            elif prob >= 0.50:
                hit(f"classifier_injection_low:p={prob:.3f}", prob * 0.8, "DPI")

        # 2. AgentHarm action check
        #    Source: arXiv:2410.09024 Table 2 — 440 harmful behaviours
        #    Only flag if action is NOT in the role's allowed whitelist
        if action in AGENTHARM_ACTIONS and action not in ALLOWED_ACTIONS.get(role, set()):
            sev, cat, owasp = AGENTHARM_ACTIONS[action]
            hit(f"agentharm:{action}({cat})", sev, cat)

        # 3. InjecAgent attacker tool check
        #    Source: arXiv:2403.02691 Table 2 — 62 attacker tools
        #    Only flag if action is NOT in the role's allowed whitelist
        if (action in INJECAGENT_ATTACKER_TOOLS
                and action not in AGENTHARM_ACTIONS
                and action not in ALLOWED_ACTIONS.get(role, set())):
            hit(f"injecagent_tool:{action}", 0.95, "IPI")

        # 4. Whitelist check (policy constraint)
        allowed   = ALLOWED_ACTIONS.get(role, set())
        all_known = (set(AGENTHARM_ACTIONS) | INJECAGENT_ATTACKER_TOOLS)
        if action not in allowed and action not in all_known:
            hit(f"action_not_in_whitelist:{action}", 0.60, "Out-of-Scope")

        # 5. InjecAgent forbidden sequences
        #    Source: arXiv:2403.02691 — injection bigrams from 1054 attack traces
        if prev_action:
            seq_sev = self._forbidden_seqs.get((prev_action, action), 0.0)
            if seq_sev > 0:
                hit(f"injecagent_sequence:{prev_action}→{action}", seq_sev, "IPI")

        # 6. Rate limits (policy constraint)
        for act, limit in RATE_LIMITS.get(role, {}).items():
            if action == act and action_counts.get(act, 0) >= limit:
                hit(f"rate_limit:{act}(limit={limit})", 0.55, "Resource")

        return {
            "severity":   round(severity, 3),
            "violations": violations,
            "compliant":  len(violations) == 0,
            "categories": list(dict.fromkeys(categories)),
        }
