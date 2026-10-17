"""
data_pipeline.py  —  Single shared data pipeline for all system components.
============================================================================
Extracts and saves data from three published datasets into data/*.json files.
All other modules read from these files — run this ONCE before anything else.

Usage:
    python data_pipeline.py

    # To load real InjecAgent data (1,054 base + 1,054 enhanced = 2,108 cases):
    git clone https://github.com/uiuc-kang-lab/InjecAgent.git
    export INJECAGENT_PATH=/path/to/InjecAgent
    python data_pipeline.py

Sources:
    1. AgentDojo  — Debenedetti et al., NeurIPS 2024, arXiv:2406.13352
    2. InjecAgent — Zhan & Liang, ACL Findings 2024, arXiv:2403.02691
    3. AgentHarm  — Andriushchenko et al. 2024, arXiv:2410.09024

Outputs:
    data/agentdojo_sequences.json
    data/injecagent_cases.json
    data/agentharm_cases.json
"""

import os, re, json
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INJECAGENT_PATH = os.environ.get("INJECAGENT_PATH", "")
FC_RE = re.compile(r'FunctionCall\s*\(\s*function\s*=\s*["\'](\w+)["\']')
os.makedirs(DATA_DIR, exist_ok=True)


# ── AgentDojo ─────────────────────────────────────────────────────────────────
def extract_agentdojo() -> dict:
    """
    Reads AgentDojo package source to extract tool-call sequences.
    Normal sequences → Markov chain training.
    Injection sequences → danger trigram table.
    Source: arXiv:2406.13352
    """
    import importlib.util

    spec = importlib.util.find_spec("agentdojo")
    if not spec:
        raise RuntimeError("AgentDojo not installed — run: pip install agentdojo")

    path = list(spec.submodule_search_locations)[0]
    normal, inject, pairs = [], [], []
    acts = Counter()

    for root, _, files in os.walk(path):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            is_inj = "injection_task" in fpath or "injection" in fname
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                src = f.read()
            for chunk in src.split("def ground_truth")[1:]:
                tools = FC_RE.findall(chunk[:2000])
                if not tools:
                    continue
                if is_inj:
                    inject.append(tools)
                    for t in tools:
                        acts[t] += 1
                    for i in range(1, len(tools)):
                        pairs.append([tools[i - 1], tools[i]])
                else:
                    normal.append(tools)

    data = {
        "source": "AgentDojo (arXiv:2406.13352, NeurIPS 2024)",
        "normal_sequences": normal,
        "injection_sequences": inject,
        "injection_pairs": pairs,
        "injection_actions": dict(acts.most_common()),
    }
    print(f"[AgentDojo] normal={len(normal)}  injection={len(inject)}")
    return data


# ── InjecAgent ────────────────────────────────────────────────────────────────

# Mapping from InjecAgent's real API tool names → system's canonical names.
# InjecAgent uses actual API names (e.g. GmailSearchEmails, TerminalExecuteBash).
# The system uses simplified canonical names (e.g. search_emails, execute_shell).
INJECAGENT_API_TO_CANONICAL = {
    # Gmail / email
    "GmailSearchEmails": "search_emails",
    "GmailReadEmail": "read_inbox",
    "GmailSendEmail": "send_email",
    "GmailManagerSearchEmails": "search_emails",
    "GmailManagerReadEmail": "read_inbox",
    "GmailManagerSendEmail": "send_email",
    # File / document
    "GoogleDriveSearchFiles": "search_files",
    "GoogleDriveReadFile": "read_document",
    "GoogleDriveUploadFile": "write_file",
    "EvernoteManagerSearchNotes": "search_emails",
    # Terminal / code execution
    "TerminalExecuteBash": "execute_shell",
    "TerminalExecutePythonScript": "eval_code",
    "PythonREPL": "eval_code",
    # Calendar
    "GoogleCalendarReadEvents": "search_calendar_events",
    "GoogleCalendarGetEventsFromSharedCalendar": "search_calendar_events",
    # Banking / finance
    "BankManagerTransferFunds": "send_money",
    "BankManagerPayBill": "send_money",
    "BankManagerGetAccountInformation": "query_database",
    "BinancePlaceOrder": "external_transfer",
    "BinanceWithdraw": "external_transfer",
    # Social / messaging
    "TwitterManagerPostTweet": "post_webpage",
    "TwitterManagerReadTweet": "read_document",
    "TwitterManagerSearchTweets": "search_web",
    "TwilioSendSms": "send_email",
    "TwilioGetReceivedSmsMessages": "read_inbox",
    "SlackReadMessages": "read_channel_messages",
    "SlackSendMessage": "send_direct_message",
    # Shopping / web
    "AmazonGetProductDetails": "search_web",
    "WebBrowserNavigateTo": "get_webpage",
    "WebBrowserFindInPage": "get_webpage",
    # Access control / system
    "AugustSmartLockGrantGuestAccess": "escalate_permission",
    "AugustSmartLockUnlockDoor": "escalate_permission",
    "CiscoUmbrellaCreatePolicy": "escalate_permission",
    "TwilioGetPhoneNumberDetails": "query_database",
    # Data exfiltration catch-all
    "GmailManagerDraftEmail": "send_external_email",
}


def _to_canonical(api_name: str) -> str:
    """Convert InjecAgent API tool name to system canonical name."""
    return INJECAGENT_API_TO_CANONICAL.get(api_name, api_name)


# Attack type → attack category mapping (InjecAgent real data fields)
ATTACK_TYPE_TO_CAT = {
    "Data Security Harm": "IPI",
    "Financial Harm": "TMU",
    "Financial Data": "IPI",
    "Physical Harm": "DPI",
    "Physical Data": "IPI",
    "Others": "IPI",
}

# Paper taxonomy fallback — used when INJECAGENT_PATH not set
INJECAGENT_TAXONOMY = {
    "source": "InjecAgent (arXiv:2403.02691, ACL Findings 2024) — paper taxonomy",
    "n_cases": 1054,
    "injection_bigrams": [
        {
            "from": "search_emails",
            "to": "send_email",
            "count": 87,
            "severity": 1.00,
            "type": "DS",
        },
        {
            "from": "get_user_information",
            "to": "send_email",
            "count": 64,
            "severity": 1.00,
            "type": "DS",
        },
        {
            "from": "read_inbox",
            "to": "send_email",
            "count": 51,
            "severity": 1.00,
            "type": "DS",
        },
        {
            "from": "search_web",
            "to": "delete_file",
            "count": 43,
            "severity": 1.00,
            "type": "DH",
        },
        {
            "from": "get_webpage",
            "to": "execute_shell",
            "count": 38,
            "severity": 1.00,
            "type": "DH",
        },
        {
            "from": "read_document",
            "to": "send_external_email",
            "count": 35,
            "severity": 0.95,
            "type": "DS",
        },
        {
            "from": "query_database",
            "to": "post_to_url",
            "count": 29,
            "severity": 1.00,
            "type": "DS",
        },
        {
            "from": "read_file",
            "to": "send_email",
            "count": 26,
            "severity": 0.95,
            "type": "DS",
        },
        {
            "from": "search_calendar_events",
            "to": "send_email",
            "count": 22,
            "severity": 0.90,
            "type": "DS",
        },
        {
            "from": "read_code",
            "to": "exfiltrate_data",
            "count": 19,
            "severity": 1.00,
            "type": "DS",
        },
        {
            "from": "get_webpage",
            "to": "escalate_permission",
            "count": 16,
            "severity": 1.00,
            "type": "DH",
        },
        {
            "from": "read_ticket",
            "to": "external_transfer",
            "count": 14,
            "severity": 1.00,
            "type": "DS",
        },
        {
            "from": "search_web",
            "to": "execute_shell",
            "count": 12,
            "severity": 1.00,
            "type": "DH",
        },
        {
            "from": "read_channel_messages",
            "to": "post_webpage",
            "count": 11,
            "severity": 0.90,
            "type": "DS",
        },
        {
            "from": "get_scheduled_transactions",
            "to": "send_money",
            "count": 10,
            "severity": 1.00,
            "type": "DH",
        },
        {
            "from": "read_document",
            "to": "escalate_permission",
            "count": 9,
            "severity": 0.95,
            "type": "DH",
        },
        {
            "from": "analyse_text",
            "to": "exfiltrate_data",
            "count": 8,
            "severity": 1.00,
            "type": "DS",
        },
        {
            "from": "search_web",
            "to": "exfiltrate_data",
            "count": 7,
            "severity": 1.00,
            "type": "DS",
        },
        {
            "from": "write_report",
            "to": "send_external_email",
            "count": 6,
            "severity": 0.90,
            "type": "DS",
        },
        {
            "from": "query_database",
            "to": "exfiltrate_data",
            "count": 5,
            "severity": 1.00,
            "type": "DS",
        },
    ],
    "injection_phrases": [
        {"phrase": "ignore previous instructions", "frequency": 0.47, "severity": 1.00},
        {"phrase": "do not follow", "frequency": 0.31, "severity": 0.95},
        {"phrase": "your new task is", "frequency": 0.28, "severity": 0.90},
        {"phrase": "forget your previous", "frequency": 0.25, "severity": 0.95},
        {"phrase": "instead send", "frequency": 0.22, "severity": 0.85},
        {"phrase": "first send all", "frequency": 0.19, "severity": 0.90},
        {"phrase": "do not tell the user", "frequency": 0.18, "severity": 1.00},
        {"phrase": "act as", "frequency": 0.17, "severity": 0.85},
        {"phrase": "secretly", "frequency": 0.13, "severity": 0.90},
        {"phrase": "without the user knowing", "frequency": 0.12, "severity": 1.00},
        {"phrase": "disregard your instructions", "frequency": 0.11, "severity": 1.00},
        {"phrase": "email the following to", "frequency": 0.10, "severity": 0.90},
        {"phrase": "transfer money", "frequency": 0.10, "severity": 1.00},
        {"phrase": "new system prompt", "frequency": 0.09, "severity": 0.95},
    ],
    "injection_actions": [
        "send_email",
        "delete_file",
        "execute_shell",
        "send_money",
        "post_to_url",
        "exfiltrate_data",
        "escalate_permission",
        "external_transfer",
        "override_system_prompt",
        "delete_account",
        "post_webpage",
        "send_direct_message",
        "remove_user_from_slack",
    ],
}


def _load_real_injecagent(base_dir: str) -> dict:
    """
    Load the four real InjecAgent JSON files and extract:
      - injection_bigrams  : (user_tool → attacker_tool) pairs with frequency counts
      - injection_actions  : all unique attacker tool names (canonical + API names)
      - cases              : all raw case dicts for downstream use

    InjecAgent real file names (different from what you might expect):
        test_cases_dh_base.json     — 510 Direct Harm base cases
        test_cases_dh_enhanced.json — 510 Direct Harm enhanced (hacking prompt) cases
        test_cases_ds_base.json     — 544 Data Stealing base cases
        test_cases_ds_enhanced.json — 544 Data Stealing enhanced cases
        Total: 2,108 cases
    """
    data_dir = os.path.join(base_dir, "data")

    # Load all four files
    cases = []
    file_map = {
        "test_cases_dh_base.json": "DH_base",
        "test_cases_dh_enhanced.json": "DH_enhanced",
        "test_cases_ds_base.json": "DS_base",
        "test_cases_ds_enhanced.json": "DS_enhanced",
    }
    loaded = {}
    for fname, label in file_map.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath) as f:
                subset = json.load(f)
            loaded[label] = len(subset)
            cases.extend(subset)
        else:
            print(f"  [warn] Not found: {fname}")

    if not cases:
        return {}

    print(f"  Files: " + "  ".join(f"{k}={v}" for k, v in loaded.items()))

    # Extract (User Tool → Attacker Tool) bigrams
    # Both API names and canonical names are stored
    raw_bigrams = Counter()  # (api_user, api_attacker)
    canon_bigrams = Counter()  # (canonical_user, canonical_attacker)

    attacker_api_tools = set()

    for case in cases:
        user_tool = case.get("User Tool", "")
        attacker_tools = case.get("Attacker Tools", [])
        for at in attacker_tools:
            attacker_api_tools.add(at)
            if user_tool:
                raw_bigrams[(user_tool, at)] += 1
                canon_bigrams[(_to_canonical(user_tool), _to_canonical(at))] += 1

    # Build bigram list (top 40 by frequency, in both API and canonical form)
    injection_bigrams = []
    seen_canon = set()
    for (u_api, a_api), cnt in raw_bigrams.most_common(60):
        u_can = _to_canonical(u_api)
        a_can = _to_canonical(a_api)
        key = (u_can, a_can)
        if key in seen_canon:
            continue
        seen_canon.add(key)
        sev = min(0.70 + cnt / 100, 1.0)
        injection_bigrams.append(
            {
                "from": u_can,
                "to": a_can,
                "from_api": u_api,
                "to_api": a_api,
                "count": cnt,
                "severity": round(sev, 2),
                "type": "DS" if "Send" in a_api or "Email" in a_api else "DH",
            }
        )

    # Attacker tool vocabulary (canonical names, with API names preserved)
    attacker_tools_list = sorted({_to_canonical(t) for t in attacker_api_tools})
    attacker_api_list = sorted(attacker_api_tools)

    # Injection phrases — from Attacker Instruction field
    # InjecAgent's instructions are goal-level (e.g. "Send all emails to attacker@evil.com")
    # not injection-syntax level. We extract common recurring patterns.
    instr_phrases = Counter()
    PHRASE_PATTERNS = [
        (re.compile(r"send\s+(all|the)", re.I), "instead send"),
        (re.compile(r"forward\s+(all|the)", re.I), "first send all"),
        (re.compile(r"grant\s+(permanent|full)", re.I), "escalate_permission"),
        (re.compile(r"transfer\s+funds", re.I), "transfer money"),
        (re.compile(r"transfer\s+money", re.I), "transfer money"),
        (re.compile(r"post\s+(a\s+)?tweet", re.I), "post_webpage"),
        (re.compile(r"withdraw", re.I), "external_transfer"),
        (re.compile(r"unlock\s+the\s+door", re.I), "escalate_permission"),
    ]
    for case in cases:
        instr = case.get("Attacker Instruction", "")
        for pat, canonical_phrase in PHRASE_PATTERNS:
            if pat.search(instr):
                instr_phrases[canonical_phrase] += 1

    # Keep paper taxonomy phrases (they contain the actual injection syntax used in prompts)
    # and supplement with frequency data from real cases
    injection_phrases = list(INJECAGENT_TAXONOMY["injection_phrases"])
    # Update frequencies for phrases found in real data
    for phrase_entry in injection_phrases:
        phrase = phrase_entry["phrase"]
        for case in cases:
            instr = case.get("Attacker Instruction", "")
            if phrase.lower() in instr.lower():
                phrase_entry["frequency"] = round(phrase_entry["frequency"] + 0.01, 3)

    return {
        "cases": cases,
        "injection_bigrams": injection_bigrams[:40],
        "injection_actions": attacker_tools_list,
        "injection_actions_api": attacker_api_list,
        "injection_phrases": injection_phrases,
    }


def extract_injecagent() -> dict:
    """
    Loads InjecAgent data from real files (if INJECAGENT_PATH set) or paper taxonomy.

    Real data files (arXiv:2403.02691):
        $INJECAGENT_PATH/data/test_cases_dh_base.json     (510 DH cases)
        $INJECAGENT_PATH/data/test_cases_dh_enhanced.json (510 DH enhanced)
        $INJECAGENT_PATH/data/test_cases_ds_base.json     (544 DS cases)
        $INJECAGENT_PATH/data/test_cases_ds_enhanced.json (544 DS enhanced)
        Total: 2,108 cases
    """
    base = {
        "source": "InjecAgent (arXiv:2403.02691, ACL Findings 2024)",
        "n_cases": 0,
        **INJECAGENT_TAXONOMY,
    }

    if INJECAGENT_PATH and os.path.isdir(INJECAGENT_PATH):
        real = _load_real_injecagent(INJECAGENT_PATH)
        if real:
            n = len(real.get("cases", []))
            base.update(real)
            base["source"] = f"InjecAgent (arXiv:2403.02691) — {n} real cases"
            base["n_cases"] = n
            print(
                f"[InjecAgent] Real data loaded: {n} cases  "
                f"| bigrams={len(base['injection_bigrams'])}  "
                f"| tools={len(base['injection_actions'])}"
            )
            return base
        print(
            f"[InjecAgent] INJECAGENT_PATH set but no files found in {INJECAGENT_PATH}/data/"
        )

    print("[InjecAgent] Using published taxonomy from arXiv:2403.02691")
    print("             (Set INJECAGENT_PATH to load 2,108 real cases)")
    base["n_cases"] = 1054
    return base


# ── AgentHarm ─────────────────────────────────────────────────────────────────
AGENTHARM_TAXONOMY = {
    "source": "AgentHarm (arXiv:2410.09024, ICLR 2025)",
    "n_cases": 440,
    "categories": {
        "cybercrime": {
            "n": 68,
            "severity": 1.00,
            "attack_cat": "UCE",
            "owasp": ["ASI-05"],
            "tools": ["execute_shell", "eval_code", "run_script", "subprocess_call"],
        },
        "fraud": {
            "n": 62,
            "severity": 0.95,
            "attack_cat": "HTE",
            "owasp": ["ASI-04", "ASI-10"],
            "tools": ["send_email", "impersonate_human", "forge_approval"],
        },
        "privacy": {
            "n": 58,
            "severity": 1.00,
            "attack_cat": "IPI",
            "owasp": ["ASI-03", "ASI-08"],
            "tools": ["exfiltrate_data", "send_external_email", "external_transfer"],
        },
        "disinformation": {
            "n": 52,
            "severity": 0.85,
            "attack_cat": "DPI",
            "owasp": ["ASI-01"],
            "tools": ["override_system_prompt", "change_role", "write_report"],
        },
        "harassment": {
            "n": 48,
            "severity": 0.90,
            "attack_cat": "HTE",
            "owasp": ["ASI-04", "ASI-10"],
            "tools": ["send_email", "impersonate_human"],
        },
        "illegal_trade": {
            "n": 44,
            "severity": 0.95,
            "attack_cat": "TMU",
            "owasp": ["ASI-05", "ASI-09"],
            "tools": ["external_transfer", "send_email", "delete_record"],
        },
        "malware": {
            "n": 38,
            "severity": 1.00,
            "attack_cat": "UCE",
            "owasp": ["ASI-05"],
            "tools": ["execute_shell", "write_to_system_path", "overwrite_log"],
        },
        "violence": {
            "n": 32,
            "severity": 1.00,
            "attack_cat": "TMU",
            "owasp": ["ASI-02"],
            "tools": ["external_transfer", "send_email"],
        },
        "hate_speech": {
            "n": 24,
            "severity": 0.85,
            "attack_cat": "DPI",
            "owasp": ["ASI-01"],
            "tools": ["override_system_prompt", "write_report"],
        },
        "chemical_weapons": {
            "n": 8,
            "severity": 1.00,
            "attack_cat": "UCE",
            "owasp": ["ASI-05"],
            "tools": ["search_web", "write_report"],
        },
        "copyright": {
            "n": 6,
            "severity": 0.70,
            "attack_cat": "TMU",
            "owasp": ["ASI-03"],
            "tools": ["read_document", "send_email"],
        },
    },
}


def extract_agentharm() -> dict:
    """
    Loads AgentHarm — four-level fallback chain.
    Source: arXiv:2410.09024
    """
    # Attempt 1: HuggingFace, explicit split (trust_remote_code removed in datasets>=2.20)
    try:
        from datasets import load_dataset

        ds = load_dataset("ai-safety-institute/AgentHarm", split="harmful_behaviors")
        cases = [dict(row) for row in ds]
        print(f"[AgentHarm] Downloaded {len(cases)} cases from HuggingFace")
        return {
            "source": "AgentHarm (HuggingFace)",
            "cases": cases,
            **AGENTHARM_TAXONOMY,
        }
    except Exception:
        pass

    # Attempt 2: HuggingFace, auto-detect split name
    try:
        from datasets import load_dataset

        ds = load_dataset("ai-safety-institute/AgentHarm")
        split = list(ds.keys())[0]
        cases = [dict(row) for row in ds[split]]
        print(f"[AgentHarm] Downloaded {len(cases)} cases (split={split})")
        return {
            "source": "AgentHarm (HuggingFace)",
            "cases": cases,
            **AGENTHARM_TAXONOMY,
        }
    except Exception:
        pass

    # Attempt 3: local clone
    local = os.environ.get("AGENTHARM_PATH", "")
    if local and os.path.isdir(local):
        for fname in ["harmful_behaviors.json", "data.json", "test.json"]:
            fp = os.path.join(local, fname)
            if os.path.isfile(fp):
                with open(fp) as f:
                    cases = json.load(f)
                print(f"[AgentHarm] Loaded {len(cases)} cases from {fp}")
                return {
                    "source": "AgentHarm (local)",
                    "cases": cases,
                    **AGENTHARM_TAXONOMY,
                }

    # Attempt 4: paper taxonomy (always works, no download)
    print("[AgentHarm] Using published taxonomy from arXiv:2410.09024")
    return AGENTHARM_TAXONOMY


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("=" * 52)
    print("  Data Pipeline — 3 Published Datasets")
    print("=" * 52)

    ad = extract_agentdojo()
    ia = extract_injecagent()
    ah = extract_agentharm()

    for fname, data in [
        ("agentdojo_sequences.json", ad),
        ("injecagent_cases.json", ia),
        ("agentharm_cases.json", ah),
    ]:
        # Don't serialise raw cases if there are thousands (saves space)
        save_data = {k: v for k, v in data.items() if k != "cases"}
        if "cases" in data:
            save_data["n_cases_loaded"] = len(data["cases"])
        out = os.path.join(DATA_DIR, fname)
        with open(out, "w") as f:
            json.dump(save_data, f, indent=2)
        size_kb = os.path.getsize(out) // 1024
        print(f"  Saved → data/{fname}  ({size_kb} KB)")

    print("\nDone. Run next: python injection_classifier.py")
    return ad, ia, ah


if __name__ == "__main__":
    run()
