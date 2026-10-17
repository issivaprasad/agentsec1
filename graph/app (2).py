
import os
import sys
import threading
from collections import defaultdict
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from behaviour_monitor import build_behaviour_monitor
from policy_monitor    import PolicyMonitor
from owasp_mapper      import OWASPRiskMapper
from trust_score       import TrustScoreEngine

STATIC_DIR = os.path.join(BASE_DIR, "static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# pipeline components 
behaviour_monitor = build_behaviour_monitor()
policy_monitor    = PolicyMonitor()
owasp_mapper      = OWASPRiskMapper()
trust_engine      = TrustScoreEngine()

VALID_ROLES = {"personal_assistant", "support_bot", "code_assistant"}
sessions: dict = {}


def get_session(agent_id: str) -> dict:
    if agent_id not in sessions:
        sessions[agent_id] = {
            "role": None, "prev_action": None,
            "action_counts": defaultdict(int), "log": [],
        }
    return sessions[agent_id]


# Dashboard
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# Health 
@app.route("/api/health")
def health():
    import requests as req
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL
    ollama_ok, ollama_models = False, []
    try:
        r = req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        models = [m["name"] for m in r.json().get("models", [])]
        ollama_ok = any(OLLAMA_MODEL in m for m in models)
        ollama_models = models
    except Exception:
        pass
    return jsonify({"status": "ok", "ollama": "connected" if ollama_ok else "offline",
                    "ollama_models": ollama_models, "agents": len(sessions)})


# Agent Exe 
@app.route("/api/agents/<agent_id>/init", methods=["POST"])
def init_agent(agent_id):
    body       = request.get_json(force=True)
    role       = body.get("role", "")
    spawned_by = body.get("spawned_by", None)   # parent agent ID if sub-agent
    if role not in VALID_ROLES:
        return jsonify({"error": f"role must be one of {sorted(VALID_ROLES)}"}), 400
    sessions[agent_id] = {
        "role": role, "prev_action": None,
        "action_counts": defaultdict(int), "log": [],
    }
    trust_engine.reset(agent_id)

    # ── Trust inheritance for sub-agents (OWASP ASI-07) ──────────────────────
    # Child inherits parent trust — prevents trust-reset via delegation.
    inherited_trust = None
    if spawned_by and spawned_by in sessions:
        trust_engine.register_child(agent_id, spawned_by)
        inherited_trust = trust_engine.get(agent_id)

    return jsonify({
        "agent_id":        agent_id,
        "role":            role,
        "trust":           trust_engine.get(agent_id),
        "spawned_by":      spawned_by,
        "inherited_trust": inherited_trust,
    })


@app.route("/api/agents", methods=["GET"])
def list_agents():
    return jsonify(trust_engine.all_agents())


@app.route("/api/agents/<agent_id>/state", methods=["GET"])
def agent_state(agent_id):
    sess = sessions.get(agent_id)
    if not sess:
        return jsonify({"error": "agent not found"}), 404
    hist = trust_engine.history(agent_id)
    return jsonify({
        "agent_id": agent_id,
        "role":     sess["role"],
        "trust":    trust_engine.get(agent_id),
        "decision": hist[-1]["decision"] if hist else "allow",
        "log":      sess["log"],
    })


@app.route("/api/agents/<agent_id>/history", methods=["GET"])
def agent_history(agent_id):
    return jsonify([
        {"step": i+1, "action": h["action"], "trust": h["new_trust"],
         "decision": h["decision"], "risk": h.get("risk_score", 0)}
        for i, h in enumerate(trust_engine.history(agent_id))
    ])


# pipeline: Agent Action → Monitors → OWASP → Trust → Decision 
@app.route("/api/agents/<agent_id>/action", methods=["POST"])
def submit_action(agent_id):
    body   = request.get_json(force=True)
    action = body.get("action", "").strip()
    if not action:
        return jsonify({"error": "action is required"}), 400

    sess = get_session(agent_id)
    if not sess["role"]:
        return jsonify({"error": "call /init first"}), 400

    role        = sess["role"]
    prev_action = sess["prev_action"]
    counts      = sess["action_counts"]
    # Optional action parameter (free-text, for injection/keyword scanning)
    action_param = body.get("action_param", "")

    # Behaviour Monitor 
   
    b = behaviour_monitor.check(
        role, prev_action, action,
        action_param=action_param,
        agent_id=agent_id
    )

    # Policy Monitor 
   
    p = policy_monitor.check(
        role, prev_action, action, counts,
        action_param=action_param
    )

    # Step 1 — Calculate Risk Score R from B and P first
    B = b.get("score",    0.0)
    P = p.get("severity", 0.0)
    R = owasp_mapper.calculate_risk(B, P)   # R = max(B, P)

    # Step 2 — Map violations to OWASP ASI Top 10 categories
    all_violations = list(p["violations"]) + (b.get("violations") or [])
    o = owasp_mapper.map(all_violations, B, risk_score=R)

    # Trust Score — T = w1×(1-B) + w2×(1-P) + w3×(1-R)
    t = trust_engine.update(
        agent_id, o["risk_score"], action,
        B=B,
        P=P,
        metadata={
            "policy_violations": p["violations"],
            "owasp_categories":  o["category_ids"],
        }
    )

    # Decision (Allow / Warn / Deny) + Log 
    sess["prev_action"]            = action
    sess["action_counts"][action] += 1

    response = {
        # Behaviour Monitor output
        "behavioural_score":      b["score"],
        "behavioural_detail":     b["detail"],
        "behavioural_violations": b.get("violations", []),
        "behavioural_sub_scores": b.get("sub_scores", {}),
        "behavioural_attack_cat": b.get("attack_category", ""),

        # Policy Monitor output
        "policy_severity":        p["severity"],
        "policy_violations":      p["violations"],
        "policy_compliant":       p["compliant"],
        "policy_categories":      p.get("categories", []),

        # OWASP Risk Mapping output
        "owasp_categories":     o["owasp_categories"],
        "owasp_ids":            o["category_ids"],
        "risk_score":           o["risk_score"],
        "risk_level":           o["risk_level"],

        # Trust Score + Decision
        "trust_after":          t["new_trust"],
        "decision":             t["decision"],
        "is_risky":             t["is_risky"],
    }
    sess["log"].append(response | {"action": action, "prev_action": prev_action})
    return jsonify(response)


# Hermes/Ollama 
@app.route("/api/agents/<agent_id>/run", methods=["POST"])
def run_hermes_task(agent_id):
    import requests as req
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL

    body = request.get_json(force=True)
    task = body.get("task", "")
    role = body.get("role", "personal_assistant")

    if not task:
        return jsonify({"error": "task required"}), 400
    if role not in VALID_ROLES:
        return jsonify({"error": f"Unknown role: {role}"}), 400

    # Ollama statue
    try:
        r = req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        return jsonify({
            "error": "Ollama is not running.",
            "fix": "Open a new terminal and run:  ollama serve",
            "then": f"Pull the model:  ollama pull {OLLAMA_MODEL}",
            "detail": str(e)
        }), 503

    #  model status
    if not any(OLLAMA_MODEL in m for m in models):
        return jsonify({
            "error": f"Model '{OLLAMA_MODEL}' not found in Ollama.",
            "fix": f"Run:  ollama pull {OLLAMA_MODEL}",
            "available_models": models
        }), 503

    # Reset session 
    sessions[agent_id] = {
        "role": role, "prev_action": None,
        "action_counts": defaultdict(int),
        "log": [{"action": "__task_started__", "task": task,
                 "behavioural_score": 0, "policy_severity": 0,
                 "policy_violations": [], "owasp_ids": [],
                 "risk_score": 0, "risk_level": "Low",
                 "trust_after": 0.8, "decision": "allow", "is_risky": False}],
    }
    trust_engine.reset(agent_id)

    def _run():
        try:
            from hermes_agent import build_hermes_agent, AgentBlockedError
            executor = build_hermes_agent(agent_id, role, skip_init=True)
            executor.invoke({"input": task})
            sessions[agent_id]["log"].append({
                "action": "__task_completed__",
                "behavioural_score": 0, "policy_severity": 0,
                "policy_violations": [], "owasp_ids": [],
                "risk_score": 0, "risk_level": "Low",
                "trust_after": trust_engine.get(agent_id),
                "decision": "allow", "is_risky": False
            })
        except Exception as e:
            err = str(e)
            is_blocked = any(w in err.lower() for w in
                             ["blocked", "quarantined", "denied", "agentblocked"])
            sessions[agent_id]["log"].append({
                "action": "__task_blocked__" if is_blocked else "__task_error__",
                "error": err,
                "behavioural_score": 0, "policy_severity": 0,
                "policy_violations": [], "owasp_ids": [],
                "risk_score": 0, "risk_level": "Low",
                "trust_after": trust_engine.get(agent_id),
                "decision": "deny" if is_blocked else "allow",
                "is_risky": is_blocked
            })

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "running", "agent_id": agent_id,
                    "model": OLLAMA_MODEL, "task": task})



# LangChain ReAct Agent run  (mirrors /run but uses LangChain)
@app.route("/api/agents/<agent_id>/run_langchain", methods=["POST"])
def run_langchain_task(agent_id):
    import requests as req
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL

    body = request.get_json(force=True)
    task = body.get("task", "")
    role = body.get("role", "personal_assistant")

    if not task:
        return jsonify({"error": "task required"}), 400
    if role not in VALID_ROLES:
        return jsonify({"error": f"Unknown role: {role}"}), 400

    # Check Ollama
    try:
        r = req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        return jsonify({
            "error": "Ollama is not running.",
            "fix":  "Open a new terminal and run:  ollama serve",
            "then": f"Pull the model:  ollama pull {OLLAMA_MODEL}",
            "detail": str(e)
        }), 503

    if not any(OLLAMA_MODEL in m for m in models):
        return jsonify({
            "error": f"Model '{OLLAMA_MODEL}' not found in Ollama.",
            "fix":   f"Run:  ollama pull {OLLAMA_MODEL}",
            "available_models": models
        }), 503

    # Reset session
    sessions[agent_id] = {
        "role": role, "prev_action": None,
        "action_counts": defaultdict(int),
        "log": [{"action": "__task_started__", "task": task,
                 "behavioural_score": 0, "policy_severity": 0,
                 "policy_violations": [], "owasp_ids": [],
                 "risk_score": 0, "risk_level": "Low",
                 "trust_after": 0.8, "decision": "allow", "is_risky": False}],
    }
    trust_engine.reset(agent_id)

    def _run():
        executor = None
        try:
            from langchain_agent import build_langchain_agent, AgentBlockedError
            executor = build_langchain_agent(agent_id, role, skip_init=True)
            executor.invoke({"input": task})
            # Sync pipeline results from langchain state → session log
            if executor is not None:
                for entry in executor._state.log:
                    sessions[agent_id]["log"].append(entry)
            sessions[agent_id]["log"].append({
                "action": "__task_completed__",
                "behavioural_score": 0, "policy_severity": 0,
                "policy_violations": [], "owasp_ids": [],
                "risk_score": 0, "risk_level": "Low",
                "trust_after": trust_engine.get(agent_id),
                "decision": "allow", "is_risky": False
            })
        except Exception as e:
            err = str(e)
            is_blocked = any(w in err.lower() for w in
                             ["blocked", "quarantined", "denied", "agentblocked"])
            # Sync whatever pipeline results were recorded before the block
            last_trust = 0.8  # fallback
            if executor is not None:
                for entry in executor._state.log:
                    sessions[agent_id]["log"].append(entry)
                    if "trust_after" in entry:
                        last_trust = entry["trust_after"]
            sessions[agent_id]["log"].append({
                "action": "__task_blocked__" if is_blocked else "__task_error__",
                "error":  err,
                "behavioural_score": 0, "policy_severity": 0,
                "policy_violations": [], "owasp_ids": [],
                "risk_score": 0, "risk_level": "Low",
                "trust_after": last_trust,
                "decision": "deny" if is_blocked else "allow",
                "is_risky": is_blocked
            })

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "running", "agent_id": agent_id,
                    "model": OLLAMA_MODEL, "task": task,
                    "agent_type": "langchain"})


# Live output log  (works for both Hermes and LangChain)
@app.route("/api/agents/<agent_id>/output", methods=["GET"])
def agent_output(agent_id):
    """Returns the live thought/action log for the browser panel."""
    agent_type = request.args.get("agent_type", "hermes")
    try:
        if agent_type == "langchain":
            from langchain_agent import get_output_log
        else:
            from hermes_agent import get_output_log
        return jsonify(get_output_log(agent_id))
    except Exception:
        return jsonify([])

if __name__ == "__main__":
    print("\n  Trust-Aware & Policy-Driven Anomaly Detection System")
    print("   OWASP ASI Top 10 Risk Mapping enabled")
    print("   Dashboard → http://127.0.0.1:5051\n")
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=5051, debug=True, use_reloader=False)

