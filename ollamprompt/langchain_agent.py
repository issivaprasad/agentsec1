"""
langchain_agent.py — LangChain ReAct Agent with Trust Pipeline integration
===========================================================================
Drop-in companion to hermes_agent.py.  Mirrors its public interface so
app.py can run either agent transparently:

    from langchain_agent import build_langchain_agent, get_output_log, AgentBlockedError

Every tool call is intercepted and run through the same five-stage pipeline:

    Stage 1  BehaviourMonitor.check()              → B score
    Stage 2  PolicyMonitor.check()                 → P score
    Stage 3  OWASPRiskMapper.calculate_risk(B,P)   → R = max(B,P)
    Stage 4  OWASPRiskMapper.map()                 → ASI-XX tags
    Stage 5  TrustScoreEngine.update()             → T + ALLOW/WARN/DENY

No session_state dependency — all state is managed inside _AgentState,
one instance per agent_id, stored in _AGENT_STATES.

Requirements
------------
    pip install langchain langchain-community langchain-core langchain-ollama
    # Ollama must be running:  ollama serve  &&  ollama pull nous-hermes2
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── LangChain ─────────────────────────────────────────────────────────────────
try:
    from langchain_community.llms import Ollama
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
    from langchain.prompts import PromptTemplate
    _LC_OK = True
except ImportError:
    _LC_OK = False

# ── Pipeline components (same directory) ──────────────────────────────────────
from behaviour_monitor import build_behaviour_monitor
from policy_monitor    import PolicyMonitor
from owasp_mapper      import OWASPRiskMapper
from trust_score       import TrustScoreEngine

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
except ImportError:
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL    = "nous-hermes2"
    OLLAMA_TIMEOUT  = 120

# XEPV action-type mapping for the eight standard tools
_TOOL_XEPV: Dict[str, str] = {
    "web_search":  "X",
    "read_file":   "X",
    "write_file":  "E",
    "bash":        "E",
    "send_email":  "E",
    "run_tests":   "V",
    "reason":      "P",
    "finish":      "V",
}


# ══════════════════════════════════════════════════════════════════════════════
# Per-agent output log  (mirrors hermes_agent.get_output_log interface)
# ══════════════════════════════════════════════════════════════════════════════

_OUTPUT_LOGS: Dict[str, List[Dict]] = {}
_LOG_LOCK = threading.Lock()


def _log(agent_id: str, kind: str, text: str) -> None:
    with _LOG_LOCK:
        if agent_id not in _OUTPUT_LOGS:
            _OUTPUT_LOGS[agent_id] = []
        _OUTPUT_LOGS[agent_id].append({"type": kind, "content": text,
                                        "ts": time.time()})


def get_output_log(agent_id: str) -> List[Dict]:
    """Return the live output log for agent_id (same signature as hermes_agent)."""
    with _LOG_LOCK:
        return list(_OUTPUT_LOGS.get(agent_id, []))


# ══════════════════════════════════════════════════════════════════════════════
# Exception (same name as hermes_agent so app.py catches both)
# ══════════════════════════════════════════════════════════════════════════════

class AgentBlockedError(RuntimeError):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Module-level pipeline singletons (shared across agents, thread-safe reads)
# ══════════════════════════════════════════════════════════════════════════════

_bm   = build_behaviour_monitor()
_pm   = PolicyMonitor()
_om   = OWASPRiskMapper()
_te   = TrustScoreEngine()
_pipe_lock = threading.Lock()


def _run_pipeline(
    agent_id:    str,
    role:        str,
    prev_action: Optional[str],
    action:      str,
    action_param: str,
    counts:      Dict[str, int],
) -> Dict[str, Any]:
    """Run all five pipeline stages and return the combined result dict."""
    with _pipe_lock:
        b = _bm.check(role, prev_action, action,
                      action_param=action_param, agent_id=agent_id)
        p = _pm.check(role, prev_action, action, counts,
                      action_param=action_param)

        B = b.get("score",    0.0)
        P = p.get("severity", 0.0)
        R = _om.calculate_risk(B, P)

        all_viol = list(p["violations"]) + (b.get("violations") or [])
        o = _om.map(all_viol, B, risk_score=R)

        t = _te.update(
            agent_id, o["risk_score"], action,
            B=B, P=P,
            metadata={
                "policy_violations": p["violations"],
                "owasp_categories":  o["category_ids"],
            },
        )

    return {
        "behavioural_score":      B,
        "behavioural_detail":     b.get("detail", ""),
        "behavioural_violations": b.get("violations", []),
        "behavioural_attack_cat": b.get("attack_category", ""),
        "policy_severity":        P,
        "policy_violations":      p["violations"],
        "policy_compliant":       p["compliant"],
        "policy_categories":      p.get("categories", []),
        "owasp_categories":       o["owasp_categories"],
        "owasp_ids":              o["category_ids"],
        "risk_score":             o["risk_score"],
        "risk_level":             o["risk_level"],
        "trust_after":            t["new_trust"],
        "decision":               t["decision"],
        "is_risky":               t["is_risky"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ReAct prompt  (Nous-Hermes2 tuned)
# ══════════════════════════════════════════════════════════════════════════════

_REACT_TEMPLATE = """\
You are a helpful AI assistant. Complete the task using the available tools.

Tools available:
{tools}

Tool names: {tool_names}

Task: {input}

Use this format EXACTLY:
Thought: <your reasoning>
Action: <tool name from [{tool_names}]>
Action Input: <input for the tool>
Observation: <result of the tool>
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have the final answer.
Final Answer: <your answer>

Begin!
{agent_scratchpad}"""


# ══════════════════════════════════════════════════════════════════════════════
# Tool implementations  (stubs — same as hermes_agent's tool set)
# ══════════════════════════════════════════════════════════════════════════════

def _web_search(query: str) -> str:
    return f"[web_search stub] Results for: {query}"

def _read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()[:4000]
    except Exception as e:
        return f"[read_file error] {e}"

def _write_file(arg: str) -> str:
    try:
        parts = arg.split("|", 1)
        path, content = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 \
                        else (arg.strip(), "")
        with open(path, "w") as f:
            f.write(content)
        return f"[write_file] Written to {path}"
    except Exception as e:
        return f"[write_file error] {e}"

def _bash(cmd: str) -> str:
    return f"[bash stub] Would execute: {cmd}"

def _send_email(arg: str) -> str:
    return f"[send_email stub] Would send: {arg}"

def _run_tests(arg: str) -> str:
    return f"[run_tests stub] Would test: {arg}"

def _reason(arg: str) -> str:
    return f"[reason] {arg}"

def _finish(arg: str) -> str:
    return f"[finish] {arg}"

_TOOL_IMPLS: Dict[str, Callable[[str], str]] = {
    "web_search": _web_search,
    "read_file":  _read_file,
    "write_file": _write_file,
    "bash":       _bash,
    "send_email": _send_email,
    "run_tests":  _run_tests,
    "reason":     _reason,
    "finish":     _finish,
}


# ══════════════════════════════════════════════════════════════════════════════
# build_langchain_agent  — public factory (mirrors build_hermes_agent)
# ══════════════════════════════════════════════════════════════════════════════

def build_langchain_agent(
    agent_id:  str,
    role:      str,
    skip_init: bool = True,   # kept for API parity; always True here
) -> "LangChainExecutorWrapper":
    """Return a wrapper with an .invoke({"input": task}) method."""
    if not _LC_OK:
        raise ImportError(
            "LangChain is not installed. "
            "Run: pip install langchain langchain-community langchain-core langchain-ollama"
        )
    return LangChainExecutorWrapper(agent_id=agent_id, role=role)


# ══════════════════════════════════════════════════════════════════════════════
# Per-agent mutable state (no global session_state)
# ══════════════════════════════════════════════════════════════════════════════

class _AgentState:
    def __init__(self, agent_id: str, role: str):
        self.agent_id    = agent_id
        self.role        = role
        self.prev_action: Optional[str] = None
        self.counts: Dict[str, int]     = defaultdict(int)
        self.log: List[Dict]            = []

    def record(self, result: Dict, action: str) -> None:
        entry = dict(result, action=action, prev_action=self.prev_action)
        self.log.append(entry)
        self.prev_action = action
        self.counts[action] += 1


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline-intercepting tool wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _make_secure_tool(
    name:   str,
    impl:   Callable[[str], str],
    state:  _AgentState,
) -> Tool:
    """Wrap a tool function so every call passes through the 5-stage pipeline."""

    def _secure_call(arg: str) -> str:
        result = _run_pipeline(
            agent_id=state.agent_id,
            role=state.role,
            prev_action=state.prev_action,
            action=name,
            action_param=arg,
            counts=state.counts,
        )
        state.record(result, name)

        trust    = result["trust_after"]
        decision = result["decision"].lower()
        owasp    = ", ".join(result["owasp_ids"]) or "none"
        _log(state.agent_id, "trust",
             f"[{name}] T={trust:.3f} B={result['behavioural_score']:.3f} "
             f"P={result['policy_severity']:.3f} R={result['risk_score']:.3f} "
             f"decision={decision.upper()} owasp=[{owasp}]")

        if decision == "deny":
            msg = (f"[BLOCKED] Action '{name}' denied by trust pipeline "
                   f"(T={trust:.3f}, R={result['risk_score']:.3f}). "
                   f"OWASP: {owasp}")
            _log(state.agent_id, "blocked", msg)
            raise AgentBlockedError(msg)

        if decision == "warn":
            _log(state.agent_id, "warning",
                 f"[WARN] '{name}' allowed with caution (T={trust:.3f})")

        _log(state.agent_id, "tool_call", f"→ {name}({arg[:120]})")
        out = impl(arg)
        _log(state.agent_id, "result", f"← {str(out)[:300]}")
        return out

    return Tool(name=name, func=_secure_call,
                description=f"Secure tool: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Executor wrapper  (has .invoke() like LangChain AgentExecutor)
# ══════════════════════════════════════════════════════════════════════════════

class LangChainExecutorWrapper:
    """Wraps LangChain AgentExecutor; exposes .invoke({"input": task})."""

    def __init__(self, agent_id: str, role: str):
        self._agent_id = agent_id
        self._role     = role
        self._state    = _AgentState(agent_id=agent_id, role=role)

        # Reset trust engine for this agent
        _te.reset(agent_id)

        # Build pipeline-wrapped tools
        tools = [
            _make_secure_tool(name, impl, self._state)
            for name, impl in _TOOL_IMPLS.items()
        ]

        # LLM
        llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.2,
            num_predict=512,
        )

        # ReAct agent
        prompt = PromptTemplate.from_template(_REACT_TEMPLATE)
        react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        self._executor = AgentExecutor(
            agent=react_agent,
            tools=tools,
            max_iterations=10,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        task = inputs.get("input", "")
        _log(self._agent_id, "task", f"Task: {task}")
        _log(self._agent_id, "system",
             f"LangChain ReAct agent | role={self._role} | model={OLLAMA_MODEL}")
        try:
            result = self._executor.invoke({"input": task})
            final  = result.get("output", "")
            _log(self._agent_id, "final", f"Final Answer: {final}")
            return result
        except AgentBlockedError:
            raise
        except Exception as e:
            _log(self._agent_id, "error", f"Error: {e}")
            raise

    def get_log(self) -> List[Dict]:
        return get_output_log(self._agent_id)
