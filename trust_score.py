"""
trust_score.py  —  Stage 4: Weighted Trust Score Engine
========================================================
Formula:
    T = w1×(1−B) + w2×(1−P) + w3×(1−R)

Source:
    George, A.T., Neve, H.R., & Muraleedharan, N. (2023).
    A Trust Score calculation approach for Zero Trust Access System.
    Annual IEEE India Conference (INDICON 2023), pp. 392–397. IEEE.

    The paper defines: TS = Σ(pi×wi) / Σ(pi).
    With bounded [0,1] factors and weights summing to 1, the
    denominator = 1 and the formula reduces to the weighted sum above.

Complement transformation (1−score):
    Jøsang, A., & Lo Presti, S. (2004). Analysing the Relationship
    Between Risk and Trust. iTrust 2004, LNCS 2995, Springer.
    Risk and trust are inversely related; (1−score) converts each
    risk/anomaly score to a trust contribution.

Weight ordering:
    NIST AI 600-1 (2024) — behavioural monitoring is the primary
    control mechanism for agentic AI, justifying w1 > w2 > w3.

Decision bands (CVSS v3.1, FIRST.org 2023, normalised to [0,1]):
    ALLOW  T ≥ 0.70   (High/Critical boundary: CVSS 7.0/10)
    WARN   T ≥ 0.40   (Low/Medium boundary:    CVSS 4.0/10)
    DENY   T < 0.40
"""

import os, json, time

W_BEHAVIOUR = 0.40
W_POLICY = 0.35
W_OWASP = 0.25
assert abs(W_BEHAVIOUR + W_POLICY + W_OWASP - 1.0) < 1e-9

BAND_ALLOW = 0.70
BAND_WARN = 0.40
RISK_THRESHOLD = 0.45

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def compute_trust(B: float, P: float, R: float) -> float:
    B = max(0.0, min(1.0, float(B)))
    P = max(0.0, min(1.0, float(P)))
    R = max(0.0, min(1.0, float(R)))
    return round(W_BEHAVIOUR * (1 - B) + W_POLICY * (1 - P) + W_OWASP * (1 - R), 4)


def decide(T: float) -> str:
    if T >= BAND_ALLOW:
        return "allow"
    if T >= BAND_WARN:
        return "warn"
    return "deny"


class TrustScoreEngine:
    def __init__(self):
        self._history: dict[str, list[dict]] = {}
        self._last: dict[str, dict] = {}
        self._parent: dict[str, str] = {}
        os.makedirs(LOG_DIR, exist_ok=True)

    def get(self, agent_id: str) -> float:
        last = self._last.get(agent_id)
        return last["new_trust"] if last else 1.0

    def reset(self, agent_id: str):
        self._history.pop(agent_id, None)
        self._last.pop(agent_id, None)

    def register_child(self, child_id: str, parent_id: str):
        """Trust inheritance for multi-agent systems (OWASP ASI-07)."""
        self._parent[child_id] = parent_id
        parent_T = self.get(parent_id)
        self._last[child_id] = {
            "new_trust": parent_T,
            "decision": decide(parent_T),
            "B": 0.0,
            "P": 0.0,
            "R": 0.0,
        }

    def get_lineage_trust(self, agent_id: str) -> float:
        min_T, cur = self.get(agent_id), agent_id
        while cur in self._parent:
            cur = self._parent[cur]
            min_T = min(min_T, self.get(cur))
        return round(min_T, 4)

    def update(
        self,
        agent_id: str,
        risk_score: float,
        action: str,
        B: float = 0.0,
        P: float = 0.0,
        metadata: dict | None = None,
    ) -> dict:
        R = max(0.0, min(1.0, float(risk_score)))
        B = max(0.0, min(1.0, float(B)))
        P = max(0.0, min(1.0, float(P)))
        prev_T = self._last.get(agent_id, {}).get("new_trust", 1.0)

        cB = round(W_BEHAVIOUR * (1 - B), 4)
        cP = round(W_POLICY * (1 - P), 4)
        cR = round(W_OWASP * (1 - R), 4)
        T = round(cB + cP + cR, 4)
        dec = decide(T)

        # Trust inheritance cap (OWASP ASI-07)
        if agent_id in self._parent:
            parent_T = self.get(self._parent[agent_id])
            if T > parent_T:
                T = parent_T
                dec = decide(T)

        entry = {
            "ts": agent_id,
            "agent_id": agent_id,
            "action": action,
            "B": round(B, 4),
            "P": round(P, 4),
            "R": round(R, 4),
            "contrib_B": cB,
            "contrib_P": cP,
            "contrib_R": cR,
            "risk_score": round(R, 4),
            "prev_trust": round(prev_T, 4),
            "new_trust": T,
            "decision": dec,
            "is_risky": R >= RISK_THRESHOLD,
            **(metadata or {}),
        }
        self._history.setdefault(agent_id, []).append(entry)
        self._last[agent_id] = entry
        try:
            with open(os.path.join(LOG_DIR, "audit.jsonl"), "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
        return entry

    def history(self, agent_id: str) -> list[dict]:
        return self._history.get(agent_id, [])

    def all_agents(self) -> dict:
        return {
            aid: {
                "trust": last["new_trust"],
                "band": last["decision"],
                "B": last["B"],
                "P": last["P"],
                "R": last["R"],
            }
            for aid, last in self._last.items()
        }
