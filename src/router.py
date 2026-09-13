"""
Escalation router — multi-signal, rule-based with confidence + intent + keywords.

Policy: escalate if ANY of:
 - max_prob < 0.55 (low_confidence)
 - intent == complaint_escalation (high_risk) OR intent==account_access with hacked/2fa keywords (sensitive)
 - regex(threat|legal|lawyer|fraud|police|sue|lawsuit|attorney) (policy)
 - thread_len >4 or "again"/"still not" (repeat)
 - explicit human request ("human","agent","manager","real person","talk to someone")

Returns (escalate:bool, reason:str, reason_category:str)
"""
import re
from typing import Tuple

ESCALATE_KEYWORDS = re.compile(r"\b(legal|lawyer|sue|lawsuit|attorney|fraud|police|threat|manager|human|agent|real person|talk to someone|escalate)\b", re.I)
SENSITIVE_RE = re.compile(r"\b(hacked|2fa|unauthorized|stolen|fraud|verify identity)\b", re.I)
REPEAT_RE = re.compile(r"\b(again|still not|already|multiple times|2nd time|second time|third time)\b", re.I)

def decide(text: str, intent: str, confidence: float, thread_len: int = 1, prev_turn: str = "") -> Tuple[bool, str, str]:
    t = (text or "") + " " + (prev_turn or "")
    tl = t.lower()
    # 1. explicit human request
    if re.search(r"\b(human|real person|talk to (a )?human|agent please|speak to manager)\b", tl):
        return True, "Customer explicitly requested human agent", "policy"
    # 2. high-risk intent
    if intent == "complaint_escalation":
        return True, "Complaint/escalation intent — high risk, needs human empathy", "high_risk"
    # 3. low confidence
    if confidence < 0.55:
        return True, f"Low intent confidence ({confidence:.2f} < 0.55) — ambiguous, risk of wrong auto-reply", "low_confidence"
    # 4. policy keywords (legal etc) even if intent not complaint
    if ESCALATE_KEYWORDS.search(t):
        # limit: only escalate if not already high_risk, and keyword is legal/fraud
        if re.search(r"\b(legal|lawyer|sue|lawsuit|attorney|fraud|police)\b", tl):
            return True, "Sensitive legal/fraud language detected — policy requires human review", "policy"
    # 5. sensitive account
    if intent == "account_access" and SENSITIVE_RE.search(t):
        return True, "Sensitive account issue (hacked/fraud/2FA) — requires human verification", "sensitive"
    # 6. repeat / long thread
    if thread_len > 4:
        return True, f"Long thread (len {thread_len} >4) — likely unresolved, needs human", "repeat"
    if REPEAT_RE.search(tl):
        return True, "Repeat-contact language ('again'/'still not') — needs human to break loop", "repeat"
    # default auto
    return False, "Confident intent, no risk signals — safe to auto-handle", "auto"

# legacy keyword-only baseline for comparison
def decide_keyword_only(text: str, intent: str, confidence: float, **kw) -> Tuple[bool,str,str]:
    if ESCALATE_KEYWORDS.search(text.lower() + " " + intent):
        return True, "Keyword match", "policy"
    return False, "No keyword", "auto"
