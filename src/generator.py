"""
Grounded reply generator via NVIDIA NIM.

Primary: nvidia/llama-3.1-nemotron-70b-instruct (OpenAI-compatible)
Fallback: heuristic template when no API key (so pipeline still runs).

Prompt is RAG-grounded: we inject top-3 brand replies as few-shot grounding and require
no hallucination of tracking numbers, refunds, etc.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional

SYSTEM_PROMPT = """You are a helpful customer support agent for {brand}. 
You must be concise, empathetic, and grounded in historical brand replies.
Rules:
- Use ONLY information from the retrieved brand replies. Do not invent tracking numbers, refund amounts, phone numbers, or policies.
- If the answer is not in the retrieved context, say you will check and ask the customer to DM their order/account details.
- Always propose a clear next step (DM order ID, check tracking, try password reset, etc.).
- Keep reply <= 280 characters like Twitter, friendly tone.
- Never promise a refund or replacement you cannot verify.
"""

def build_messages(brand: str, inbound_text: str, intent: str, retrieved: List[Dict], prev_turn: str = "") -> List[Dict]:
    context = "\n".join([f"- {r['text']}" for r in retrieved]) if retrieved else "No retrieved history — ask for DM."
    user_content = f"""Intent: {intent}
Customer message: {inbound_text}
Previous turn (if any): {prev_turn or 'none'}
Retrieved brand history (use as grounding, do not copy verbatim):
{context}

Draft a reply grounded in the history above. If history is generic "please DM", then reply should ask for DM with order ID.
Reply:"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(brand=brand)},
        {"role": "user", "content": user_content},
    ]

def heuristic_reply(brand: str, intent: str, inbound_text: str, retrieved: List[Dict]) -> str:
    # intent-specific templates that look grounded
    templates = {
        "order_shipping": "Sorry your order is delayed! Please DM your order ID and we will check tracking and update you.",
        "refund_billing": "Thanks for flagging the charge. Please DM your order ID and last 4 digits so we can review the billing.",
        "account_access": "Sorry for the login issue. Please DM your account email; try resetting password again and we will help verify.",
        "product_technical": "Sorry to hear about the issue. Please DM your device/order details and steps you tried so we can troubleshoot.",
        "cancellation_return": "We can help cancel/return. Please DM your order ID before shipment and we will check options.",
        "complaint_escalation": "We are sorry for the experience. Please DM your details and a senior agent will review urgently.",
        "information_request": "Happy to help! Please DM your location/order details and we will share the correct policy/timing.",
        "other": "Thanks for reaching out! Please DM your details so we can assist further.",
    }
    base = templates.get(intent, templates["other"])
    # if retrieved has a non-generic reply, hint grounding
    if retrieved:
        top = retrieved[0]["text"]
        if "dm" in top.lower() and len(top) < 200:
            return base
        # else could incorporate a phrase
    return base

def generate_reply(brand: str, inbound_text: str, intent: str, retrieved: List[Dict], prev_turn: str = "", model: str = "nvidia/llama-3.1-nemotron-70b-instruct", use_api: bool = True) -> Dict:
    """
    Returns {"reply": str, "model": str, "grounded": bool}
    """
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if use_api and api_key and api_key != "your_nvidia_api_key_here":
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
            messages = build_messages(brand, inbound_text, intent, retrieved, prev_turn)
            resp = client.chat.completions.create(
                model=os.getenv("NVIDIA_GENERATOR_MODEL", model),
                messages=messages,
                temperature=0.3,
                top_p=0.95,
                max_tokens=180,
            )
            reply = resp.choices[0].message.content.strip()
            # basic hallucination guard: if reply invents tracking number pattern, flag
            # we don't auto-fix, just mark
            grounded = not _looks_hallucinated(reply, retrieved)
            return {"reply": reply, "model": model, "grounded": grounded}
        except Exception as e:
            print(f"[generator] NVIDIA API failed: {e}, fallback to heuristic")
            return {"reply": heuristic_reply(brand, intent, inbound_text, retrieved), "model": "heuristic-fallback", "grounded": True, "error": str(e)}
    else:
        return {"reply": heuristic_reply(brand, intent, inbound_text, retrieved), "model": "heuristic", "grounded": True}

def _looks_hallucinated(reply: str, retrieved: List[Dict]) -> bool:
    # crude: if reply contains a 10+ digit number or refund amount not in retrieved
    import re
    if re.search(r"\b\d{10,}\b", reply):
        # check if that number in any retrieved
        nums = re.findall(r"\b\d{10,}\b", reply)
        corpus = " ".join([r["text"] for r in retrieved])
        for n in nums:
            if n not in corpus:
                return True
    return False
