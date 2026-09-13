# Intent Taxonomy — 8 Classes (Twitter Support)

Derived from 150-tweet pilot of AmazonHelp/AppleSupport + Banking77 inspiration. Keep to 7+other for reliable κ >0.6.

| # | Intent | Definition | Includes | Excludes | Example |
|---|---|---|---|---|---|
| 1 | `order_shipping` | Where is order, delivery delay, tracking | "where is my order #123", tracking no update, delivered to wrong address | Refund after delivery → refund_billing | "My order #12345 hasn't arrived, tracking shows no update for 5 days" |
| 2 | `refund_billing` | Charges, double charge, refund status, invoice | double charged, refund not received, billing error | Cancel before ship → cancellation | "I was charged twice, please refund $49" |
| 3 | `account_access` | Login, password, hacked, 2FA, locked | can't log in, password reset fails, account hacked, email changed | General app crash → product_technical | "Can't log in, says password incorrect after reset, account hacked" |
| 4 | `product_technical` | Device/app/bug, crash, setup, defect | phone won't turn on, app crashes on pay, broken item, update fails | Delivery damage → order_shipping | "My phone keeps restarting after update, won't turn on" |
| 5 | `cancellation_return` | Cancel order, return, exchange, stop shipment | cancel order #987, return request, want exchange | Refund status after return → refund_billing | "How do I cancel order #98765? Need to stop shipment" |
| 6 | `complaint_escalation` | Legal threat, manager request, severe dissatisfaction, repeat failure | "legal action", "speak to manager", "terrible service" repeated, "considering sue" | Single delay without threat → order_shipping | "Your support is terrible, want manager, considering legal action" |
| 7 | `information_request` | Policy, shipping to country, store hours, how-to | "do you ship to Canada?", "what's delivery time?", "how to track?" | Specific order → order_shipping | "Do you ship to Canada? What's delivery time?" |
| 8 | `other` | Gratitude, spam, vague, out-of-scope | "thanks!", "love service", unclear single word | — | "Thank you so much! Love it" |

## Labeling Rules
1. **Single label only** — pick highest urgency. If multi-intent e.g., "refund and change address", choose `refund_billing` (money > logistics) and note in `notes`.
2. **Use thread context**: if `prev_turn` shows prior brand asked for order ID and user now provides it → keep original intent, not information_request.
3. **Complaint override**: any legal/threat/manager language → `complaint_escalation` regardless of underlying issue.
4. **Low-confidence**: if <55% sure, mark `other` and note uncertainty — these should escalate.
5. **Language**: non-English → `other` (no langdetect needed, just use intent field).

## Escalation Reasons (for router)
* `low_confidence` — max_prob <0.55 or annotator low_conf
* `high_risk` — complaint_escalation intent
* `sensitive` — account_access hacked/2FA, billing fraud keywords
* `repeat` — thread_len >4 or "again", "still not"
* `policy` — legal threat, explicit human request

## Examples per Intent (3 each)
See `data/golden/golden.jsonl` — first 24 cover anchors.
