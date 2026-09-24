import json, requests

URL = "http://127.0.0.1:8765/v1/systemone"

ticket = {
    "id": "TKT-48812",
    "plan": "Team (12 seats)",
    "body": "We were billed 480 USD twice on 3 September for "
            "the same 12 seats. I have now written three times "
            "and the only reply was an automated one. Our "
            "finance close is on Wednesday and I need the "
            "duplicate reversed before then.",
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should own this ticket?",
        "criteria": {
            "billing": "payments, invoices, refunds, charges",
            "technical_support": "the product is broken or down",
            "sales": "new seats, upgrades, quotes, renewals",
            "product_questions": "how-to questions, feature requests",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "This ticket needs a reply today.",
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound?",
        "criteria": [
            "calm and matter-of-fact",
            "impatient: chasing a reply or repeating themselves",
            "angry: threatening to escalate, churn or dispute",
        ],
    },
}

r = requests.post(URL, json={"state": {"ticket": ticket},
                             "questions": questions})
print(json.dumps(r.json(), indent=2))
