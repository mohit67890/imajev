"""Example requests for the playground's "Load example" menu (docs/playground-spec.md).

Each example carries repository-relative image paths so the server can load the images itself;
`GET /examples/{n}/image/{i}` serves them to the UI. `load_examples()` drops any example whose
images are missing from the checkout rather than advertising a broken menu entry.

Every question here was checked against the v1 adapter (see "verified" notes): the menu is meant to
show the model's real behaviour, so an example either gets the answer right or exists to show an
abstention. Two things the v1 model is known to be shaky on are deliberately avoided: score scales
for subjective photo quality with fewer than 4 levels (it puts ~all mass on unknown, because the
training data only ever used a 5-point scale), and asking which state field a *second* image
contradicts (it answers that correctly from one image and flips with two).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STATE_AWARE = "data/decision-v1/state_aware/images"
TALLYQA = "data/decision-v1/tallyqa/images"
ABSTAIN = "data/decision-v1/heldout_abstention/images"
TEXTVQA = "data/decision-v1/textvqa/images"

CUTLERY = f"{STATE_AWARE}/21e6886fae2b31f208de0150a03c9d46af5fed5e6904d50e03f8107c99880278.jpg"
OXFORD = f"{STATE_AWARE}/a0e8fbdd569a7f7c0cec0aafb0c6bf63df05f50ba02a9814261f7d7bf07783cd.jpg"
SNEAKER = f"{STATE_AWARE}/1b3ac44bf8066c9894fefcf53af026a030e61b04280cf2f3386dcbde601f0a87.jpg"

# Photo-quality scores need a 5-level scale; on 3 levels the model abstains with unknown ≈ 0.99.
USABILITY = ["unusable: blurry, dark or badly cropped", "poor", "acceptable", "good",
             "excellent: sharp, well lit, product fills the frame"]

EXAMPLES = [
    {
        "name": "Product listing check",
        "description": "One product photo against a marketplace listing that is correct: colour match, category, photo usability. Verified: true 0.99, cutlery 0.97, usability 2.5 of 4.",
        "images": [CUTLERY],
        "request": {
            "state": {"listing": {"title": "24 Piece Stainless Steel Cutlery Set, Stripes (Contains: 6 Table Spoons, 6 Tea Spoons, 6 Forks, 6 Soup Spoons)", "color": "silver", "product_type": "cutlery"}},
            "questions": {
                "color_matches": {
                    "type": "noul",
                    "instructions": "The product in the photo matches `listing.color`.",
                    "criteria": {"true": "the dominant colour of the product is the listed colour",
                                 "false": "the product is clearly a different colour from the listed one"},
                },
                "product_type": {
                    "type": "choice",
                    "instructions": "Which category does the photographed item belong to?",
                    "criteria": {"cutlery": "knives, forks, spoons", "cookware": "pots and pans", "tableware": "plates, bowls, glasses", "other": None},
                },
                "photo_usability": {
                    "type": "score",
                    "instructions": "How usable is this photo as the main listing image?",
                    "criteria": USABILITY,
                },
            },
        },
    },
    {
        "name": "Listing field contradiction",
        "description": "The same check on a listing that is wrong: the shoe in the photo is black, the listing says brown. Verified: false 0.02, and the choice picks `listing.color` at 0.98.",
        "images": [SNEAKER],
        "request": {
            "state": {"listing": {"title": "Men's Lace-Up Shoes", "color": "brown", "product_type": "shoe"}},
            "questions": {
                "color_matches": {
                    "type": "noul",
                    "instructions": "The product in the photo matches `listing.color`.",
                    "criteria": {"true": "the product is the listed colour",
                                 "false": "the product is clearly a different colour from the listed one"},
                },
                "contradicted_field": {
                    "type": "choice",
                    "instructions": "Which listed field does this photo contradict?",
                    "criteria": {"listing.color": "the photo shows a different colour than `listing.color`",
                                 "listing.product_type": "the photo shows a different kind of product than `listing.product_type`",
                                 "none of them": "the photo is consistent with the listing"},
                },
            },
        },
    },
    {
        "name": "Two-image same-product",
        "description": "Reference listing photo vs a second photo of a different shoe: both questions reason across the two images. Verified: same_item 0.00, the first photo 0.66, black 0.60 — note how much lower the confidence is than on one image.",
        "images": [OXFORD, SNEAKER],
        "request": {
            "state": {"listing": {"title": "Men's Formal Oxford Shoes", "color": "brown", "product_type": "shoe"}},
            "questions": {
                "same_item": {
                    "type": "noul",
                    "instructions": "The second photo shows the item described in `listing`.",
                    "criteria": {"true": "the second photo shows the same product as the listing and the reference photo",
                                 "false": "the second photo shows a different product"},
                },
                "matching_photo": {
                    "type": "choice",
                    "instructions": "Which photo shows the product described in `listing`?",
                    "criteria": {"the first photo": None, "the second photo": None,
                                 "both photos": None, "neither photo": None},
                },
                "second_color": {
                    "type": "choice",
                    "instructions": "What is the dominant colour of the product in the second photo?",
                    "criteria": {"brown": None, "black": None, "white": None, "other": None},
                },
            },
        },
    },
    {
        "name": "Counting as score",
        "description": "An ordinal question used as a counter: the score type gives a distribution over 0-6 people. Verified: 3.01, with 0.78 on 'three people'.",
        "images": [f"{TALLYQA}/dcdc070d27438efb56ae31076174113440103634877d074c9ac5083a5df70bd6.jpg"],
        "request": {
            "state": {},
            "questions": {
                "people": {
                    "type": "score",
                    "instructions": "How many people are visible in the photo?",
                    "criteria": ["no people", "one person", "two people", "three people",
                                 "four people", "five people", "six or more people"],
                },
            },
        },
    },
    {
        "name": "Abstention showcase",
        "description": "Two questions should abstain: there is no dog (false premise) and the shirt colour the model reads, green, is not among the options (not listed). Bag colour and mood are answerable controls. Add an 'other' option to shirt_color to see the model pick it (0.98) instead of abstaining.",
        "images": [f"{ABSTAIN}/dc020c1f7b783a0fbd6d518b0e5e021a5a735312bd9a3b14b940ce8ca02adeb0.jpg"],
        "request": {
            "state": {},
            "questions": {
                "dog_color": {
                    "type": "choice",
                    "instructions": "What colour is the dog standing next to him?",
                    "criteria": {"red": None, "blue": None, "black": None, "white": None},
                },
                "shirt_color": {
                    "type": "choice",
                    "instructions": "What colour is his shirt?",
                    "criteria": {"red": None, "blue": None, "black": None},
                },
                "bag_color": {
                    "type": "choice",
                    "instructions": "What colour is the bag?",
                    "criteria": {"purple": None, "green": None, "gray": None, "black": None},
                },
                "mood": {
                    "type": "choice",
                    "instructions": "Which mood does the figure convey?",
                    "criteria": {"sad": None, "angry": None, "happy": None},
                },
            },
        },
    },
    {
        "name": "Document / scene text",
        "description": "A held-out TextVQA photo: a choice over candidate OCR tokens plus a legibility score. Verified: palm 1.00, legibility 3.3 of 4.",
        # TextVQA's val split, so the answer is read off the photo rather than remembered from training.
        "images": [f"{TEXTVQA}/1739daa601540017c980dce24fadae37fe9f2368790df2ae232854c0d39f2ac3.jpg"],
        "request": {
            "state": {"task": "read the branding on the device"},
            "questions": {
                "brand": {
                    "type": "choice",
                    "instructions": "Which brand name is printed on the device?",
                    "criteria": {"palm": None, "sony": None, "nokia": None, "tokyo bullet": None},
                },
                "text_legible": {
                    "type": "score",
                    "instructions": "How legible is the branding in this photo?",
                    "criteria": ["illegible", "mostly illegible", "partly legible", "mostly legible", "fully legible"],
                },
            },
        },
    },
    {
        "name": "Text-only: support ticket triage",
        "description": "No image at all: a support email as state, routed and triaged in one pass. The model runs the same prompt and decision position as an image request, with the vision tower skipped. Verified: billing 0.89, urgent 0.98, frustration 1.5 of 2.",
        "images": [],
        "request": {
            "state": {
                "ticket": {
                    "id": "TKT-48812",
                    "from": "dana.okafor@northwind-labs.example",
                    "received": "2026-09-21T08:14:00Z",
                    "subject": "Charged twice for the September seats - third email",
                    "body": ("We were billed 480 USD twice on 3 September for the same 12 seats. "
                             "I have now written three times and the only reply was an automated one. "
                             "Our finance close is on Wednesday and I need the duplicate reversed before then. "
                             "Nothing is broken in the product itself."),
                    "plan": "Team (12 seats)",
                    "prior_tickets_90d": 2,
                },
            },
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department should own this ticket?",
                    "criteria": {
                        "billing": "payments, invoices, refunds and subscription charges",
                        "technical_support": "the product is broken, erroring or unavailable",
                        "sales": "new seats, upgrades, quotes and renewals",
                        "account_management": "ownership, access and contract administration",
                    },
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "This ticket needs a reply today.",
                    "criteria": {"true": "money is at stake or the customer has a dated deadline within days",
                                 "false": "it can wait for the normal queue"},
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated does the customer sound?",
                    "criteria": ["calm and matter-of-fact",
                                 "impatient: chasing a reply or repeating themselves",
                                 "angry: threatening to escalate, churn or dispute the charge"],
                },
            },
        },
    },
    {
        "name": "Text-only: resume screening",
        "description": "State as a plain string rather than JSON, and object-form instructions ({\"question\": ..., \"today\": ...}) so the model can do date arithmetic. Six-option choice, a 6-level score and a noul. Verified: experience 3.5 of 5 (the candidate is at about 7 years, so this is one level low), mentorship 0.93, and it picks engineering_management at 0.49 over backend_systems — a good example of a low-confidence choice you would gate rather than act on.",
        "images": [],
        "request": {
            "state": ("Priya Raghunathan - Bengaluru, India. "
                      "2022-03 to now: Senior Backend Engineer, Zeltic Payments - owns the ledger service "
                      "(Go, Postgres, Kafka), led the migration off a monolith, runs the on-call rotation for six engineers "
                      "and writes the internal design-review guide. "
                      "2019-07 to 2022-02: Backend Engineer, Turnstile Health - billing APIs in Python. "
                      "2018-06 to 2019-06: intern, then junior developer, Kaveri Systems. "
                      "B.E. Computer Science, PES University, 2018. "
                      "Speaks at the Bengaluru Go meetup; mentored four interns to full-time offers."),
            "questions": {
                "experience_years": {
                    "type": "score",
                    "instructions": {
                        "question": "How many years of professional software engineering experience does this candidate have?",
                        "today": "September 22, 2026",
                        "note": "Count paid full-time work only; internships do not count.",
                    },
                    "criteria": ["less than 1 year", "1 to 2 years", "3 to 4 years",
                                 "5 to 6 years", "7 to 9 years", "10 years or more"],
                },
                "would_mentor": {
                    "type": "noul",
                    "instructions": "This candidate would be effective mentoring junior engineers.",
                    "criteria": {"true": "there is evidence of teaching, reviewing or growing other engineers",
                                 "false": "the record is individual-contributor work only"},
                },
                "talent_profile": {
                    "type": "choice",
                    "instructions": "Which profile fits this candidate best?",
                    "criteria": {
                        "backend_systems": "services, data stores and distributed systems",
                        "frontend_product": "user interfaces and client applications",
                        "data_ml": "data pipelines, analytics or machine learning",
                        "infrastructure_sre": "platform, deployment and reliability engineering",
                        "security": "application or infrastructure security",
                        "engineering_management": "primarily leads people rather than writes code",
                    },
                },
            },
        },
    },
]


def load_examples(root=ROOT):
    """Examples whose images all exist under `root`, each with UI-ready image descriptors.

    An example with an empty `images` list is a text-only example and is always listed.
    """
    root = Path(root)
    out = []
    for example in EXAMPLES:
        paths = [root / name for name in example["images"]]
        if not all(path.is_file() for path in paths):
            continue
        index = len(out)
        out.append({
            "index": index,
            "name": example["name"],
            "description": example["description"],
            "request": example["request"],
            "images": [{"index": i, "path": name, "name": Path(name).name,
                        "role": "reference" if i == 0 else "target",
                        "bytes": path.stat().st_size,
                        "url": f"/examples/{index}/image/{i}?v={Path(path).stem[:12]}"}  # cache-buster: image files are content-addressed
                       for i, (name, path) in enumerate(zip(example["images"], paths))],
        })
    return out
