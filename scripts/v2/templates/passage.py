"""Question templates for decision-v2 TEXT passages (Stack Exchange, Wikipedia, support chats).

Every family here is a pure function of ``(record_inputs, rng)`` -- no I/O, no globals mutated,
no model -- so the converter can be replayed byte-for-byte from a seed.  Each family carries at
least eight paraphrases of its question so the model cannot key on one string.

Deviation from the spec's literal ``make(record_inputs, rng) -> list[field]``: the functions
return ``list[Question]``, where ``Question.field`` *is* the contract field.  The wrapper exists
because the converter has to know two things the field itself cannot express -- which template
made it (``template_id``) and whether the question was deliberately built so that the honest
answer is ``unknown`` (``unknown_by_construction``).  ``fields(questions)`` unwraps them.

``record_inputs`` (built by ``scripts/v2/convert_passages.py``):

    id                str    stable passage id
    source            str    stackexchange | wikipedia_paragraphs | support_reviews
    kind              str    qa | article | conversation
    title             str    title / subject line ("" when there is none)
    passage           str    the passage body
    answer            str    accepted answer or final agent turn ("" when there is none)
    topic_hints       list   topic labels the passage plausibly belongs to (may be empty)
    tags              list   upstream tags / services / domains
    entities          list   candidate entity strings lifted from the passage
    numbers           list   {"raw", "value", "unit"} quantities lifted from the passage
    clauses           list   clauses lifted from the passage (claims that ARE stated)
    foreign_clauses   list   clauses lifted from a DIFFERENT passage (claims that are NOT stated)
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

MAX_OPTION = 128


@dataclass(frozen=True)
class Question:
    """One decision field plus the bookkeeping the converter needs."""

    field: dict
    family: str
    template_id: str
    unknown_by_construction: bool = False
    notes: dict = _dc_field(default_factory=dict)


def fields(questions) -> list[dict]:
    """The contract fields of a list of Questions, as the spec's `list[field]`."""
    return [q.field for q in questions]


# --------------------------------------------------------------------------------------- banks

TOPIC_GROUPS = {
    "food": ["cooking and recipes", "food safety and storage", "home brewing and fermentation",
             "vegetarian and vegan eating", "restaurant reservations", "food delivery orders"],
    "home": ["home repair and DIY", "woodworking", "gardening and plants", "interior decorating",
             "everyday household tips", "architecture and construction"],
    "travel": ["travel and transport", "visas and border formalities", "hotel booking",
               "outdoor recreation and camping", "flight booking", "taxi and ride booking",
               "train travel", "car rental", "tourist attractions"],
    "money": ["personal finance", "taxes and accounting", "insurance", "investing and markets",
              "banking support", "business and commerce"],
    "work": ["law and legal procedure", "workplace and careers", "hiring and recruitment",
             "academic life and research", "education and teaching", "project management"],
    "health": ["fitness and exercise", "health and medicine", "nutrition",
               "mental health and relationships", "martial arts and self defence"],
    "family": ["pets and animal care", "parenting and childcare"],
    "tech": ["computer security", "software and computing", "consumer electronics",
             "photography", "user experience design", "graphic design", "music streaming"],
    "vehicles": ["car maintenance and repair", "cycling and bicycles"],
    "arts": ["music", "film and television", "books and writing", "board games and tabletop",
             "movie tickets", "event and ticket booking"],
    "science": ["biology and life sciences", "physics and astronomy", "chemistry",
                "earth science and geography", "mathematics", "environment and sustainability",
                "agriculture"],
    "society": ["history", "politics and government", "religion and philosophy",
                "sports and athletics", "sports information", "language and linguistics",
                "military history", "weather information", "appointment scheduling"],
}
ALL_TOPICS = [t for group in TOPIC_GROUPS.values() for t in group]
TOPIC_GROUP_OF = {t: g for g, ts in TOPIC_GROUPS.items() for t in ts}

TOPIC_QUESTIONS = [
    "Which subject area does this passage belong to?",
    "What is this text mainly about?",
    "Pick the category that best describes the passage.",
    "Classify the passage into one of the listed subject areas.",
    "Which of these topics does the passage fall under?",
    "Reading the text, which area is being discussed?",
    "Assign the passage to the topic it fits best.",
    "Which listed subject does this passage cover?",
    "Choose the topic this passage would be filed under.",
    "Which area of interest does the text concern?",
]

USER_INTENTS = [
    "ask for a recommendation", "report a problem that needs fixing",
    "ask for an explanation of how something works", "request step-by-step instructions",
    "ask whether something is safe or allowed", "compare two or more options",
    "ask for a diagnosis of a symptom", "share an experience without asking anything",
    "ask for help choosing between products", "check whether a plan will work",
]
SERVICE_INTENTS = [
    "book or reserve something", "cancel an existing booking", "change an existing booking",
    "request a refund", "check the status of an order", "complain about the service received",
    "ask about opening hours or availability", "ask about prices or fees",
    "update account or contact details", "ask for directions or travel details",
]
REFERENCE_INTENTS = [
    "define a term", "give background on a subject", "summarise a sequence of events",
    "describe how something is made", "list the properties of something",
    "explain where something is located", "record who did something and when",
    "compare related concepts", "state the outcome of a process",
]
INTENT_QUESTIONS = [
    "What is the writer trying to do in this passage?",
    "What is the purpose of this text?",
    "Which intent best matches the passage?",
    "Reading the passage, what does the author want?",
    "Select the intent behind this text.",
    "What is the author's goal here?",
    "Which of these best describes why the passage was written?",
    "Identify the writer's intent.",
    "What outcome is the writer after?",
]

STANCE_OPTIONS = [
    ("supports it", "The passage argues for the claim or recommends it."),
    ("opposes it", "The passage argues against the claim or advises against it."),
    ("stays neutral", "The passage mentions the subject without taking a side."),
    ("is mixed", "The passage gives reasons on both sides without settling on one."),
]
STANCE_QUESTIONS = [
    "What position does the passage take on {claim}?",
    "How does the text treat the idea that {claim}?",
    "Where does the writer stand on {claim}?",
    "Judge the passage's stance towards {claim}.",
    "Does the passage back or reject {claim}?",
    "Select the attitude the passage shows towards {claim}.",
    "Reading the text, what is its position on {claim}?",
    "Characterise how the passage treats {claim}.",
    "What line does the writer take on {claim}?",
]

ADEQUACY_QUESTIONS = [
    "How well does the supplied answer address the question?",
    "Rate how completely the answer resolves the question that was asked.",
    "Judge the adequacy of the answer shown in the state.",
    "Score the answer against the question it replies to.",
    "How satisfying is this answer for the person who asked?",
    "Assess whether the reply actually answers the question.",
    "Grade the answer on how much of the question it covers.",
    "Using the levels below, rate the quality of the response.",
    "How far does the answer go towards solving the asker's problem?",
]
ADEQUACY_LEVELS = [
    {"value": 0, "description": "Does not address the question at all."},
    {"value": 1, "description": "Touches on the question but leaves the main point unanswered."},
    {"value": 2, "description": "Answers the main point but leaves gaps or caveats unaddressed."},
    {"value": 3, "description": "Fully answers the question, with the detail the asker needs."},
]

SENTIMENT_QUESTIONS = [
    "What sentiment does {who} express?",
    "Rate the tone of {who}.",
    "How positive or negative is {who}?",
    "Judge the feeling conveyed by {who}.",
    "Score the sentiment shown by {who}.",
    "Place {who} on the sentiment scale below.",
    "How happy or unhappy does {who} sound?",
    "Assess the emotional colour of {who}.",
    "Using the levels, rate how favourable {who} is.",
]
SENTIMENT_LEVELS = [
    {"value": 0, "description": "Strongly negative: angry, hostile or badly disappointed."},
    {"value": 1, "description": "Mildly negative: dissatisfied, frustrated or critical."},
    {"value": 2, "description": "Neutral: factual, with no clear positive or negative feeling."},
    {"value": 3, "description": "Mildly positive: satisfied, appreciative or hopeful."},
    {"value": 4, "description": "Strongly positive: delighted, enthusiastic or grateful."},
]

URGENCY_QUESTIONS = [
    "How urgent is {what}?",
    "Rate how quickly {what} needs attention.",
    "Judge the severity of {what}.",
    "Score how pressing {what} is.",
    "Using the levels below, triage {what}.",
    "How serious is {what}?",
    "Place {what} on the urgency scale.",
    "Assess how soon someone must act on {what}.",
    "Grade the priority of {what}.",
]
URGENCY_LEVELS = [
    {"value": 0, "description": "Not urgent: curiosity or planning, nothing is at stake now."},
    {"value": 1, "description": "Routine: should be handled in the normal course of things."},
    {"value": 2, "description": "Pressing: money, travel or comfort is at risk within days."},
    {"value": 3, "description": "Urgent: safety, health, legal exposure or an imminent deadline."},
]

CLAIM_QUESTIONS = [
    "Does the passage state that {claim}?",
    "Is it written in the passage that {claim}?",
    "According to the text, {claim} -- is that stated?",
    "Does the text say that {claim}?",
    "Is the following asserted in the passage: {claim}?",
    "Can you find in the passage that {claim}?",
    "Does the passage claim that {claim}?",
    "Is \"{claim}\" something the passage actually says?",
    "Answer yes only if the passage states that {claim}.",
]

ENTITY_TYPES = [
    ("a person", "A named individual."),
    ("an organisation", "A company, institution, team or public body."),
    ("a place", "A country, city, region or building."),
    ("a product or brand", "A commercial item, model or trademark."),
    ("a date or period", "A point or span of time."),
    ("a measurement or amount", "A quantity, price, distance or duration."),
    ("an event", "Something that happened at a time and place."),
    ("a work of art or publication", "A book, film, album, game or paper."),
    ("a species or material", "A plant, animal or substance."),
    ("a job title or role", "A position a person holds."),
]
ENTITY_QUESTIONS = [
    "In this passage, what kind of thing is \"{entity}\"?",
    "What does \"{entity}\" refer to in the text?",
    "Classify \"{entity}\" as it is used in this passage.",
    "Which category does \"{entity}\" belong to here?",
    "Reading the passage, \"{entity}\" is what sort of entity?",
    "Pick the type that fits \"{entity}\" in this text.",
    "What type of entity is \"{entity}\" in this passage?",
    "Identify what \"{entity}\" names in the text.",
    "Given the passage, how should \"{entity}\" be typed?",
]

ROUTING_QUEUES = {
    "support": ["billing and payments", "technical support", "returns and refunds",
                "account and login", "shipping and delivery", "sales enquiries",
                "complaints and escalations", "bookings and reservations"],
    "community": ["answer the question directly", "ask the poster for more detail",
                  "point the poster at existing documentation", "suggest professional help",
                  "close as off topic", "move to a different community",
                  "flag for a moderator", "suggest a safety warning"],
    "editorial": ["publish as it is", "send back for fact checking", "request a citation",
                  "shorten before publishing", "route to a subject specialist",
                  "hold pending legal review", "merge with an existing article"],
}
OTHER_OPTIONS = ["other", "none of the above", "something else", "no match", "none of these"]
ROUTING_QUESTIONS = [
    "What should happen next with this message?",
    "Which queue should this be routed to?",
    "Choose the next action for whoever picks this up.",
    "Where should this be sent?",
    "Decide what to do with this passage.",
    "Pick the handling step that fits best.",
    "Which team or action does this call for?",
    "Select the next step in the workflow.",
    "How should this be dispatched?",
]

PII_TARGETS = [
    ("a personal email address", "an email address belonging to an individual"),
    ("a telephone number", "a phone number written out in full"),
    ("a home or street address", "a residential or postal address"),
    ("a full personal name", "a person's given name together with a family name"),
    ("a payment card or bank account number", "card digits, an IBAN or an account number"),
    ("a national identity or passport number", "a government-issued identifier"),
    ("a date of birth", "a person's birth date"),
]
PII_QUESTIONS = [
    "Does the passage contain {what}?",
    "Is {what} present anywhere in this text?",
    "Check the passage for {what}: is any there?",
    "Would this passage need redacting because it contains {what}?",
    "Does the text disclose {what}?",
    "Answer yes if the passage includes {what}.",
    "Is there {what} written in the passage?",
    "Scanning the passage, do you find {what}?",
    "Does this text reveal {what}?",
]

NUMERIC_QUESTIONS = [
    "Is the {quantity} stated in the passage {comparison} {threshold}?",
    "According to the text, is the {quantity} {comparison} {threshold}?",
    "Does the passage put the {quantity} {comparison} {threshold}?",
    "Judging only by the passage, is the {quantity} {comparison} {threshold}?",
    "Is the {quantity} given in the text {comparison} {threshold}?",
    "From the stated figures, is the {quantity} {comparison} {threshold}?",
    "Check the numbers: is the {quantity} {comparison} {threshold}?",
    "Answer yes if the passage's {quantity} is {comparison} {threshold}.",
    "Using the passage's own figures, is the {quantity} {comparison} {threshold}?",
]

ABSENT_PARTIES = [
    "the supervisor who reviewed the case", "the second reviewer",
    "the customer's follow-up message the next day", "the engineer assigned to the ticket",
    "the reply that was never posted", "the person who closed the thread",
    "the manager quoted in the appendix", "the survey the reader filled in afterwards",
]
ABSENT_THINGS = [
    "the escalation recorded later in the thread", "the refund request mentioned in the appendix",
    "the outage referenced in the follow-up", "the deadline given in the attachment",
    "the complaint filed after this message", "the safety incident logged separately",
    "the invoice dispute raised the following week", "the recall notice referred to elsewhere",
]
ABSENT_QUANTITIES = [
    ("delivery time in days", "longer than", "3 days"),
    ("warranty length in months", "longer than", "12 months"),
    ("shipping fee in dollars", "above", "20"),
    ("number of people affected", "more than", "50"),
    ("battery life in hours", "above", "8 hours"),
    ("annual interest rate", "above", "5 per cent"),
    ("cancellation window in hours", "longer than", "24 hours"),
    ("weight in kilograms", "above", "10"),
]
ABSENT_ENTITIES = [
    "Northgate Logistics", "Dr. Eleanor Vance", "the Halberd Programme", "Marsden & Co.",
    "the Kestrel Report", "Ridgeway Terminal", "the Ashcombe Trial", "Vellmark Industries",
]
ABSENT_CLAIM_SUBJECTS = [
    "compulsory carbon labelling on menus", "a four-day working week",
    "banning private cars from city centres", "mandatory helmet laws",
    "taxing sugary drinks", "replacing juries with professional judges",
    "universal basic income", "requiring licences for home electrical work",
]


# ------------------------------------------------------------------------------------- helpers

def _pick(rng, bank):
    return bank[rng.randrange(len(bank))]


def _distinct(values):
    """Option values that are unique and not substrings of one another."""
    out = []
    for v in values:
        v = str(v).strip()[:MAX_OPTION]
        if not v or any(v == o or v in o or o in v for o in out):
            continue
        out.append(v)
    return out


def _options(rng, values, descriptions=None, describe_rate=0.30):
    """Shuffled option list; ~30 % of questions carry the Jev-style per-option criteria."""
    values = _distinct(values)
    rng.shuffle(values)
    describe = descriptions is not None and rng.random() < describe_rate
    out = []
    for v in values:
        option = {"value": v}
        if describe and descriptions.get(v):
            option["description"] = descriptions[v][:2000]
        out.append(option)
    return out


def _topic_describe(labels):
    return {label: f"The passage is mainly about {label}." for label in labels}


def _far_topics(rng, hints, count):
    """Topic labels from groups the passage does not belong to -- nothing listed can be right."""
    banned = {TOPIC_GROUP_OF[h] for h in hints if h in TOPIC_GROUP_OF}
    pool = [t for t in ALL_TOPICS if TOPIC_GROUP_OF[t] not in banned and t not in hints]
    rng.shuffle(pool)
    return pool[:count]


def _near_topics(rng, hints, count):
    """The passage's own topic plus hard neighbours drawn mostly from the same group."""
    hint = _pick(rng, hints)
    same = [t for t in TOPIC_GROUPS[TOPIC_GROUP_OF[hint]] if t != hint]
    other = [t for t in ALL_TOPICS if TOPIC_GROUP_OF[t] != TOPIC_GROUP_OF[hint]]
    rng.shuffle(same)
    rng.shuffle(other)
    near = same[: max(1, count // 2)]
    rest = other[: count - 1 - len(near)]
    return [hint] + near + rest


def _intent_bank(kind):
    return {"conversation": SERVICE_INTENTS, "article": REFERENCE_INTENTS}.get(kind, USER_INTENTS)


def _other_intents(kind):
    banks = {"conversation": REFERENCE_INTENTS, "article": SERVICE_INTENTS, "qa": SERVICE_INTENTS}
    return banks[kind]


def _queue_bank(source):
    return {"support_reviews": "support", "wikipedia_paragraphs": "editorial"}.get(source, "community")


def _subject(inputs):
    """A readable name for what the passage is about, for question wording."""
    title = (inputs.get("title") or "").strip()
    if title:
        return title if len(title) <= 90 else title[:87].rsplit(" ", 1)[0] + "..."
    return {"conversation": "this conversation", "article": "this article"}.get(inputs.get("kind"), "this passage")


# --------------------------------------------------------------------------------------- state

IRRELEVANT_FIELDS = [
    ("request_source", "mobile_app"), ("locale", "en-GB"), ("trace_id", "b7f2-91ac"),
    ("queue_depth", 12), ("retrieved_at", "2026-09-21T11:04:00Z"), ("tenant", "acme"),
    ("client_version", "4.2.1"), ("batch", 17), ("dry_run", False), ("operator", "queue-3"),
]


def build_state(inputs, rng, string_rate=0.40, irrelevant_rate=0.17, with_answer=True):
    """The request state for a passage: ~40 % a bare string, ~60 % a named-field JSON object.

    About one state in ten also carries a field that has nothing to do with the question, because
    real requests do: the model must learn to ignore it rather than treat it as evidence.
    """
    title = (inputs.get("title") or "").strip()
    body = inputs["passage"]
    answer = (inputs.get("answer") or "").strip() if with_answer else ""
    if rng.random() < string_rate:
        parts = [p for p in (title, body) if p]
        text = "\n\n".join(parts)
        if answer:
            text += "\n\nAnswer: " + answer
        return text, "string"
    state = {}
    if title:
        state[{"conversation": "subject", "article": "title"}.get(inputs["kind"], "question_title")] = title
    state[{"conversation": "transcript", "article": "summary"}.get(inputs["kind"], "question_body")] = body
    if answer:
        state[{"conversation": "agent_resolution", "article": "note"}.get(inputs["kind"], "accepted_answer")] = answer
    if inputs.get("tags"):
        state["tags"] = list(inputs["tags"])[:6]
    if rng.random() < irrelevant_rate:
        key, value = _pick(rng, IRRELEVANT_FIELDS)
        state[key] = value
        return state, "object+irrelevant"
    return state, "object"


# ------------------------------------------------------------------------------------ families

def topic(inputs, rng, unknown=False):
    """choice, 6-12 options. Unknown variant lists only topics from unrelated groups."""
    hints = inputs.get("topic_hints") or []
    count = rng.randint(6, 12)
    if unknown or not hints:
        values = _far_topics(rng, hints, count)
        unknown = True
    else:
        values = _near_topics(rng, hints, count)
    values = _distinct(values)
    if len(values) < 6:
        return []
    question = _pick(rng, TOPIC_QUESTIONS)
    return [Question(
        field={"id": "topic", "type": "choice", "question": question,
               "options": _options(rng, values, _topic_describe(values))},
        family="topic", template_id=f"topic.v1.{TOPIC_QUESTIONS.index(question)}"
                                   f"{'.unknown' if unknown else ''}",
        unknown_by_construction=unknown)]


def intent(inputs, rng, unknown=False):
    """choice. Unknown variant offers only intents from a different kind of text."""
    bank = _other_intents(inputs["kind"]) if unknown else _intent_bank(inputs["kind"])
    pool = list(bank)
    rng.shuffle(pool)
    values = _distinct(pool[: rng.randint(4, min(9, len(pool)))])
    if len(values) < 3:
        return []
    question = _pick(rng, INTENT_QUESTIONS)
    return [Question(
        field={"id": "intent", "type": "choice", "question": question,
               "options": _options(rng, values, {v: f"The writer wants to {v}." for v in values})},
        family="intent", template_id=f"intent.v1.{INTENT_QUESTIONS.index(question)}"
                                     f"{'.unknown' if unknown else ''}",
        unknown_by_construction=unknown)]


def stance(inputs, rng, unknown=False):
    """choice. Unknown variant asks about a subject the passage never raises."""
    if unknown:
        claim = _pick(rng, ABSENT_CLAIM_SUBJECTS)
    else:
        clauses = inputs.get("clauses") or []
        if not clauses:
            return []
        claim = _pick(rng, clauses)
        claim = claim[:150].rstrip(" ,.;:")
    pairs = list(STANCE_OPTIONS)
    rng.shuffle(pairs)
    keep = pairs[: rng.randint(3, 4)]
    if not any(v == "supports it" for v, _ in keep):
        keep[0] = ("supports it", dict(STANCE_OPTIONS)["supports it"])
    values = [v for v, _ in keep]
    template = _pick(rng, STANCE_QUESTIONS)
    return [Question(
        field={"id": "stance", "type": "choice", "question": template.format(claim=claim),
               "options": _options(rng, values, dict(STANCE_OPTIONS))},
        family="stance", template_id=f"stance.v1.{STANCE_QUESTIONS.index(template)}"
                                     f"{'.unknown' if unknown else ''}",
        unknown_by_construction=unknown)]


def adequacy_of_answer(inputs, rng, unknown=False):
    """ordinal, 4 levels, for question+answer passages.

    The unknown variant is built by the converter withholding the answer from the state: the
    question then asks the model to rate a reply it was never shown.
    """
    if not unknown and not (inputs.get("answer") or "").strip():
        return []
    question = _pick(rng, ADEQUACY_QUESTIONS)
    return [Question(
        field={"id": "adequacy", "type": "ordinal", "question": question,
               "levels": [dict(level) for level in ADEQUACY_LEVELS]},
        family="adequacy_of_answer",
        template_id=f"adequacy.v1.{ADEQUACY_QUESTIONS.index(question)}"
                    f"{'.unknown' if unknown else ''}",
        unknown_by_construction=unknown,
        notes={"withhold_answer": bool(unknown)})]


def sentiment(inputs, rng, unknown=False):
    """ordinal, 5 levels. Unknown variant asks about somebody who does not appear."""
    if unknown:
        who = _pick(rng, ABSENT_PARTIES)
    else:
        who = {"conversation": "the customer in this conversation",
               "article": "this article towards its subject"}.get(inputs["kind"], "the writer of this post")
    template = _pick(rng, SENTIMENT_QUESTIONS)
    return [Question(
        field={"id": "sentiment", "type": "ordinal", "question": template.format(who=who),
               "levels": [dict(level) for level in SENTIMENT_LEVELS]},
        family="sentiment", template_id=f"sentiment.v1.{SENTIMENT_QUESTIONS.index(template)}"
                                        f"{'.unknown' if unknown else ''}",
        unknown_by_construction=unknown)]


def urgency(inputs, rng, unknown=False):
    """ordinal, 4 levels (severity / urgency). Unknown variant triages something absent."""
    if unknown:
        what = _pick(rng, ABSENT_THINGS)
    else:
        what = {"conversation": "the customer's request in this conversation",
                "article": "the situation this article describes"}.get(inputs["kind"], "the problem described in this post")
    template = _pick(rng, URGENCY_QUESTIONS)
    return [Question(
        field={"id": "urgency", "type": "ordinal", "question": template.format(what=what),
               "levels": [dict(level) for level in URGENCY_LEVELS]},
        family="severity_urgency", template_id=f"urgency.v1.{URGENCY_QUESTIONS.index(template)}"
                                               f"{'.unknown' if unknown else ''}",
        unknown_by_construction=unknown)]


def contains_claim(inputs, rng, absent=None):
    """boolean. `absent` forces a claim that is NOT in the passage; the converter balances 50/50."""
    if absent is None:
        absent = rng.random() < 0.5
    pool = inputs.get("foreign_clauses") if absent else inputs.get("clauses")
    if not pool:
        return []
    claim = _pick(rng, pool)[:180].rstrip(" ,.;:")
    if len(claim) < 25:
        return []
    template = _pick(rng, CLAIM_QUESTIONS)
    return [Question(
        field={"id": "claim_stated", "type": "boolean", "question": template.format(claim=claim),
               "yes_description": "The passage states this, in these words or plainly equivalent ones.",
               "no_description": "The passage does not state this."},
        family="contains_claim",
        template_id=f"claim.v1.{CLAIM_QUESTIONS.index(template)}{'.absent' if absent else '.present'}",
        unknown_by_construction=False,
        notes={"claim_absent": bool(absent)})]


def entity_type(inputs, rng, unknown=False):
    """choice. Unknown variant asks about a name that never appears in the passage."""
    if unknown:
        entity = _pick(rng, ABSENT_ENTITIES)
    else:
        candidates = [e for e in (inputs.get("entities") or []) if 3 <= len(e) <= 60]
        if not candidates:
            return []
        entity = _pick(rng, candidates)
    pairs = list(ENTITY_TYPES)
    rng.shuffle(pairs)
    keep = pairs[: rng.randint(4, 8)]
    values = [v for v, _ in keep]
    template = _pick(rng, ENTITY_QUESTIONS)
    return [Question(
        field={"id": "entity", "type": "choice", "question": template.format(entity=entity),
               "options": _options(rng, values, dict(ENTITY_TYPES))},
        family="entity_type", template_id=f"entity.v1.{ENTITY_QUESTIONS.index(template)}"
                                          f"{'.unknown' if unknown else ''}",
        unknown_by_construction=unknown)]


def routing(inputs, rng, unknown=False, other_rate=0.70):
    """choice with a user-supplied "other" option.

    Jev has no built-in abstention, so a user who wants a no-match answer supplies one; when that
    option is present the honest answer is that option, never `unknown`.  The unknown variant
    therefore drops it and lists queues from a workflow the passage cannot belong to.
    """
    bank = _queue_bank(inputs["source"])
    if unknown:
        bank = _pick(rng, [b for b in ROUTING_QUEUES if b != bank])
    pool = list(ROUTING_QUEUES[bank])
    rng.shuffle(pool)
    values = pool[: rng.randint(4, min(7, len(pool)))]
    has_other = (not unknown) and rng.random() < other_rate
    if has_other:
        values = values + [_pick(rng, OTHER_OPTIONS)]
    values = _distinct(values)
    if len(values) < 3:
        return []
    question = _pick(rng, ROUTING_QUESTIONS)
    return [Question(
        field={"id": "next_action", "type": "choice", "question": question,
               "options": _options(rng, values, None)},
        family="next_action_routing",
        template_id=f"routing.v1.{ROUTING_QUESTIONS.index(question)}.{bank}"
                    f"{'.other' if has_other else ''}{'.unknown' if unknown else ''}",
        unknown_by_construction=unknown,
        notes={"user_supplied_other": has_other})]


def pii_present(inputs, rng):
    """boolean. Always answerable from the passage, so never an unknown construction."""
    what, detail = _pick(rng, PII_TARGETS)
    template = _pick(rng, PII_QUESTIONS)
    return [Question(
        field={"id": "pii", "type": "boolean", "question": template.format(what=what),
               "yes_description": f"The passage contains {detail}.",
               "no_description": f"The passage does not contain {detail}."},
        family="pii_present", template_id=f"pii.v1.{PII_QUESTIONS.index(template)}",
        unknown_by_construction=False)]


def numeric_comparison(inputs, rng, unknown=False):
    """boolean over a figure in the passage. Unknown variant asks about a figure never given."""
    if unknown:
        quantity, comparison, threshold = _pick(rng, ABSENT_QUANTITIES)
    else:
        numbers = [n for n in (inputs.get("numbers") or []) if n.get("value") is not None]
        if not numbers:
            return []
        number = _pick(rng, numbers)
        value = float(number["value"])
        quantity = number.get("quantity") or "figure"
        # a threshold on either side of the stated value, so yes and no are both common
        factor = _pick(rng, [0.5, 0.75, 0.9, 1.1, 1.5, 2.0])
        bound = value * factor
        bound = round(bound) if abs(bound) >= 10 else round(bound, 2)
        comparison = _pick(rng, ["above", "below", "at least", "at most"])
        unit = number.get("unit") or ""
        threshold = f"{bound}{(' ' + unit) if unit else ''}".strip()
    template = _pick(rng, NUMERIC_QUESTIONS)
    question = template.format(quantity=quantity, comparison=comparison, threshold=threshold)
    return [Question(
        field={"id": "numeric", "type": "boolean", "question": question,
               "yes_description": "The passage's own figures make this true.",
               "no_description": "The passage's own figures make this false."},
        family="numeric_comparison",
        template_id=f"numeric.v1.{NUMERIC_QUESTIONS.index(template)}{'.unknown' if unknown else ''}",
        unknown_by_construction=unknown)]


SINGLE_FAMILIES = {
    "topic": topic, "intent": intent, "stance": stance, "adequacy_of_answer": adequacy_of_answer,
    "sentiment": sentiment, "severity_urgency": urgency, "contains_claim": contains_claim,
    "entity_type": entity_type, "next_action_routing": routing, "pii_present": pii_present,
    "numeric_comparison": numeric_comparison,
}
UNKNOWN_CAPABLE = {"topic", "intent", "stance", "adequacy_of_answer", "sentiment",
                   "severity_urgency", "entity_type", "next_action_routing", "numeric_comparison"}


def make_one(family, inputs, rng, unknown=False):
    """Dispatch to one family. Families that cannot honestly be made unknown ignore the flag."""
    fn = SINGLE_FAMILIES[family]
    if family == "contains_claim":
        return fn(inputs, rng)
    if family == "pii_present":
        return fn(inputs, rng)
    return fn(inputs, rng, unknown=unknown and family in UNKNOWN_CAPABLE)


def multi_question(inputs, rng, families, unknown_family=None):
    """2-5 questions about one passage, with unique field ids.

    Families that return nothing for this passage are skipped rather than faked, so the caller
    must be ready for fewer than it asked for.
    """
    out: list[Question] = []
    used: set[str] = set()
    for family in families:
        made = make_one(family, inputs, rng, unknown=(family == unknown_family))
        for q in made:
            ident = q.field["id"]
            if ident in used:
                continue
            used.add(ident)
            out.append(q)
        if len(out) >= 5:
            break
    return out[:5]
