"""Boilerplate sections for long documents: realistic, varied and irrelevant to every question the generators ask.

These sentences never state thresholds, windows, amounts or outcomes for the request types the questions are about; they only talk
about process (records, privacy, training, review, contacts). Numbers they contain are about process (retention years, training
hours), never about the decision rules. Sentences are assembled from independent pools with frequent slot fills so two documents
rarely share long word runs (the 13-gram leakage check compares held-out and train items later).
"""
from __future__ import annotations

import random

from .common import P, chance

OPENERS = ["", "", "", "", "Where practicable, ", "As a general principle, ", "For the avoidance of doubt, ", "In all cases, ", "Unless {org} agrees otherwise, ",
           "Subject to applicable law, ", "In line with good practice, ", "Wherever possible, "]

TOPICS: dict[str, dict] = {
    "purpose": dict(
        titles=["Purpose", "Purpose and objectives", "Why this document exists", "Introduction", "Background and purpose", "Aims"],
        s=["This document sets out how {org} handles {requests} and the standards that {parties} can expect from the {dept} team.",
           "It is intended to give {parties} and staff a consistent, transparent basis for decisions made by {dept}.",
           "The {dept} team owns this document and is responsible for keeping it accurate and up to date.",
           "It replaces all earlier guidance notes, emails and local practices on the same subject issued before {yr}.",
           "{org} aims to handle every case fairly, promptly and in a way that can be explained afterwards.",
           "Staff should read this document together with the {system} user guide and the internal escalation contacts list.",
           "Nothing in this document limits any statutory rights that {parties} may have.",
           "The objective is that two colleagues looking at the same case reach the same decision for the same reasons.",
           "Where this document and a local work instruction disagree, this document takes precedence.",
           "The {dept} team publishes a plain-language summary of this document on the {org} website.",
           "This document was prepared by {person} on behalf of the {dept} leadership group.",
           "It explains both the rules and the reasons for them, so that staff can apply them sensibly to unusual cases."]),
    "scope": dict(
        titles=["Scope", "Who this applies to", "Application", "Coverage", "Scope and application"],
        s=["This document applies to all {parties} of {org} and to every member of staff who handles {requests}.",
           "It covers cases received through {system}, by telephone, by post and in person.",
           "Contractors and agency staff working in the {dept} team must follow it in the same way as employees.",
           "It does not cover internal disputes between departments, which are handled by the {dept2} team.",
           "Separate arrangements apply to cases handled by partner organisations acting on behalf of {org}.",
           "Where a case also raises a complaint about staff conduct, the conduct element is handled under the separate complaints process.",
           "Cases already closed before this version took effect are not reopened only because of a change in this document.",
           "The document applies in every region where {org} operates unless a regional annex says otherwise.",
           "It applies to cases regardless of the channel through which the {party} first made contact."]),
    "roles": dict(
        titles=["Roles and responsibilities", "Responsibilities", "Who does what", "Accountabilities", "Ownership"],
        s=["The {role} is responsible for the first review of each case and for recording the reasons for the decision.",
           "The {role2} reviews a sample of decisions every {freq} and reports any themes to the {dept} leadership group.",
           "{person}, as the document owner, approves changes to templates and standard wording.",
           "Team leaders make sure that new starters complete the induction module before handling live cases.",
           "The {dept2} team provides second-line assurance and may review any case file without notice.",
           "Staff must declare any personal connection to a {party} whose case they are asked to handle.",
           "The {role} must not approve a case that they themselves raised or submitted.",
           "All staff are responsible for flagging gaps or inconsistencies in this document to the document owner.",
           "The {dept} leadership group meets every {freq} to review workload, backlogs and quality results.",
           "Deputies act for the {role2} during planned absences, using the same guidance as set out here."]),
    "records": dict(
        titles=["Record keeping", "Records and documentation", "Case records", "Documentation standards", "Audit trail"],
        s=["Every case must be logged in {system} on the day it is received, even if it cannot be assessed straight away.",
           "The case record should show who made each decision, when it was made and which clause was relied on.",
           "Supporting documents are attached to the case record rather than stored in personal folders or email inboxes.",
           "Case records are retained for {n} years after closure and then securely destroyed.",
           "Handwritten notes must be scanned into {system} within two working days and the originals shredded.",
           "Records must be written in clear, neutral language that the {party} could read without offence.",
           "Where a decision is changed on review, the original decision stays visible in the record with a note explaining the change.",
           "Staff should avoid abbreviations in case notes other than those listed in the {dept} style guide.",
           "The audit log in {system} is read-only and may not be edited by any member of the {dept} team.",
           "Case files selected for quality review are locked until the review is complete."]),
    "privacy": dict(
        titles=["Data protection", "Personal data", "Privacy", "Handling personal information", "Confidentiality and data protection"],
        s=["Personal data collected while handling a case is used only for that purpose and for related quality checks.",
           "Staff must verify the identity of the {party} before discussing any case details by telephone.",
           "Information must not be shared with a third party unless the {party} has given written authority or the law requires it.",
           "Special category data, such as health information, is accessible only to staff who need it to decide the case.",
           "Suspected data breaches are reported to the data protection lead within {m} hours of discovery.",
           "Copies of identity documents are deleted once verification has been recorded in {system}.",
           "Requests from {parties} for a copy of their data are passed to the privacy team and are not answered by case handlers.",
           "Screens showing case data must be locked whenever a member of staff leaves their desk.",
           "Emails containing personal data are sent only through the secure messaging option in {system}.",
           "{org} does not use case data for marketing without separate consent."]),
    "training": dict(
        titles=["Training", "Training and competence", "Learning requirements", "Staff development", "Competence"],
        s=["New members of the {dept} team complete {n} hours of supervised case handling before deciding cases alone.",
           "Refresher training on this document is completed every {freq} and recorded on the learning platform.",
           "Staff who fall below the quality benchmark in two consecutive reviews receive targeted coaching from the {role2}.",
           "Training materials are updated within one month of any change to this document.",
           "Case studies used in training are anonymised and approved by the document owner.",
           "{person} coordinates the training calendar and keeps attendance records.",
           "Temporary staff receive a shortened induction but may not decide cases that need a second approval.",
           "Completion of the annual ethics module is a condition of continued access to {system}."]),
    "complaints": dict(
        titles=["Complaints and appeals", "Appeals", "If you disagree with a decision", "Review of decisions", "Challenging a decision"],
        s=["A {party} who disagrees with a decision may ask for it to be reviewed by someone who was not involved in the original decision.",
           "Review requests are acknowledged within {m} working days and answered in full as soon as the review is complete.",
           "The reviewer may confirm the decision, replace it, or return the case for further information.",
           "If the {party} remains dissatisfied after review, the final response letter explains any external options available.",
           "A review does not suspend any action already taken unless the reviewer decides otherwise.",
           "The {dept2} team tracks review outcomes and reports overturn rates to the leadership group every {freq}.",
           "Staff must not discourage a {party} from asking for a review.",
           "Where a review finds a systemic error, similar open cases are checked before the error is corrected."]),
    "review": dict(
        titles=["Document review", "Review and maintenance", "Monitoring and review", "Keeping this document current", "Governance"],
        s=["This document is reviewed every {freq} by the document owner, or sooner if the law or {org}'s products change.",
           "Proposed changes are circulated to the {dept} and {dept2} teams for comment before approval.",
           "Each version is numbered, and the version history below records what changed and when.",
           "Minor corrections that do not change meaning, such as fixing typing errors, may be made without a new version number.",
           "The leadership group approves each new version and records the approval in its minutes.",
           "Superseded versions are archived and remain available to staff for reference on older cases.",
           "Performance against the standards in this document is reported in the {dept} quarterly pack.",
           "{person} is the named contact for questions about this document."]),
    "communication": dict(
        titles=["Communicating with customers", "Communication standards", "Letters and messages", "Tone and communication", "Keeping people informed"],
        s=["Decisions are communicated in writing, using the approved templates in {system}.",
           "Letters should explain the decision, the reason for it and what the {party} can do next.",
           "Staff should use plain English and avoid internal jargon when writing to {parties}.",
           "Where a {party} has asked for communication in an accessible format, that format is used for every message on the case.",
           "Telephone calls about a decision are followed up with a written summary within {m} working days.",
           "Staff should not give an indication of the likely outcome before the case has been fully assessed.",
           "If a case is delayed, the {party} is told why and given a realistic new date.",
           "Messages sent through {system} are copied automatically to the case record."]),
    "conflicts": dict(
        titles=["Conflicts of interest", "Independence", "Declarations of interest", "Integrity"],
        s=["Staff must not handle a case involving a relative, close friend or business associate.",
           "Gifts or hospitality offered by a {party} in connection with a case must be declined and recorded.",
           "Any potential conflict is declared to the {role2}, who reassigns the case.",
           "The register of declared interests is maintained by the {dept2} team and reviewed every {freq}.",
           "Failure to declare a conflict may lead to disciplinary action under the staff code of conduct."]),
    "quality": dict(
        titles=["Quality assurance", "Quality checks", "Assurance", "Quality monitoring"],
        s=["A random sample of closed cases is checked every {freq} against a standard quality scorecard.",
           "Quality reviewers look at accuracy, timeliness, record keeping and the tone of written communication.",
           "Results are shared with individual staff members privately and with the team in aggregate.",
           "Cases found to have been decided incorrectly are reopened and the {party} is contacted with an explanation.",
           "Themes from quality checks feed into the training plan for the following period.",
           "The {role2} signs off the quality summary before it goes to the leadership group."]),
    "continuity": dict(
        titles=["Business continuity", "System outages", "Contingency arrangements", "If systems are unavailable"],
        s=["If {system} is unavailable, cases are logged on the paper contingency form and entered once the system is restored.",
           "During an outage, staff should prioritise cases involving vulnerable {parties}.",
           "The {dept} team keeps an offline copy of the escalation contacts list for use during outages.",
           "Contingency arrangements are tested at least once every {freq}.",
           "Delays caused by a system outage are recorded on the affected case records."]),
    "accessibility": dict(
        titles=["Accessibility and support", "Reasonable adjustments", "Supporting vulnerable customers", "Additional support"],
        s=["{org} will make reasonable adjustments for {parties} who need extra support to use this process.",
           "Staff should record any support needs, with consent, so that the {party} does not have to repeat them.",
           "Interpreting and translation services can be arranged through the {dept2} team at no cost to the {party}.",
           "A {party} may nominate a representative to act on their behalf, provided written authority is on file.",
           "Signs that a {party} may be in vulnerable circumstances should be referred to the specialist support team."]),
    "fraud": dict(
        titles=["Fraud awareness", "Preventing misuse", "Suspicious activity", "Financial crime"],
        s=["Staff who suspect that a case may be fraudulent should refer it to the {dept2} team without alerting the {party}.",
           "A referral for suspected fraud does not by itself change the outcome of a case; it pauses the case until the referral is closed.",
           "Documents that appear altered should be copied to the case record and flagged in {system}.",
           "Staff must never carry out their own investigation beyond the checks described in this document.",
           "Fraud indicators are refreshed every {freq} and circulated to the {dept} team."]),
    "channels": dict(
        titles=["How to contact us", "Submission channels", "Making contact", "Ways to get in touch", "Channels"],
        s=["{parties} can contact the {dept} team through {system}, by email or by post to the address on the {org} website.",
           "Telephone lines are open from 08:30 to 18:00 on weekdays; calls may be recorded for training purposes.",
           "Messages received through social media are redirected to {system} and are not handled on the social media platform itself.",
           "A {party} who cannot use online services may ask a member of staff to complete the form on their behalf.",
           "Each contact receives a reference number, which should be quoted in any follow-up.",
           "Attachments sent by email must not exceed 20 MB in total; larger files can be uploaded through {system}.",
           "The {dept} team does not accept documents sent through personal messaging apps.",
           "Where a {party} writes to a named member of staff, the message is still logged in {system} in the usual way."]),
    "evidence": dict(
        titles=["Supporting documents", "Evidence", "What to provide", "Documents we may ask for", "Information requirements"],
        s=["Staff may ask for documents that are reasonably needed to understand the case, but should not ask for the same document twice.",
           "Copies are acceptable unless there is a specific reason to see an original, which must be recorded on the case.",
           "Documents in a language other than English may be accepted with a translation arranged by the {dept2} team.",
           "Where a document is illegible, the {party} is asked for a clearer copy and the case is marked as awaiting information.",
           "Evidence is assessed on the balance of probabilities; staff should not demand certainty that the {party} could not reasonably provide.",
           "Staff must not alter, annotate or crop documents supplied by a {party}.",
           "Photographs should show the relevant detail clearly and, where possible, include a reference number in the frame.",
           "The {role} decides whether the evidence supplied is sufficient and records that judgement."]),
    "payments": dict(
        titles=["Payments", "How payments are made", "Settlement arrangements", "Paying approved amounts", "Refund method"],
        s=["Approved amounts are paid by the same method the {party} originally used, unless that method is no longer available.",
           "Where the original method is unavailable, payment is made by bank transfer to an account in the {party}'s name.",
           "{org} does not pay approved amounts in cash.",
           "Payment confirmations are sent automatically from {system} once the payment run has completed.",
           "Payment runs take place twice a week; staff cannot trigger an individual payment outside a run.",
           "Where a {party} owes {org} an undisputed amount, staff may ask the {dept2} team whether an offset is appropriate.",
           "Currency conversions, where needed, use the rate published by {org}'s bank on the day of payment.",
           "Payments returned by the receiving bank are held on the case until the {party} confirms new details."]),
    "service_standards": dict(
        titles=["Service standards", "Our commitments", "What you can expect", "Customer service standards"],
        s=["We aim to acknowledge every contact within {m} working days and to keep the {party} informed at each stage.",
           "Staff introduce themselves by name and give a direct contact route for the rest of the case.",
           "Where we make a mistake, we say so, put it right and explain what we have changed.",
           "Service standards are published on the {org} website and reviewed every {freq}.",
           "Performance against these standards is reported to the {dept} leadership group.",
           "Waiting times at service counters are monitored, and extra staff are deployed at peak periods where possible.",
           "Feedback forms are sent to a random sample of {parties} once a case is closed."]),
    "systems": dict(
        titles=["Systems and access", "Use of systems", "Information systems", "Access control"],
        s=["Access to {system} is granted by the {dept2} team on the request of a line manager.",
           "Shared log-ins are not permitted; each member of staff uses their own account.",
           "Access rights are reviewed every {freq} and removed promptly when a member of staff changes role.",
           "Staff must report any system fault that could affect a case record to the service desk on the same day.",
           "Reports produced from {system} are marked with the date and time they were generated.",
           "Test environments must never contain real case data."]),
}
TOPIC_IDS = list(TOPICS)
FREQS = ["quarter", "six months", "year", "month", "two years"]


def fill(s: str, rng: random.Random, ctx: dict) -> str:
    vals = dict(ctx)
    vals.setdefault("n", rng.choice([3, 5, 6, 7, 10]))
    vals.setdefault("m", rng.choice([2, 3, 5]))
    vals.setdefault("freq", rng.choice(FREQS))
    vals.setdefault("yr", rng.choice([2019, 2020, 2021, 2022, 2023, 2024]))
    return s.format(**vals)


def section(rng: random.Random, topic: str, ctx: dict, k: int | None = None) -> tuple[str, list[str]]:
    t = TOPICS[topic]
    pool = list(t["s"])
    rng.shuffle(pool)
    k = k or rng.randint(3, min(6, len(pool)))
    out = []
    for s in pool[:k]:
        op = rng.choice(OPENERS)
        s2 = fill(s, rng, ctx)
        if op and not s.startswith(("{", "It ", "This ", "Nothing", "Each ", "Where", "If ", "During")):
            s2 = fill(op, rng, ctx) + s2[0].lower() + s2[1:]
        out.append(s2)
    return P(rng, *t["titles"]), out


def pick_topics(rng: random.Random, n: int, exclude=()) -> list[str]:
    ids = [t for t in TOPIC_IDS if t not in exclude]
    rng.shuffle(ids)
    return ids[:n]


def version_history(rng: random.Random, ctx: dict, dates: list, fmt) -> list[str]:
    """Version-history table rows about editorial changes (never about decision rules)."""
    notes = ["Initial issue", "Formatting and template updates", "Contact details refreshed", "Clarified record-keeping steps",
             "Updated references to {system}", "Minor wording corrections", "Added accessibility section", "Aligned headings with the group style guide",
             "Removed obsolete appendix", "Updated owner details"]
    rng.shuffle(notes)
    rows = ["| Version | Date | Summary of change | Approved by |", "|---|---|---|---|"]
    for i, d in enumerate(sorted(dates)):
        rows.append(f"| {i + 1}.{rng.randint(0, 3)} | {fmt(d)} | {fill(notes[i % len(notes)], rng, ctx)} | {ctx['person'] if i % 2 else ctx['role2']} |")
    return rows
