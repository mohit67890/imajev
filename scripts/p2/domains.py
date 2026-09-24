"""Business domains for decision-p2 documents. Each entry: id, a short brief for the writer, typical document kinds."""
from __future__ import annotations

DOMAINS: list[dict] = [
    {"id": "ecommerce_listings", "brief": "online marketplace product listings, seller policies and buyer messages", "kinds": ["product listing", "seller policy", "buyer-seller chat"]},
    {"id": "logistics", "brief": "shipping, warehousing, delivery windows, customs paperwork", "kinds": ["shipment log", "delivery exception report", "customs form"]},
    {"id": "insurance_claims", "brief": "property, motor and health insurance claims and policy wording", "kinds": ["claim file", "policy excerpt", "adjuster notes"]},
    {"id": "hr_policy", "brief": "leave, expenses, conduct and remote-work policies with exceptions", "kinds": ["HR policy", "leave request thread", "expense claim"]},
    {"id": "it_support", "brief": "helpdesk tickets, runbooks, access requests, SLA rules", "kinds": ["support ticket", "runbook", "access request"]},
    {"id": "finance_ops", "brief": "accounts payable, reconciliations, approvals thresholds", "kinds": ["invoice batch", "reconciliation sheet", "approval matrix"]},
    {"id": "healthcare_admin", "brief": "appointment scheduling, referral rules, billing codes (no clinical advice)", "kinds": ["referral form", "scheduling policy", "billing note"]},
    {"id": "travel", "brief": "bookings, fare rules, cancellation windows, loyalty tiers", "kinds": ["itinerary", "fare rules", "change request email"]},
    {"id": "contracts", "brief": "service agreements, SLAs, termination and notice clauses", "kinds": ["contract excerpt", "amendment", "notice letter"]},
    {"id": "education", "brief": "course policies, grading rubrics, deadlines and extensions", "kinds": ["syllabus", "extension request", "grading rubric"]},
    {"id": "real_estate", "brief": "leases, inspections, deposits, notice periods", "kinds": ["lease excerpt", "inspection report", "tenant email"]},
    {"id": "hospitality", "brief": "hotel and restaurant bookings, house rules, incident logs", "kinds": ["booking record", "house rules", "incident log"]},
    {"id": "telecom", "brief": "plans, roaming, outages, credits and disputes", "kinds": ["plan terms", "outage log", "dispute ticket"]},
    {"id": "energy", "brief": "utility tariffs, meter readings, outages, safety notices", "kinds": ["tariff sheet", "meter log", "safety notice"]},
    {"id": "government_forms", "brief": "permits, eligibility rules, application checklists", "kinds": ["eligibility rules", "application form", "case note"]},
    {"id": "manufacturing_qa", "brief": "inspection tolerances, batch records, non-conformance reports", "kinds": ["batch record", "NCR", "tolerance spec"]},
    {"id": "retail_returns", "brief": "returns, refunds, warranties, restocking rules", "kinds": ["return request", "warranty terms", "store policy"]},
    {"id": "saas_billing", "brief": "subscriptions, proration, seats, dunning, plan changes", "kinds": ["billing event log", "plan matrix", "support thread"]},
    {"id": "security_incidents", "brief": "alerts, severity runbooks, access logs, escalation rules", "kinds": ["incident timeline", "severity runbook", "access log"]},
    {"id": "procurement", "brief": "RFQs, vendor scoring, approval limits, purchase orders", "kinds": ["RFQ summary", "vendor scorecard", "purchase order"]},
    {"id": "events", "brief": "venue bookings, capacity rules, schedules, vendor cutoffs", "kinds": ["run sheet", "venue terms", "vendor email"]},
    {"id": "fleet", "brief": "vehicle maintenance, driver hours, route logs, compliance", "kinds": ["maintenance log", "driver hours sheet", "route plan"]},
    {"id": "agriculture_supply", "brief": "harvest lots, grading standards, cold-chain logs, contracts", "kinds": ["lot ticket", "grading standard", "cold-chain log"]},
    {"id": "nonprofit_grants", "brief": "grant eligibility, reporting deadlines, budget rules", "kinds": ["grant terms", "budget report", "program note"]},
]
DOMAIN_IDS = [d["id"] for d in DOMAINS]
assert len(DOMAINS) == 24 and len(set(DOMAIN_IDS)) == 24


def domain(domain_id: str) -> dict:
    for d in DOMAINS:
        if d["id"] == domain_id:
            return d
    raise KeyError(domain_id)
