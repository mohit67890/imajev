"""routing: pick the correct handler for an incoming item from 5 to 8 options, including 'other'."""

ITEMS = [
{
 "key": "ro01", "domain": "retail",
 "state": {"channel": "customer email", "subject": "Wrong colour delivered",
           "body": "I ordered the navy cushion covers but the parcel contains grey ones. The order number is "
                   "N-88210 and everything else in the box was right. I would like the navy ones sent out.",
           "order_status": "delivered", "payment_status": "captured", "account_flags": []},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this email be routed to?",
           "options": [{"value": "order accuracy", "description": "The customer received something different from what was ordered."},
                       {"value": "delivery tracking", "description": "The parcel is late, lost or not yet delivered."},
                       {"value": "payments and refunds", "description": "A charge, refund or payment method question."},
                       {"value": "product advice", "description": "Help choosing or using a product."},
                       {"value": "returns", "description": "The customer wants to send something back for their own reasons."},
                       {"value": "other"}]},
 "target": "order accuracy", "abstention_cause": None,
 "rationale": "The parcel arrived but held a different colour from the one ordered, which is a pick accuracy problem rather than delivery, payment or a change of mind."},
{
 "key": "ro02", "domain": "software_incidents",
 "state": "A ticket arrives reading: 'Our nightly export to object storage has not run for three days. The job "
          "shows as succeeded in the scheduler but the bucket is empty. No error appears in the application log.' "
          "The platform has separate on-call queues for authentication, data pipelines, billing, the public API "
          "and the web front end, plus a general queue for anything that fits none of them. Sign-in, checkout and "
          "the customer-facing pages are all healthy, and no customer has reported a billing problem.",
 "field": {"id": "queue", "type": "choice",
           "question": "Which on-call queue should take this ticket?",
           "options": [{"value": "authentication"}, {"value": "data pipelines"}, {"value": "billing"},
                       {"value": "public API"}, {"value": "web front end"}, {"value": "other"}]},
 "target": "data pipelines", "abstention_cause": None,
 "rationale": "A scheduled nightly export that reports success while writing nothing is a data pipeline failure, and the other components are confirmed healthy."},
{
 "key": "ro03", "domain": "hr",
 "state": {"channel": "HR service desk", "request": "An employee writes that her manager has repeatedly made "
                                                    "comments about her accent in team meetings and that she "
                                                    "wants it dealt with formally.",
           "employee_service_months": 30, "existing_cases": [], "pay_query": False, "absence_query": False,
           "teams": ["payroll", "benefits", "employee relations", "recruitment", "learning and development",
                     "other"]},
 "field": {"id": "team", "type": "choice",
           "question": "Which HR team should take this request?",
           "options": [{"value": "payroll"}, {"value": "benefits"},
                       {"value": "employee relations", "description": "Grievances, conduct and formal complaints between colleagues."},
                       {"value": "recruitment"}, {"value": "learning and development"}, {"value": "other"}]},
 "target": "employee relations", "abstention_cause": None,
 "rationale": "A formal complaint about a manager's conduct toward an employee is a grievance, which employee relations handles."},
{
 "key": "ro04", "domain": "healthcare_admin",
 "state": "A call comes in to the practice: 'I had my blood test on Tuesday and nobody has phoned me with the "
          "result. I am not unwell, I just want to know what it said.' The caller is registered at the practice, "
          "has no appointment booked and is not asking for one. The practice routes calls to urgent clinical "
          "triage, appointment booking, the results line, prescriptions, registrations and records, billing and "
          "insurance, or other.",
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this call be routed?",
           "options": [{"value": "urgent clinical triage"}, {"value": "appointment booking"},
                       {"value": "results line"}, {"value": "prescriptions"},
                       {"value": "registrations and records"}, {"value": "billing and insurance"},
                       {"value": "other"}]},
 "target": "results line", "abstention_cause": None,
 "rationale": "The caller wants a test result and states she is not unwell, so the results line handles it rather than triage or booking."},
{
 "key": "ro05", "domain": "finance_ops",
 "state": {"document": "supplier email", "text": "Please note our bank details have changed with effect from "
                                                 "1 October. Kindly update your records and pay the attached "
                                                 "invoice to the new account.",
           "supplier": "Corvo SA", "invoice_attached": True, "email_domain": "corvo-sa-finance.net",
           "known_supplier_domain": "corvosa.com", "callback_to_known_number": "not yet made",
           "queues": ["accounts payable", "supplier master data", "treasury", "fraud and security",
                      "procurement", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this email be routed to first?",
           "options": [{"value": "accounts payable"}, {"value": "supplier master data"},
                       {"value": "treasury"},
                       {"value": "fraud and security", "description": "Anything with the hallmarks of an attempted payment diversion."},
                       {"value": "procurement"}, {"value": "other"}]},
 "target": "fraud and security", "abstention_cause": None,
 "rationale": "A bank-detail change arriving from a domain that does not match the supplier's known domain is a payment diversion attempt, which goes to fraud and security before any master data change."},
{
 "key": "ro06", "domain": "logistics",
 "state": "A driver calls the control room: 'The tail lift will not lower. I am at the customer's yard with a full "
          "load and they have a fork lift but no ramp.' He adds that the load is secure, he is not blocking the "
          "yard and he feels fine. The control room routes calls to fleet maintenance, customer service, route "
          "planning, health and safety, the transport manager, or other. Fleet maintenance owns anything wrong "
          "with the vehicle or its equipment, health and safety owns injuries and dangerous occurrences, and "
          "customer service owns the conversation with the receiving site once the cause is known.",
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this call be routed?",
           "options": [{"value": "fleet maintenance"}, {"value": "customer service"},
                       {"value": "route planning"}, {"value": "health and safety"},
                       {"value": "the transport manager"}, {"value": "other"}]},
 "target": "fleet maintenance", "abstention_cause": None,
 "rationale": "The immediate problem is a failed piece of vehicle equipment, which fleet maintenance owns; nobody is hurt and the delivery workaround follows from their answer."},
{
 "key": "ro07", "domain": "travel",
 "state": {"channel": "airline chat", "message": "My flight tomorrow morning has been cancelled by you and the "
                                                 "app will not let me pick the later flight. I need to be there "
                                                 "by Friday afternoon.",
           "booking_status": "cancelled by carrier", "rebooking_attempted": True, "payment_issue": False,
           "original_flight": "CX214, 07:35", "alternatives_same_day": 2, "fare_type": "Flex",
           "customer_tier": "none", "baggage_checked": False, "refund_requested": False,
           "note": "The cancellation was published by the carrier eleven hours ago and the customer has not "
                   "asked for her money back."},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this chat be routed to?",
           "options": [{"value": "disruption rebooking", "description": "Rebooking after a cancellation or delay caused by the carrier."},
                       {"value": "voluntary changes", "description": "The customer wants to change a flight that is still operating."},
                       {"value": "refunds"}, {"value": "baggage claims"},
                       {"value": "loyalty programme"}, {"value": "special assistance"}, {"value": "other"}]},
 "target": "disruption rebooking", "abstention_cause": None,
 "rationale": "The carrier cancelled the flight and the customer wants to be rebooked, which is disruption rebooking rather than a voluntary change or a refund."},
{
 "key": "ro08", "domain": "education",
 "state": "A student emails the faculty office: 'I have been unwell for two weeks and I cannot finish the "
          "coursework due on Friday. I have a note from the health centre.' The deadline has not yet passed and no "
          "mark has been issued for the work. The office routes to extensions and mitigating circumstances, "
          "timetabling, fees and funding, academic appeals, disability support, IT services, or other.",
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this email be routed?",
           "options": [{"value": "extensions and mitigating circumstances"}, {"value": "timetabling"},
                       {"value": "fees and funding"}, {"value": "academic appeals"},
                       {"value": "disability support"}, {"value": "IT services"}, {"value": "other"}]},
 "target": "extensions and mitigating circumstances", "abstention_cause": None,
 "rationale": "The student wants more time on an upcoming deadline because of illness and has evidence, which is a mitigating circumstances request rather than an appeal against a decision already made."},
{
 "key": "ro09", "domain": "retail",
 "state": {"channel": "store escalation form", "summary": "A customer says a kettle she bought eight months ago "
                                                          "has started leaking from the base. She has the "
                                                          "receipt and wants it put right.",
           "days_since_purchase": 243, "return_window_days": 28, "product_warranty_months": 24,
           "damage_from_misuse": False, "customer_wants_refund": False,
           "queues": ["returns", "product quality and warranty", "customer relations", "store operations",
                      "supplier claims", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should handle this escalation?",
           "options": [{"value": "returns", "description": "Items brought back inside the ordinary return window."},
                       {"value": "product quality and warranty", "description": "Faults arising after the return window but inside the warranty."},
                       {"value": "customer relations"}, {"value": "store operations"},
                       {"value": "supplier claims"}, {"value": "other"}]},
 "target": "product quality and warranty", "abstention_cause": None,
 "rationale": "At 243 days the 28-day return window has closed but the 24-month warranty is live, so this is a warranty fault rather than a return."},
{
 "key": "ro10", "domain": "software_incidents",
 "state": "A ticket reads: 'Customers on the enterprise plan are being charged twice for their monthly seats. Two "
          "accounts have confirmed it on their card statements.' The two accounts are on annual terms billed "
          "monthly in arrears, and neither reports any problem signing in or using the product. Nothing suggests "
          "unauthorised access. The queues are authentication, data pipelines, billing, the public API, the web "
          "front end, security, or other.",
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should take this ticket?",
           "options": [{"value": "authentication"}, {"value": "data pipelines"}, {"value": "billing"},
                       {"value": "public API"}, {"value": "web front end"}, {"value": "security"},
                       {"value": "other"}]},
 "target": "billing", "abstention_cause": None,
 "rationale": "Duplicate charges confirmed on customer card statements are a billing defect, with no sign of an authentication, pipeline or security issue."},
{
 "key": "ro11", "domain": "hr",
 "state": {"channel": "HR service desk", "request": "A new joiner writes that her first salary payment was 400 "
                                                    "short and that the payslip shows a tax code she does not "
                                                    "recognise.",
           "start_date": "2026-08-03", "first_pay_date": "2026-08-28", "benefits_enrolled": True,
           "grievance_raised": False, "training_query": False,
           "teams": ["payroll", "benefits", "employee relations", "recruitment", "learning and development",
                     "other"]},
 "field": {"id": "team", "type": "choice",
           "question": "Which HR team should take this request?",
           "options": [{"value": "payroll"}, {"value": "benefits"}, {"value": "employee relations"},
                       {"value": "recruitment"}, {"value": "learning and development"}, {"value": "other"}]},
 "target": "payroll", "abstention_cause": None,
 "rationale": "An underpaid salary and an unexpected tax code on a payslip are payroll matters."},
{
 "key": "ro12", "domain": "healthcare_admin",
 "state": "A message arrives through the patient portal: 'I am having chest pain that started twenty minutes ago "
          "and it is spreading to my arm.' The patient is 61, is registered with the practice and sent the "
          "message at 08:40 on a weekday. No appointment is booked and no clinician has opened the message yet. "
          "The portal routes to urgent clinical triage, appointment booking, the results line, prescriptions, "
          "registrations and records, billing and insurance, or other.",
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this message be routed?",
           "options": [{"value": "urgent clinical triage", "description": "Anything suggesting an immediate clinical risk."},
                       {"value": "appointment booking"}, {"value": "results line"},
                       {"value": "prescriptions"}, {"value": "registrations and records"},
                       {"value": "billing and insurance"}, {"value": "other"}]},
 "target": "urgent clinical triage", "abstention_cause": None,
 "rationale": "New chest pain radiating to the arm is an immediate clinical risk and goes straight to urgent triage rather than any administrative queue."},
{
 "key": "ro13", "domain": "finance_ops",
 "state": {"document": "inbound post", "type": "letter from a tax authority",
           "subject": "Notice of assessment for the year ended 31 December 2024",
           "amount_stated_eur": 18400, "deadline_days": 30, "entity": "the Irish subsidiary",
           "authority": "national revenue service", "prior_correspondence": "none on file",
           "payment_instructions_included": True, "appeal_rights_stated": True,
           "received_date": "2026-09-14",
           "queues": ["accounts payable", "tax", "treasury", "financial reporting", "legal", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this letter be routed to?",
           "options": [{"value": "accounts payable"}, {"value": "tax"}, {"value": "treasury"},
                       {"value": "financial reporting"}, {"value": "legal"}, {"value": "other"}]},
 "target": "tax", "abstention_cause": None,
 "rationale": "A notice of assessment from a tax authority for a prior year is a tax matter, whatever payment or appeal follows from it."},
{
 "key": "ro14", "domain": "logistics",
 "state": "An email from a customer reads: 'Your driver reversed into our gatepost this morning. Nobody was hurt "
          "but the post is broken and we want to know how you will fix it.' The driver has filed his own account, "
          "the vehicle is undamaged and in service, and no injury is reported by anyone. The inbox routes to "
          "claims and insurance, customer service, fleet maintenance, health and safety, the transport manager, "
          "or other. Health and safety owns injuries and dangerous occurrences.",
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this email be routed?",
           "options": [{"value": "claims and insurance", "description": "Third-party property damage arising from our operations."},
                       {"value": "customer service"}, {"value": "fleet maintenance"},
                       {"value": "health and safety"}, {"value": "the transport manager"},
                       {"value": "other"}]},
 "target": "claims and insurance", "abstention_cause": None,
 "rationale": "Damage to a third party's property caused by a company vehicle is a claim, and with nobody hurt the health and safety route is not engaged."},
{
 "key": "ro15", "domain": "travel",
 "state": {"channel": "airline chat", "message": "My suitcase did not come out at the carousel last night. I "
                                                 "filed a report at the desk and have a reference number but "
                                                 "nobody has contacted me.",
           "file_reference": "PIR-88120", "hours_since_arrival": 19, "flight_operated": True,
           "compensation_claimed": False,
           "queues": ["disruption rebooking", "voluntary changes", "refunds", "baggage claims",
                      "loyalty programme", "special assistance", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this chat be routed to?",
           "options": [{"value": "disruption rebooking"}, {"value": "voluntary changes"},
                       {"value": "refunds"}, {"value": "baggage claims"},
                       {"value": "loyalty programme"}, {"value": "special assistance"}, {"value": "other"}]},
 "target": "baggage claims", "abstention_cause": None,
 "rationale": "A missing bag with an existing property irregularity report is a baggage claim, not a rebooking or refund question."},
{
 "key": "ro16", "domain": "education",
 "state": "A student writes: 'The lecture capture system has not recorded any of my module's sessions this term "
          "and the links in the virtual learning environment are dead.' The student is not asking for an "
          "extension, is not appealing a mark and has no access adjustment on file. The office routes to "
          "extensions and mitigating circumstances, timetabling, fees and funding, academic appeals, disability "
          "support, IT services, or other.",
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this message be routed?",
           "options": [{"value": "extensions and mitigating circumstances"}, {"value": "timetabling"},
                       {"value": "fees and funding"}, {"value": "academic appeals"},
                       {"value": "disability support"}, {"value": "IT services"}, {"value": "other"}]},
 "target": "IT services", "abstention_cause": None,
 "rationale": "A recording system that is not capturing sessions and broken links in the virtual learning environment are IT faults."},
{
 "key": "ro17", "domain": "retail",
 "state": {"channel": "social media mention", "text": "Just found a piece of metal in a jar of your own-brand "
                                                      "pasta sauce. Batch code on the lid is L2291.",
           "photo_attached": True, "product_type": "food", "injury_reported": False,
           "batch_still_on_shelf": True, "press_enquiries_received": 0,
           "queues": ["product safety and recall", "customer relations", "returns",
                      "product quality and warranty", "press office", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this mention be routed to first?",
           "options": [{"value": "product safety and recall", "description": "A foreign body or contamination in a food product, which may affect a whole batch."},
                       {"value": "customer relations"}, {"value": "returns"},
                       {"value": "product quality and warranty"}, {"value": "press office"},
                       {"value": "other"}]},
 "target": "product safety and recall", "abstention_cause": None,
 "rationale": "A foreign body in a food product with a batch code still on shelf is a safety matter affecting a whole batch, which takes priority over the goodwill, returns and press routes."},
{
 "key": "ro18", "domain": "software_incidents",
 "state": "A report arrives from a researcher: 'Your password reset endpoint accepts a token belonging to a "
          "different account, which lets me take over any user. Proof of concept attached.' The researcher is "
          "outside the company, has published nothing and is asking for a disclosure contact. The endpoint sits "
          "in the authentication service and is exposed through the public API. The queues are authentication, "
          "data pipelines, billing, the public API, the web front end, security, or other.",
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should take this report?",
           "options": [{"value": "authentication"}, {"value": "data pipelines"}, {"value": "billing"},
                       {"value": "public API"}, {"value": "web front end"},
                       {"value": "security", "description": "Reported vulnerabilities and suspected compromise, whichever component they touch."},
                       {"value": "other"}]},
 "target": "security", "abstention_cause": None,
 "rationale": "An externally reported account takeover vulnerability is a security report, and the security queue is defined to take vulnerabilities whichever component they touch."},
{
 "key": "ro19", "domain": "hr",
 "state": {"channel": "HR service desk", "request": "A manager asks how to advertise a newly approved headcount "
                                                    "for a data analyst and what the interview process should "
                                                    "be.",
           "requisition_approved": True, "headcount_budgeted": True, "existing_employee_involved": False,
           "pay_query": False, "training_query": False,
           "teams": ["payroll", "benefits", "employee relations", "recruitment", "learning and development",
                     "other"]},
 "field": {"id": "team", "type": "choice",
           "question": "Which HR team should take this request?",
           "options": [{"value": "payroll"}, {"value": "benefits"}, {"value": "employee relations"},
                       {"value": "recruitment"}, {"value": "learning and development"}, {"value": "other"}]},
 "target": "recruitment", "abstention_cause": None,
 "rationale": "Advertising an approved role and designing its interview process is recruitment's work."},
{
 "key": "ro20", "domain": "healthcare_admin",
 "state": "A caller says: 'I have moved from another town and I need to join your practice. I have my patient "
          "number and proof of my new address.' She adds that she has no immediate health concern, is not asking "
          "for an appointment today, and has a repeat prescription she will need transferring once she is "
          "registered. The practice routes to urgent clinical triage, appointment booking, the results line, "
          "prescriptions, registrations and records, billing and insurance, or other.",
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this call be routed?",
           "options": [{"value": "urgent clinical triage"}, {"value": "appointment booking"},
                       {"value": "results line"}, {"value": "prescriptions"},
                       {"value": "registrations and records"}, {"value": "billing and insurance"},
                       {"value": "other"}]},
 "target": "registrations and records", "abstention_cause": None,
 "rationale": "Joining the practice as a new patient with identity and address evidence is a registration task, and the prescription transfer follows once she is registered."},
{
 "key": "ro21", "domain": "finance_ops",
 "state": {"channel": "inbound email", "text": "Attached is our statement showing three invoices you have not "
                                               "paid. Two of them we can see were settled; the third we cannot "
                                               "find in your remittances.",
           "supplier": "Ferris Parts", "statement_attached": True, "dispute_raised": False,
           "bank_details_change_requested": False, "sender_domain_matches_supplier": True,
           "queues": ["accounts payable", "tax", "treasury", "financial reporting", "procurement", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this email be routed to?",
           "options": [{"value": "accounts payable", "description": "Supplier statements, remittance queries and unpaid invoices."},
                       {"value": "tax"}, {"value": "treasury"}, {"value": "financial reporting"},
                       {"value": "procurement"}, {"value": "other"}]},
 "target": "accounts payable", "abstention_cause": None,
 "rationale": "A supplier statement reconciliation over unmatched invoices and remittances is accounts payable work, and nothing here looks like a diversion attempt."},
{
 "key": "ro22", "domain": "logistics",
 "state": "A message arrives: 'We need to add a second weekly collection from our Swindon site from next month "
          "and want a price for it.' The customer is on an existing contract covering one weekly collection from "
          "that site, and the contract says any change to the collection schedule is priced and documented before "
          "it starts. The inbox routes to claims and insurance, customer service, route planning, commercial and "
          "pricing, fleet maintenance, or other.",
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this message be routed?",
           "options": [{"value": "claims and insurance"}, {"value": "customer service"},
                       {"value": "route planning"},
                       {"value": "commercial and pricing", "description": "New or changed services that need a quotation or a contract change."},
                       {"value": "fleet maintenance"}, {"value": "other"}]},
 "target": "commercial and pricing", "abstention_cause": None,
 "rationale": "The customer wants a price for an additional service, which the contract says must be priced and documented before it starts, so it is a commercial change rather than operational scheduling."},
{
 "key": "ro23", "domain": "travel",
 "state": {"channel": "airline chat", "message": "I use a wheelchair and need assistance from check-in to the "
                                                 "gate on Thursday, and my chair needs to travel in the cabin if "
                                                 "possible.",
           "departure_days_away": 4, "existing_assistance_request": False, "booking_confirmed": True,
           "flight_operating_normally": True, "bag_lost": False,
           "queues": ["disruption rebooking", "voluntary changes", "refunds", "baggage claims",
                      "loyalty programme", "special assistance", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this chat be routed to?",
           "options": [{"value": "disruption rebooking"}, {"value": "voluntary changes"},
                       {"value": "refunds"}, {"value": "baggage claims"},
                       {"value": "loyalty programme"}, {"value": "special assistance"}, {"value": "other"}]},
 "target": "special assistance", "abstention_cause": None,
 "rationale": "Mobility assistance through the airport and carriage of a wheelchair is special assistance work, not a baggage claim."},
{
 "key": "ro24", "domain": "education",
 "state": "A student writes: 'My tuition instalment came out twice this month and my bank has confirmed both "
          "payments left my account.' She adds that her direct debit mandate is unchanged, that she has not asked "
          "for a refund yet and that she is not disputing the amount of the instalment itself. The office routes "
          "to extensions and mitigating circumstances, timetabling, fees and funding, academic appeals, "
          "disability support, IT services, or other.",
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this message be routed?",
           "options": [{"value": "extensions and mitigating circumstances"}, {"value": "timetabling"},
                       {"value": "fees and funding"}, {"value": "academic appeals"},
                       {"value": "disability support"}, {"value": "IT services"}, {"value": "other"}]},
 "target": "fees and funding", "abstention_cause": None,
 "rationale": "A duplicated tuition instalment is a fees matter for the fees and funding team."},
{
 "key": "ro25", "domain": "retail",
 "state": {"channel": "customer email", "subject": "Gift card balance",
           "body": "I was given a 50 pound gift card in 2019 and the website says the balance is zero. I have "
                   "never spent it. Can you tell me what happened to it?",
           "gift_card_terms": "balances are held indefinitely and do not expire",
           "card_registered": False, "order_attached": False,
           "queues": ["payments and refunds", "gift cards and vouchers", "order accuracy", "returns",
                      "customer relations", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this email be routed to?",
           "options": [{"value": "payments and refunds"}, {"value": "gift cards and vouchers"},
                       {"value": "order accuracy"}, {"value": "returns"},
                       {"value": "customer relations"}, {"value": "other"}]},
 "target": "gift cards and vouchers", "abstention_cause": None,
 "rationale": "The question is about the balance and history of a gift card, which the gift cards and vouchers queue owns."},
{
 "key": "ro26", "domain": "software_incidents",
 "state": {"ticket": "T-9912", "reported_by": "a customer's operations team",
           "text": "The status page says everything is green but our dashboards show the marketing site returning "
                   "500 responses for the last hour. This is the public brochure site, not the product.",
           "verified": {"product_front_end": "healthy", "public_api": "healthy",
                        "brochure_site": "returning 500s from two regions"},
           "ownership_note": "The brochure site is built and run by the marketing engineering group and has its "
                             "own queue; the product front-end queue does not cover it.",
           "queues": ["authentication", "data pipelines", "billing", "public API", "web front end",
                      "marketing site", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should take this ticket?",
           "options": [{"value": "authentication"}, {"value": "data pipelines"}, {"value": "billing"},
                       {"value": "public API"}, {"value": "web front end"},
                       {"value": "marketing site", "description": "The public brochure site, which is run separately from the product front end."},
                       {"value": "other"}]},
 "target": "marketing site", "abstention_cause": None,
 "rationale": "The failing site is the public brochure site rather than the product front end, and that site has its own queue."},
{
 "key": "ro27", "domain": "hr",
 "state": {"channel": "HR service desk", "request": "An employee asks whether the company will fund a part-time "
                                                    "master's degree and what the study leave arrangements are.",
           "employee_service_months": 41, "budget_year_open": True, "pay_query": False,
           "grievance_raised": False, "role_change_requested": False,
           "teams": ["payroll", "benefits", "employee relations", "recruitment", "learning and development",
                     "other"]},
 "field": {"id": "team", "type": "choice",
           "question": "Which HR team should take this request?",
           "options": [{"value": "payroll"}, {"value": "benefits"}, {"value": "employee relations"},
                       {"value": "recruitment"},
                       {"value": "learning and development", "description": "Study sponsorship, qualifications and training."},
                       {"value": "other"}]},
 "target": "learning and development", "abstention_cause": None,
 "rationale": "Sponsorship of a qualification and the associated study leave sit with learning and development."},
{
 "key": "ro28", "domain": "logistics",
 "state": {"channel": "warehouse escalation", "summary": "A pallet of aerosols has been found leaking in the "
                                                         "inbound bay. The area has been cordoned off and nobody "
                                                         "is unwell.",
           "hazard_class": "2", "spill_contained": True, "injury": False, "vehicle_involved": False,
           "customer_notified": False,
           "queues": ["claims and insurance", "customer service", "health and safety", "fleet maintenance",
                      "site operations", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this escalation go to?",
           "options": [{"value": "claims and insurance"}, {"value": "customer service"},
                       {"value": "health and safety", "description": "Hazardous material incidents on site, whether or not anyone is hurt."},
                       {"value": "fleet maintenance"}, {"value": "site operations"}, {"value": "other"}]},
 "target": "health and safety", "abstention_cause": None,
 "rationale": "A leaking pallet of class 2 hazardous goods on site is a hazardous material incident, and that queue takes them whether or not anyone is hurt."},
{
 "key": "ro29", "domain": "healthcare_admin",
 "state": {"channel": "practice inbox", "message": "I need the letter you sent to my employer's occupational "
                                                   "health service about my sick leave, and I would like to know "
                                                   "who authorised sending it.",
           "request_type": "a copy of a disclosure already made, and the authority for it",
           "patient_registered": True, "clinical_question": False, "appointment_requested": False,
           "queues": ["urgent clinical triage", "appointment booking", "results line", "prescriptions",
                      "registrations and records", "information governance", "other"]},
 "field": {"id": "destination", "type": "choice",
           "question": "Where should this message be routed?",
           "options": [{"value": "urgent clinical triage"}, {"value": "appointment booking"},
                       {"value": "results line"}, {"value": "prescriptions"},
                       {"value": "registrations and records"},
                       {"value": "information governance", "description": "Disclosures of patient information to third parties and the authority for them."},
                       {"value": "other"}]},
 "target": "information governance", "abstention_cause": None,
 "rationale": "The patient is asking about a disclosure of her information to a third party and who authorised it, which is an information governance question rather than a records copy request."},
{
 "key": "ro30", "domain": "finance_ops",
 "state": {"channel": "shared inbox", "text": "Please action the attached.", "attachment": "unreadable scan",
           "sender": "an internal address with no signature block",
           "context_available": "none: the thread has no earlier messages, the attachment will not open, and the "
                                "sender has not replied to a request for clarification",
           "subject_line": "(blank)", "amounts_visible": False, "supplier_named": False,
           "queues": ["accounts payable", "tax", "treasury", "financial reporting", "procurement",
                      "fraud and security", "other"]},
 "field": {"id": "queue", "type": "choice",
           "question": "Which queue should this email be routed to?",
           "options": [{"value": "accounts payable"}, {"value": "tax"}, {"value": "treasury"},
                       {"value": "financial reporting"}, {"value": "procurement"},
                       {"value": "fraud and security"}, {"value": "other"}]},
 "target": None, "abstention_cause": "insufficient_evidence",
 "rationale": "The attachment will not open, the thread carries no context and the sender has not replied, so nothing in the state identifies the subject matter, including whether it belongs in the general queue."},
]
