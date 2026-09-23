"""long_policy: a policy of 8-15 rules in the state, one case, one permission/outcome question."""

ITEMS = [
{
 "key": "lp01", "domain": "retail",
 "state": {
  "policy_name": "Northbay Retail - In-Store Returns",
  "rules": [
   "R1. Unworn clothing may be returned within 30 days of purchase with a receipt.",
   "R2. Clothing returned between 31 and 60 days may only be exchanged or refunded to store credit.",
   "R3. Swimwear and underwear may not be returned once the hygiene seal is broken.",
   "R4. Items marked 'final sale' may not be returned or exchanged.",
   "R5. Items bought with a staff discount may be returned only to the store that sold them.",
   "R6. A return without a receipt needs a manager's approval and is always store credit.",
   "R7. Footwear may be returned within 30 days if the original box is included.",
   "R8. Electronics may be returned within 14 days, unopened.",
   "R9. Gift recipients may exchange but not refund.",
   "R10. No return may be accepted after 90 days under any circumstance."],
  "case": {"item": "linen shirt", "category": "clothing", "condition": "unworn, tags attached",
           "purchased_days_ago": 24, "receipt_present": True, "final_sale": False,
           "staff_discount": False, "purchase_store": "Northbay Central",
           "return_store": "Northbay Central", "gift": False}},
 "field": {"id": "refund_allowed", "type": "boolean",
           "question": "May the store refund this return to the original payment method?",
           "yes_description": "The policy permits a refund to the original payment method.",
           "no_description": "The policy does not permit that refund."},
 "target": True, "abstention_cause": None,
 "rationale": "R1 allows a receipted refund on unworn clothing inside 30 days and none of the exclusions (final sale, staff discount, gift, hygiene seal) applies at day 24."},
{
 "key": "lp02", "domain": "hr",
 "state": "Ashgrove Group - Annual Leave Carry-Over Policy (effective 1 January 2026)\n"
          "1. The leave year runs 1 January to 31 December.\n"
          "2. Full-time staff accrue 25 days of annual leave per leave year.\n"
          "3. Up to 5 unused days may be carried into the next leave year without approval.\n"
          "4. Carrying more than 5 days needs written approval from the department head before 15 December.\n"
          "5. Carried days expire on 31 March of the following year.\n"
          "6. Staff on long-term sick leave for more than eight consecutive weeks may carry up to 15 days, with HR approval.\n"
          "7. Staff serving notice may not carry any days; untaken leave is paid out at the contractual rate.\n"
          "8. Days bought under the leave purchase scheme may not be carried.\n"
          "9. Part-time staff carry a pro-rata share of the 5-day allowance.\n"
          "10. Public holidays are not annual leave and never carry.\n"
          "11. An approval given under rule 4 must name the number of days approved.\n\n"
          "Case: Priya is full-time and has 9 unused days on 31 December 2026. None of them were bought under the "
          "purchase scheme. She was not on long-term sick leave at any point in 2026, she is not serving notice, "
          "and she made no request to her department head in December.",
 "field": {"id": "carry_all_nine", "type": "boolean",
           "question": "May Priya carry all 9 unused days into 2027?"},
 "target": False, "abstention_cause": None,
 "rationale": "Rule 3 allows only 5 days without approval and rule 4 requires written approval before 15 December for more, which Priya never requested."},
{
 "key": "lp03", "domain": "logistics",
 "state": {
  "policy_name": "Meridian Freight - Mixed Load Rules",
  "rules": [
   "1. Class 3 flammable liquids may not travel in the same trailer as Class 5.1 oxidisers.",
   "2. Class 8 corrosives may share a trailer with Class 3 if separated by at least 1.2 m.",
   "3. Any load containing Class 1 explosives requires a dedicated trailer.",
   "4. Food-grade goods may not share a trailer with any hazardous class.",
   "5. A trailer carrying hazardous goods must display a placard for every class on board.",
   "6. The driver must hold an ADR certificate valid on the day of travel for the highest class carried.",
   "7. More than 1,000 kg of any single hazardous class requires a second driver on journeys over 6 hours.",
   "8. Class 2 gases must be loaded upright and secured to the trailer wall.",
   "9. No hazardous load may stand unattended in a public car park for more than 30 minutes.",
   "10. A trailer may carry at most three distinct hazardous classes."],
  "case": {"trailer": "T-114", "load": [{"class": "3", "kg": 400}, {"class": "8", "kg": 260}],
           "separation_m": 1.5, "food_grade_on_board": False, "placards": ["3", "8"],
           "driver_adr_valid_until": "2027-02-01", "travel_date": "2026-10-12",
           "journey_hours": 4, "unattended_public_car_park_minutes": 0}},
 "field": {"id": "compliant", "type": "boolean",
           "question": "Does this load comply with the mixed-load rules?"},
 "target": True, "abstention_cause": None,
 "rationale": "Only classes 3 and 8 are on board, separated by 1.5 m (rule 2), both placarded, each under 1,000 kg, the ADR certificate is valid on 12 October and no other rule is engaged."},
{
 "key": "lp04", "domain": "healthcare_admin",
 "state": "Lakeside Community Clinic - Missed Appointment Charge\n"
          "1. A patient who does not attend a booked appointment is charged a 30 GBP non-attendance fee.\n"
          "2. The fee is waived if the appointment is cancelled at least 24 hours in advance.\n"
          "3. The fee is waived if the patient was admitted to hospital on the day of the appointment.\n"
          "4. The fee is waived for patients under 18.\n"
          "5. The fee is waived for patients on the income-support register.\n"
          "6. The fee is never charged more than twice in a rolling 12 months.\n"
          "7. A first non-attendance by a patient registered for less than 90 days is waived.\n"
          "8. Appointments cancelled by the clinic never attract a fee.\n"
          "9. A waiver under rule 3 or rule 5 must be recorded by the reception supervisor.\n\n"
          "Case: Mr Oduya is 46 and has been registered with the clinic for six years. He is not on the "
          "income-support register. He booked a 09:00 appointment on 14 October and telephoned at 17:00 on "
          "13 October to cancel it. He was charged the fee once before, in March of the same year. He was not "
          "admitted to hospital, and the clinic did not cancel the appointment.",
 "field": {"id": "charge_fee", "type": "boolean",
           "question": "Should the 30 GBP fee be charged for this appointment?"},
 "target": True, "abstention_cause": None,
 "rationale": "The cancellation came 16 hours ahead, short of the 24 hours in rule 2, and no other waiver applies; rule 6 still allows a second charge in the year."},
{
 "key": "lp05", "domain": "finance_ops",
 "state": {
  "policy_name": "Halden Industries - Expense Reimbursement",
  "rules": [
   "1. Claims must be submitted within 60 days of the expense date.",
   "2. Any single item over 150 EUR requires an itemised receipt.",
   "3. Alcohol is never reimbursed.",
   "4. Client entertainment over 300 EUR requires prior written approval from a director.",
   "5. Taxi fares are reimbursed only where public transport was unavailable, or the journey began before 06:00 or ended after 23:00.",
   "6. Hotel rooms are reimbursed up to 180 EUR per night in tier-1 cities and 120 EUR per night elsewhere.",
   "7. Personal car mileage is reimbursed at 0.30 EUR per km.",
   "8. Claims are paid in the employee's payroll currency at the rate on the submission date.",
   "9. A claim containing any non-reimbursable line is returned in full to the claimant for correction.",
   "10. Expenses already charged to a company card may not be claimed again.",
   "11. A foreign per-diem replaces meal receipts and may not be claimed alongside meal lines."],
  "case": {"claimant": "E-2291", "submitted": "2026-09-20", "company_card_used": False,
           "per_diem_claimed": False,
           "lines": [{"item": "hotel, Lisbon (tier-2), 2 nights", "eur": 260, "receipt": True, "date": "2026-08-11"},
                     {"item": "dinner with client", "eur": 88, "receipt": True, "date": "2026-08-11"},
                     {"item": "bottle of wine with dinner", "eur": 26, "receipt": True, "date": "2026-08-11"},
                     {"item": "taxi, left client office 23:40", "eur": 19, "receipt": True, "date": "2026-08-11"}]}},
 "field": {"id": "payable_as_submitted", "type": "boolean",
           "question": "Can this claim be paid as submitted?"},
 "target": False, "abstention_cause": None,
 "rationale": "The wine line is alcohol, which rule 3 never reimburses, so rule 9 returns the whole claim; the 260 EUR for two tier-2 nights also exceeds the 120 EUR per night cap."},
{
 "key": "lp06", "domain": "software_incidents",
 "state": "Orbital Payments - Change Freeze Policy\n"
          "1. A change freeze runs from 20 December to 2 January inclusive.\n"
          "2. A freeze also runs for the 48 hours before and after any scheduled network migration.\n"
          "3. During a freeze no change may be deployed to production.\n"
          "4. Severity-1 incident fixes are exempt from the freeze.\n"
          "5. Severity-2 fixes are exempt only with the on-call incident commander's approval.\n"
          "6. Configuration-only changes to a feature flag are exempt if the flag is already live.\n"
          "7. An exempt change still requires two reviewers.\n"
          "8. Documentation and dashboard changes are not production changes.\n"
          "9. An exemption must be logged in the change record within one hour of deployment.\n"
          "10. Rolling back a change deployed before the freeze is always permitted.\n\n"
          "Case: On 27 December the payments team wants to deploy a fix for a Severity-2 defect that double-charges "
          "some customers. The on-call incident commander has approved the deployment in writing, two engineers have "
          "reviewed the pull request, and the change record is open and ready to be annotated.",
 "field": {"id": "may_deploy", "type": "boolean",
           "question": "May the team deploy this change on 27 December?"},
 "target": True, "abstention_cause": None,
 "rationale": "27 December falls in the freeze, but rule 5 exempts a Severity-2 fix with the incident commander's approval and rule 7's two reviewers are in place."},
{
 "key": "lp07", "domain": "travel",
 "state": {
  "policy_name": "Cirrus Air - Cabin Baggage",
  "rules": [
   "1. Each passenger may bring one cabin bag up to 55 x 40 x 20 cm and 8 kg.",
   "2. Each passenger may also bring one personal item up to 40 x 30 x 15 cm.",
   "3. Basic fares include the personal item only; the cabin bag must be purchased.",
   "4. Flex and Business fares include both at no charge.",
   "5. Cirrus Club members at Silver tier or above include the cabin bag on any fare.",
   "6. A musical instrument longer than 55 cm requires a purchased seat.",
   "7. Duty-free bought after security does not count toward the allowance.",
   "8. An infant under 2 travelling on a lap has no cabin bag allowance.",
   "9. A cabin bag over 8 kg goes to the hold for a fee at the gate.",
   "10. An umbrella, a coat and one small bag of food are not counted."],
  "case": {"passenger": "A. Renko", "fare": "Basic", "club_tier": "Bronze",
           "cabin_bag": {"cm": [54, 39, 20], "kg": 7.2, "purchased": False},
           "personal_item": {"cm": [38, 28, 14]}, "duty_free_bag": True, "infant_on_lap": False}},
 "field": {"id": "free_cabin_bag", "type": "boolean",
           "question": "May this passenger take the cabin bag into the cabin at no extra charge?"},
 "target": False, "abstention_cause": None,
 "rationale": "The bag is within size and weight, but on a Basic fare rule 3 requires it to be purchased and Bronze tier is below the Silver threshold in rule 5."},
{
 "key": "lp08", "domain": "education",
 "state": "Westmere University - Late Submission and Extensions (Undergraduate)\n"
          "1. Coursework is due at 14:00 on the stated date.\n"
          "2. Work submitted up to 24 hours late is capped at the pass mark of 40.\n"
          "3. Work submitted between 24 and 120 hours late receives zero but is recorded as submitted.\n"
          "4. Work submitted more than 120 hours late is not marked.\n"
          "5. A student may self-certify one extension of up to 5 working days per module per year without evidence.\n"
          "6. A self-certification must be filed before the deadline.\n"
          "7. An extension beyond 5 working days needs documented mitigating circumstances approved by the board.\n"
          "8. Self-certified extensions do not apply to examinations or to group work.\n"
          "9. A cap is not applied where an approved extension covers the submission.\n"
          "10. Resubmission is offered only where the board finds a procedural error.\n\n"
          "Case: Tomas submitted the individual essay for module WM214 at 09:30 on the second working day after the "
          "14:00 deadline. He filed a self-certified 5-working-day extension at 11:00 on the day before the deadline, "
          "and he had not used self-certification on this module before.",
 "field": {"id": "cap_at_forty", "type": "boolean",
           "question": "Should Tomas's essay be capped at 40?"},
 "target": False, "abstention_cause": None,
 "rationale": "The self-certification was filed before the deadline on individual work and covers the second working day, so rule 9 removes the cap."},
{
 "key": "lp09", "domain": "retail",
 "state": {
  "policy_name": "Fairhill Electronics - Price Match Promise",
  "rules": [
   "1. We match the advertised price of any retailer on the approved competitor list.",
   "2. The item must be identical: same brand, model number, colour and UK specification.",
   "3. The competitor must have it in stock for immediate delivery or collection.",
   "4. Marketplace sellers, auction sites and membership-only clubs are not matched.",
   "5. Clearance, ex-display and refurbished prices are not matched.",
   "6. A match may be claimed up to 14 days after purchase, with the receipt.",
   "7. Bundles and multi-buy offers are not matched.",
   "8. A matched price may not fall below our cost price; such claims are declined.",
   "9. One match per customer per item per 30 days.",
   "10. A match is paid as a refund of the difference to the original payment method."],
  "case": {"item": {"brand": "Kessel", "model": "KX-880", "colour": "graphite", "spec": "UK"},
           "purchase_date": "2026-09-12", "claim_date": "2026-09-20",
           "our_price_gbp": 429, "our_cost_gbp": 362,
           "competitor": {"name": "Brightway", "on_approved_list": True, "marketplace_seller": False,
                          "price_gbp": 399, "stock": "in stock for collection", "condition": "new",
                          "offer_type": "single item"},
           "matches_on_this_item_in_last_30_days": 0}},
 "field": {"id": "honour_match", "type": "boolean",
           "question": "Should Fairhill honour this price-match claim?"},
 "target": True, "abstention_cause": None,
 "rationale": "The claim is 8 days after purchase against an identical new item in stock at an approved competitor, and 399 GBP is still above the 362 GBP cost floor in rule 8."},
{
 "key": "lp10", "domain": "hr",
 "state": "Calder Bank - Hybrid Working Standard\n"
          "1. Colleagues in eligible roles may work remotely up to 2 days a week.\n"
          "2. Eligible roles are those not listed in Annex A (branch, cash handling, vault and regulated trading roles).\n"
          "3. A remote day must be agreed with the line manager and recorded in the rota by the Thursday of the previous week.\n"
          "4. Colleagues in their first 12 weeks work on site every day.\n"
          "5. Remote work from outside the country of employment is not permitted.\n"
          "6. A colleague under a formal performance plan works on site every day for its duration.\n"
          "7. A team on-site day named by the department head overrides an individual remote day.\n"
          "8. Equipment for remote work is issued by IT and may not be substituted.\n"
          "9. Remote work is not a contractual right and may be withdrawn on 48 hours' notice.\n"
          "10. Customer appointments are held on site unless the customer asks otherwise.\n\n"
          "Case: Naomi is a credit analyst, a role not listed in Annex A. She joined seven months ago and is not on a "
          "performance plan. On Monday she asked her line manager to work remotely on the Wednesday of that same "
          "week; the rota for that week had closed on the previous Thursday. Her team has no named on-site day that "
          "Wednesday and she would work from her home in the country of employment.",
 "field": {"id": "remote_allowed", "type": "boolean",
           "question": "Under the standard, may Naomi work remotely on that Wednesday?"},
 "target": False, "abstention_cause": None,
 "rationale": "Rule 3 requires the remote day to be in the rota by the Thursday of the previous week, and Naomi asked only on the Monday of the week itself."},
{
 "key": "lp11", "domain": "logistics",
 "state": {
  "policy_name": "Port Hollan - Reefer Container Yard Standard",
  "rules": [
   "1. This standard governs refrigerated containers operating between -25 C and +12 C.",
   "2. Frozen cargo (-25 C to -12 C) is plugged in within 30 minutes of arrival.",
   "3. Chilled cargo (-2 C to +12 C) is plugged in within 60 minutes of arrival.",
   "4. Every plugged container is inspected twice a day and the set point logged.",
   "5. A container whose set point drifts more than 2 C is moved to a monitored bay.",
   "6. Genset-powered containers may stand unplugged for up to 8 hours.",
   "7. Stacking of plugged containers is limited to three high.",
   "8. A container without a valid pre-trip inspection is refused.",
   "9. Cargo types outside the temperature range in rule 1 are handled under the yard's separate special-cargo procedure, which is not reproduced in this standard and is not summarised here.",
   "10. Nothing in this standard is a permission or a refusal for cargo covered by rule 9."],
  "case": {"container": "PHRU-5512099", "cargo": "clinical trial vaccine", "set_point_c": -70,
           "unit_type": "ultra-low-temperature cryogenic reefer", "pre_trip_inspection": "valid",
           "arrival": "2026-11-02T04:10Z", "requested": "plug-in storage in the reefer yard"}},
 "field": {"id": "accept_container", "type": "boolean",
           "question": "Does this standard allow the yard to accept this container for plug-in storage?",
           "yes_description": "The standard permits acceptance.",
           "no_description": "The standard refuses acceptance."},
 "target": None, "abstention_cause": "insufficient_evidence",
 "rationale": "At -70 C the cargo falls outside the range this standard governs, and rules 9 and 10 expressly send it to a separate procedure that is not reproduced, so the standard neither permits nor refuses it."},
{
 "key": "lp12", "domain": "healthcare_admin",
 "state": "Ridgeway Trust - Release of Medical Records\n"
          "1. A patient may request a copy of their own record at no charge.\n"
          "2. A request must be in writing and the requester's identity verified.\n"
          "3. Records are provided within one calendar month of a valid request.\n"
          "4. The period may be extended by two months for complex requests, with notice to the requester.\n"
          "5. A third party may receive records only with the patient's written consent or a court order.\n"
          "6. A parent may request the record of a child under 13.\n"
          "7. A child aged 13 to 15 may request their own record if judged competent by a clinician.\n"
          "8. Information that would cause serious harm to the patient may be withheld on a clinician's decision.\n"
          "9. Information identifying a third party is redacted unless that person consents.\n"
          "10. A request may be refused as manifestly unfounded only with the Caldicott Guardian's agreement.\n\n"
          "Case: Mrs Achebe asked in writing for a copy of her own record on 2 September and her identity was "
          "verified at the desk the same day. The record is 40 pages, contains no third-party identifying "
          "information, and the responsible clinician has raised no concern about harm. No extension notice has been "
          "sent. It is now 25 September and the copy has not yet been posted.",
 "field": {"id": "within_time", "type": "boolean",
           "question": "On 25 September, is the trust still inside the time this policy allows for the request?"},
 "target": True, "abstention_cause": None,
 "rationale": "Rule 3 gives one calendar month from the valid request of 2 September, which runs to 2 October, so 25 September is still inside the period."},
{
 "key": "lp13", "domain": "finance_ops",
 "state": {
  "policy_name": "Vantor Group - Purchase Approval Matrix",
  "rules": [
   "1. Purchases up to 5,000 USD are approved by the team lead.",
   "2. Purchases over 5,000 and up to 50,000 USD are approved by the department director.",
   "3. Purchases over 50,000 and up to 250,000 USD are approved by the CFO.",
   "4. Purchases over 250,000 USD are approved by the board.",
   "5. A purchase from a supplier not on the approved register is approved one tier above its value tier.",
   "6. Software that will hold customer data is approved one tier above its value tier, in addition to any other uplift.",
   "7. A renewal at the same price as the prior term is approved at its value tier, with no uplift.",
   "8. Splitting a purchase to stay under a tier is prohibited and voids the approval.",
   "9. Approval is recorded in the purchasing system before the order is placed.",
   "10. Board is the highest tier; any further uplift stays at board."],
  "case": {"request": "PR-4471", "description": "new analytics platform, first term",
           "amount_usd": 42000, "supplier": {"name": "Dataline", "on_approved_register": False},
           "will_hold_customer_data": True, "renewal": False, "split_from_larger_order": False}},
 "field": {"id": "approver", "type": "choice",
           "question": "Who must approve PR-4471?",
           "options": [{"value": "team lead"}, {"value": "department director"},
                       {"value": "CFO"}, {"value": "board"}]},
 "target": "board", "abstention_cause": None,
 "rationale": "42,000 USD is the director tier, one uplift for the unregistered supplier makes it CFO and the customer-data uplift in rule 6 adds a second, reaching board."},
{
 "key": "lp14", "domain": "software_incidents",
 "state": "Nimbus Cloud - Escalation Runbook\n"
          "1. A customer-visible outage on a paid tier is Sev1.\n"
          "2. Degraded performance affecting more than 10 percent of requests is Sev2.\n"
          "3. A single-customer issue with a workaround is Sev3.\n"
          "4. Sev1 pages the incident commander and the duty VP immediately.\n"
          "5. Sev2 pages the incident commander; the duty VP is told at the 60-minute mark if it is unresolved.\n"
          "6. Sev3 is queued to the owning team in business hours.\n"
          "7. An incident with suspected data loss is Sev1 whatever its scope.\n"
          "8. An incident with suspected unauthorised access goes to the security on-call instead of the normal path.\n"
          "9. An incident confined to the free tier is capped at Sev3 unless rule 7 or rule 8 applies.\n"
          "10. Severity may be raised after paging but never lowered.\n\n"
          "Case: At 02:10 the platform team sees elevated 500 responses on the free tier only, affecting roughly 30 "
          "percent of free-tier requests. Paid tiers are healthy. There is no sign of data loss and no sign of "
          "unauthorised access.",
 "field": {"id": "escalation", "type": "choice",
           "question": "Which escalation path applies at 02:10?",
           "options": [{"value": "page the incident commander and the duty VP"},
                       {"value": "page the incident commander only"},
                       {"value": "queue to the owning team in business hours"},
                       {"value": "route to the security on-call"},
                       {"value": "other"}]},
 "target": "queue to the owning team in business hours", "abstention_cause": None,
 "rationale": "The fault is confined to the free tier with no data loss or unauthorised access, so rule 9 caps it at Sev3 and rule 6 queues Sev3 to the owning team in business hours."},
{
 "key": "lp15", "domain": "travel",
 "state": {
  "policy_name": "Talon Rail - Ticket Change Rules",
  "rules": [
   "1. An Advance ticket may be changed up to 18:00 on the day before travel for a 15 EUR fee plus any fare difference.",
   "2. An Advance ticket may not be changed after that time and has no refund value.",
   "3. An Off-Peak ticket may be changed free of charge before departure, paying any fare difference.",
   "4. An Anytime ticket may be changed or refunded free of charge within 28 days of the travel date.",
   "5. A ticket bought with a Railcard may only be changed to a journey on which the same Railcard is valid.",
   "6. Where Talon Rail cancels the train, no change fee applies to any ticket type.",
   "7. A Season ticket is not changed; it is refunded pro rata with a 10 EUR administration charge.",
   "8. A group ticket for 10 or more is changed at 5 EUR per person.",
   "9. A change made at a staffed counter attracts no additional booking charge.",
   "10. Fees and fare differences are charged per passenger, per change."],
  "case": {"booking": "TR-99213", "ticket_type": "Advance", "passengers": 2,
           "travel_date": "2026-11-04", "change_requested_at": "2026-11-03T16:40",
           "railcard": None, "operator_cancellation": False,
           "fare_difference_eur_per_passenger": 8, "channel": "staffed counter"}},
 "field": {"id": "amount_due", "type": "choice",
           "question": "What should the two passengers be charged in total for this change?",
           "options": [{"value": "no charge"}, {"value": "8 EUR"}, {"value": "23 EUR"},
                       {"value": "30 EUR"}, {"value": "38 EUR"}, {"value": "46 EUR"}]},
 "target": "46 EUR", "abstention_cause": None,
 "rationale": "The request at 16:40 the day before travel is inside rule 1, so each passenger pays the 15 EUR fee plus the 8 EUR fare difference, and rule 10 makes that 46 EUR for two."},
{
 "key": "lp16", "domain": "education",
 "state": "Brantwood College - Year 1 Progression Rules\n"
          "1. A module is passed with a mark of 40 or more.\n"
          "2. A student progresses clear to Year 2 having passed at least 100 of 120 credits.\n"
          "3. A student who fails no more than 20 credits may progress carrying those credits.\n"
          "4. A student who fails more than 20 but no more than 40 credits is offered reassessment in August.\n"
          "5. A student who fails more than 40 credits repeats the year.\n"
          "6. A reassessment mark is capped at 40.\n"
          "7. A student who does not sit an offered reassessment fails the year.\n"
          "8. Mitigating circumstances accepted by the board convert a fail into a deferred first attempt, uncapped.\n"
          "9. A student may repeat the year once only.\n"
          "10. A deferred attempt under rule 8 does not count toward the carried-credit limit in rule 3.\n\n"
          "Case: Ines attempted 120 credits in Year 1. She passed 90 credits and failed two 15-credit modules. The "
          "board accepted mitigating circumstances for one of the two failed modules. She has not repeated a year "
          "before.",
 "field": {"id": "outcome", "type": "choice",
           "question": "What is the board's outcome for Ines?",
           "options": [{"value": "progress clear"}, {"value": "progress carrying credits"},
                       {"value": "reassessment in August"}, {"value": "repeat the year"},
                       {"value": "fail the year"}]},
 "target": "progress carrying credits", "abstention_cause": None,
 "rationale": "One of the two 15-credit fails becomes a deferred attempt that rule 10 excludes from the carried-credit count, leaving 15 carried credits, which rule 3 allows while rule 2 rules out a clear progression at 90 passed."},
{
 "key": "lp17", "domain": "retail",
 "state": {
  "policy_name": "Marle & Co - Refund Method",
  "rules": [
   "1. A refund goes to the original payment method wherever that method can accept it.",
   "2. A cash purchase under 50 GBP is refunded in cash.",
   "3. A cash purchase of 50 GBP or more is refunded by bank transfer within 5 working days.",
   "4. A card purchase is refunded to the same card; if the card has expired, a bank transfer is used.",
   "5. A gift-card purchase is refunded to a new gift card.",
   "6. A purchase made with a mix of methods is refunded proportionally to each method.",
   "7. A return without a receipt is refunded as store credit at the lowest price the item sold for in 90 days.",
   "8. An online order returned to a store follows the online payment method.",
   "9. A refund for an online order is never issued in cash.",
   "10. Store credit does not expire."],
  "case": {"order": "M-77120", "channel": "online", "returned_in": "Marle & Co Leeds",
           "receipt": True, "refund_total_gbp": 85,
           "payment": [{"method": "gift card", "gbp": 20},
                       {"method": "card ending 4417", "gbp": 65, "card_status": "expired"}]}},
 "field": {"id": "refund_method", "type": "choice",
           "question": "How should the 85 GBP be refunded?",
           "options": [{"value": "85 GBP in cash"},
                       {"value": "85 GBP to a new gift card"},
                       {"value": "85 GBP by bank transfer"},
                       {"value": "85 GBP as store credit"},
                       {"value": "20 GBP to a new gift card and 65 GBP in cash"},
                       {"value": "20 GBP to a new gift card and 65 GBP by bank transfer"}]},
 "target": "20 GBP to a new gift card and 65 GBP by bank transfer", "abstention_cause": None,
 "rationale": "Rule 8 keeps the online payment methods, rule 6 splits the refund in proportion, rule 5 sends the gift-card share to a new gift card and rule 4 sends the expired-card share to a bank transfer."},
{
 "key": "lp18", "domain": "hr",
 "state": "Kerrow Logistics - Disciplinary Procedure\n"
          "1. Minor misconduct is addressed first with an informal conversation recorded by the manager.\n"
          "2. A repeat of the same minor misconduct within 6 months moves to a first written warning.\n"
          "3. A first written warning is live for 6 months.\n"
          "4. Further misconduct while a first written warning is live moves to a final written warning.\n"
          "5. A final written warning is live for 12 months.\n"
          "6. Gross misconduct may result in dismissal without prior warnings.\n"
          "7. Gross misconduct includes theft, violence, falsifying records and driving under the influence.\n"
          "8. The employee may be accompanied at any formal hearing.\n"
          "9. A warning that is no longer live is disregarded when choosing the next step.\n"
          "10. Lateness is minor misconduct unless it causes a missed delivery window, in which case it is a first written warning offence in itself.\n\n"
          "Case: Dev received a first written warning for lateness on 3 February 2026. On 2 September 2026 he was "
          "late again; no delivery window was missed. Nothing else sits on his file and no gross misconduct is "
          "alleged.",
 "field": {"id": "next_step", "type": "choice",
           "question": "What is the correct next step for the lateness on 2 September?",
           "options": [{"value": "no action"}, {"value": "informal recorded conversation"},
                       {"value": "first written warning"}, {"value": "final written warning"},
                       {"value": "dismissal"}]},
 "target": "informal recorded conversation", "abstention_cause": None,
 "rationale": "The February warning stopped being live on 3 August and rule 9 disregards it, so the lateness with no missed window is minor misconduct starting again at rule 1."},
{
 "key": "lp19", "domain": "logistics",
 "state": {
  "policy_name": "Aster Parcels - Service Selection Rules",
  "rules": [
   "1. A parcel up to 2 kg with combined dimensions up to 60 cm goes Standard.",
   "2. A parcel over 2 kg or over 60 cm combined goes Large.",
   "3. Anything over 30 kg goes Freight.",
   "4. A parcel containing lithium batteries may not go Standard; it goes Large or Freight.",
   "5. Goods with a declared value over 500 EUR require Insured, whatever the size.",
   "6. Perishables go Chilled, and frozen goods go Frozen, whatever the size.",
   "7. Insured overrides Large, but not Freight, Chilled or Frozen.",
   "8. Freight overrides everything except Frozen.",
   "9. A delivery to an island adds two days and does not change the service.",
   "10. A parcel needing two services that do not combine is split into two consignments."],
  "case": {"consignment": "AP-3318", "weight_kg": 6.5, "combined_cm": 72,
           "contents": "cordless drill with lithium battery", "declared_value_eur": 640,
           "perishable": False, "frozen": False, "destination": "Isle of Arran"}},
 "field": {"id": "service", "type": "choice",
           "question": "Which service must this consignment travel on?",
           "options": [{"value": "Standard"}, {"value": "Large"}, {"value": "Freight"},
                       {"value": "Insured"}, {"value": "Chilled"}, {"value": "Frozen"}]},
 "target": "Insured", "abstention_cause": None,
 "rationale": "Weight and size put it in Large, the 640 EUR declared value requires Insured under rule 5, and rule 7 makes Insured override Large while nothing triggers Freight, Chilled or Frozen."},
{
 "key": "lp20", "domain": "healthcare_admin",
 "state": "Sunland Health Plan - Outpatient Cost Share (2026)\n"
          "1. A primary care visit costs 25 USD.\n"
          "2. A specialist visit costs 50 USD.\n"
          "3. An urgent care visit costs 75 USD.\n"
          "4. An emergency department visit costs 300 USD, waived if the visit results in admission.\n"
          "5. Preventive services on the federal list cost nothing.\n"
          "6. A specialist visit referred by the member's primary care physician is charged at the primary care rate.\n"
          "7. A telehealth visit is charged at half the rate of the equivalent in-person service.\n"
          "8. Out-of-network care is charged at 50 percent coinsurance after the deductible.\n"
          "9. The annual out-of-pocket maximum is 4,000 USD, after which cost share stops.\n"
          "10. A behavioural health visit is charged at the primary care rate.\n\n"
          "Case: Ms Diaz saw an in-network dermatologist by video on 4 May. Her primary care physician had referred "
          "her. The visit was not a preventive service on the federal list. Her out-of-pocket total for the year "
          "before this visit was 900 USD and she had already met her deductible.",
 "field": {"id": "cost_share", "type": "choice",
           "question": "What is Ms Diaz's cost share for the 4 May visit?",
           "options": [{"value": "0 USD"}, {"value": "12.50 USD"}, {"value": "25 USD"},
                       {"value": "37.50 USD"}, {"value": "50 USD"}, {"value": "75 USD"}]},
 "target": "12.50 USD", "abstention_cause": None,
 "rationale": "Rule 6 drops the referred specialist visit to the 25 USD primary care rate and rule 7 halves it for telehealth, with the in-network status making rule 8 irrelevant."},
{
 "key": "lp21", "domain": "finance_ops",
 "state": {
  "policy_name": "Lumen Capital - Client Onboarding Evidence",
  "rules": [
   "1. Every new client provides proof of identity and proof of address.",
   "2. A company client also provides a certificate of incorporation.",
   "3. A company client with a beneficial owner holding 25 percent or more provides identity evidence for that owner.",
   "4. A trust provides the trust deed and evidence for the trustees.",
   "5. A politically exposed person requires senior management sign-off on top of the standard evidence.",
   "6. A client in a high-risk jurisdiction requires source-of-funds evidence.",
   "7. Source-of-wealth evidence is required only where the expected relationship exceeds 1,000,000 EUR.",
   "8. Evidence certified within the last 3 months is accepted; older certification is refreshed.",
   "9. A client introduced by a regulated intermediary may rely on the intermediary's identity evidence, but never on its address evidence.",
   "10. No account is opened until every required item is on file."],
  "case": {"client": "Renwick Holdings", "type": "company", "jurisdiction_risk": "low",
           "introduced_by": "regulated intermediary",
           "on_file": ["certificate of incorporation",
                       "identity evidence for the directors, certified last month, supplied by the intermediary",
                       "identity evidence for the sole beneficial owner (60 percent), certified last month"],
           "beneficial_owner_pep": False, "expected_relationship_eur": 400000}},
 "field": {"id": "outstanding_item", "type": "choice",
           "question": "Which item is still outstanding before this account may be opened?",
           "options": [{"value": "proof of address"}, {"value": "certificate of incorporation"},
                       {"value": "source-of-funds evidence"}, {"value": "source-of-wealth evidence"},
                       {"value": "senior management sign-off"}, {"value": "nothing is outstanding"}]},
 "target": "proof of address", "abstention_cause": None,
 "rationale": "Rule 1 still demands proof of address and rule 9 forbids relying on the intermediary for it, while the low-risk jurisdiction and 400,000 EUR size switch off rules 6 and 7."},
{
 "key": "lp22", "domain": "software_incidents",
 "state": {
  "policy_name": "Helix Platform - Rollback Authority",
  "rules": [
   "1. The deploying engineer may roll back their own change within 30 minutes of deployment.",
   "2. After 30 minutes, the owning team's on-call decides.",
   "3. During a declared incident the incident commander decides, overriding rules 1 and 2.",
   "4. A change that altered the database schema may be rolled back only with the data engineering on-call's agreement.",
   "5. A change to a shared library is rolled back by the platform on-call.",
   "6. A rollback during a change freeze follows the freeze policy's exemption route.",
   "7. If the deploying engineer is unreachable, authority passes to their team's on-call.",
   "8. A security patch is never rolled back without the security on-call's agreement.",
   "9. Where two rules name different owners, both must agree.",
   "10. The decision and its owner are recorded in the incident channel."],
  "case": {"change": "c-8821", "deployed_at": "2026-07-14T09:05Z", "now": "2026-07-14T09:20Z",
           "deploying_engineer": "reachable", "incident_declared": True,
           "incident_commander": "on duty", "schema_change": False, "shared_library": False,
           "security_patch": False, "change_freeze_active": False}},
 "field": {"id": "decision_owner", "type": "choice",
           "question": "Who owns the decision to roll back change c-8821?",
           "options": [{"value": "the deploying engineer"}, {"value": "the owning team's on-call"},
                       {"value": "the incident commander"}, {"value": "the data engineering on-call"},
                       {"value": "the platform on-call"}, {"value": "the security on-call"}]},
 "target": "the incident commander", "abstention_cause": None,
 "rationale": "An incident is declared, so rule 3 overrides the 15-minute window in rule 1, and no schema, shared-library, freeze or security rule brings a second owner into rule 9."},
{
 "key": "lp23", "domain": "travel",
 "state": {
  "policy_name": "Northlight Airways - Delay Compensation Scheme",
  "rules": [
   "1. Compensation is due when a flight reaches its destination 3 hours or more later than scheduled.",
   "2. For flights of 1,500 km or less the amount is 250 EUR.",
   "3. For flights over 1,500 km wholly within the region the amount is 400 EUR.",
   "4. For other flights over 1,500 km and up to 3,500 km the amount is 400 EUR.",
   "5. For other flights over 3,500 km the amount is 600 EUR.",
   "6. Where rule 5 applies and the delay is between 3 and 4 hours, the amount is halved.",
   "7. No compensation is due where the delay was caused by extraordinary circumstances outside the carrier's control.",
   "8. A strike by the carrier's own staff is not an extraordinary circumstance.",
   "9. Compensation is in addition to any refund or rerouting.",
   "10. A claim must be made within 2 years of the flight."],
  "case": {"flight": "NL118", "route": "regional hub to an airport outside the region",
           "distance_km": 5200, "scheduled_arrival": "2026-03-12T18:05",
           "actual_arrival": "2026-03-12T21:30", "delay": "3 hours 25 minutes",
           "cause": "strike by Northlight's own cabin crew", "claim_date": "2026-04-02"}},
 "field": {"id": "amount", "type": "choice",
           "question": "How much compensation is due to this passenger?",
           "options": [{"value": "no compensation"}, {"value": "125 EUR"}, {"value": "250 EUR"},
                       {"value": "300 EUR"}, {"value": "400 EUR"}, {"value": "600 EUR"}]},
 "target": "300 EUR", "abstention_cause": None,
 "rationale": "The 5,200 km flight outside the region sits in rule 5 at 600 EUR, and the 3 hour 25 minute delay halves it under rule 6, with rule 8 keeping the own-staff strike inside the scheme."},
{
 "key": "lp24", "domain": "education",
 "state": {
  "policy_name": "Delta Institute - Enrolment Status Rules",
  "rules": [
   "1. A student taking 12 or more credit hours in a term is full-time.",
   "2. A student taking 6 to 11 credit hours is half-time.",
   "3. A student taking fewer than 6 credit hours is less-than-half-time.",
   "4. Audited courses do not count toward credit hours.",
   "5. A thesis-stage graduate student registered for the thesis course is full-time whatever the hours.",
   "6. A course dropped before the add/drop deadline does not count.",
   "7. A course withdrawn after the deadline counts toward status for that term.",
   "8. An internship credit counts at half its stated hours.",
   "9. Status is fixed at the add/drop deadline, except where rule 7 applies.",
   "10. The student health plan requires at least half-time status."],
  "case": {"student": "D-5512", "level": "undergraduate", "term": "Fall 2026",
           "registrations": [{"course": "HIST210", "hours": 3, "status": "enrolled"},
                             {"course": "BIO150", "hours": 4, "status": "audit"},
                             {"course": "INTERN300", "stated_hours": 6, "status": "enrolled", "type": "internship"},
                             {"course": "MATH120", "hours": 3, "status": "dropped before the add/drop deadline"},
                             {"course": "ENG101", "hours": 3, "status": "withdrawn after the deadline"}]}},
 "field": {"id": "enrolment_status", "type": "choice",
           "question": "What is this student's enrolment status for Fall 2026?",
           "options": [{"value": "full-time"}, {"value": "half-time"},
                       {"value": "less-than-half-time"}, {"value": "not enrolled"}]},
 "target": "half-time", "abstention_cause": None,
 "rationale": "HIST210 gives 3 hours, the internship counts at half its 6 stated hours for 3 more and the withdrawn ENG101 still counts for 3, while the audit and the dropped course count nothing, making 9 hours."},
{
 "key": "lp25", "domain": "retail",
 "state": {
  "policy_name": "Bramble Home - Trade Account Discount Schedule",
  "rules": [
   "1. Trade accounts are graded Bronze, Silver or Gold by rolling 12-month spend.",
   "2. Bronze is spend under 20,000 GBP, Silver is 20,000 to 99,999 GBP, Gold is 100,000 GBP or more.",
   "3. On furniture the discount is 10 percent Bronze, 15 percent Silver, 20 percent Gold.",
   "4. On lighting the discount is 5 percent Bronze, 10 percent Silver, 15 percent Gold.",
   "5. On textiles the discount is 5 percent at every grade.",
   "6. Clearance lines carry no trade discount at any grade.",
   "7. Discounts apply to the goods value only and never to delivery.",
   "8. Two discounts are never combined on one line.",
   "9. Categories that are not furniture, lighting or textiles are priced line by line by the trade desk, whose decision is outside this schedule and is not summarised anywhere in it.",
   "10. Nothing in this schedule sets a rate for a category covered by rule 9."],
  "case": {"account": "TR-2098", "grade": "Silver", "rolling_12m_spend_gbp": 41000,
           "order": "BH-55301",
           "line_in_question": {"description": "installation labour, 2 fitters for one day",
                                "category": "installation labour", "goods_value_gbp": 0,
                                "labour_value_gbp": 480, "clearance": False}}},
 "field": {"id": "labour_discount", "type": "choice",
           "question": "What trade discount applies to the installation labour line on order BH-55301?",
           "options": [{"value": "0 percent"}, {"value": "5 percent"}, {"value": "10 percent"},
                       {"value": "15 percent"}, {"value": "20 percent"}]},
 "target": None, "abstention_cause": "insufficient_evidence",
 "rationale": "Installation labour is none of the three scheduled categories, and rules 9 and 10 hand it to the trade desk without stating any rate, so the schedule does not determine the answer."},
{
 "key": "lp26", "domain": "hr",
 "state": {
  "policy_name": "Tamarind Group - Notice Periods",
  "rules": [
   "1. During probation, the employee gives and receives 1 week's notice.",
   "2. Probation lasts 6 months unless extended in writing.",
   "3. After probation, an employee gives 1 month's notice.",
   "4. An employee at grade 5 or above gives 3 months' notice.",
   "5. The company gives 1 week per completed year of service, with a minimum of 1 month and a maximum of 12 weeks.",
   "6. Where the contract and the statutory minimum differ, the longer period applies.",
   "7. The statutory minimum an employee must give is 1 week.",
   "8. Notice runs from the day after it is given.",
   "9. Garden leave may be imposed for all or part of a notice period.",
   "10. Summary dismissal for gross misconduct carries no notice."],
  "case": {"employee": "T-4410", "grade": 4, "start_date": "2020-03-02",
           "probation": "completed 2020-09-02", "event": "employee resigns",
           "resignation_date": "2026-09-21", "gross_misconduct": False}},
 "field": {"id": "notice_period", "type": "choice",
           "question": "What notice period must this employee serve?",
           "options": [{"value": "1 week"}, {"value": "1 month"}, {"value": "6 weeks"},
                       {"value": "3 months"}, {"value": "12 weeks"}]},
 "target": "1 month", "abstention_cause": None,
 "rationale": "The employee is resigning, so rule 3 sets 1 month for a grade 4 past probation; rule 5 governs only what the company gives and the 1-week statutory floor in rule 7 is shorter."},
{
 "key": "lp27", "domain": "finance_ops",
 "state": {
  "policy_name": "Arden Bank - Customer Risk Rating Rulebook",
  "rules": [
   "1. Every customer starts at rating 1.",
   "2. A customer resident in a jurisdiction on the monitored list is raised one rating.",
   "3. A customer resident in a jurisdiction on the high-risk list is raised two ratings.",
   "4. A cash-intensive business is raised one rating.",
   "5. A politically exposed person is rated 4 whatever the other factors.",
   "6. A customer listed on a recognised stock exchange is lowered one rating, but never below 1.",
   "7. Ratings are capped at 4.",
   "8. Ratings 3 and 4 are reviewed annually; ratings 1 and 2 every three years.",
   "9. An adverse media finding raises the rating one, once only.",
   "10. The rating is recorded with the factors that produced it."],
  "case": {"customer": "AB-7741", "type": "company", "residence_jurisdiction": "on the monitored list",
           "cash_intensive": True, "politically_exposed_person": False,
           "listed_on_recognised_exchange": True, "adverse_media": False}},
 "field": {"id": "risk_rating", "type": "ordinal",
           "question": "What customer risk rating does the rulebook produce for AB-7741?",
           "levels": [{"value": 1, "description": "Standard risk."},
                      {"value": 2, "description": "Elevated risk."},
                      {"value": 3, "description": "High risk."},
                      {"value": 4, "description": "Highest risk."}]},
 "target": 2, "abstention_cause": None,
 "rationale": "Starting at 1, the monitored jurisdiction adds one and the cash-intensive business adds one, and the recognised-exchange listing takes one back off, leaving 2."},
{
 "key": "lp28", "domain": "software_incidents",
 "state": "Fenwick SaaS - Severity Definitions\n"
          "1. Sev1 is a complete loss of service for any paid customer, or confirmed data loss.\n"
          "2. Sev2 is a partial loss of service, or an error rate above 5 percent sustained for 15 minutes.\n"
          "3. Sev3 is a defect with a workaround, or an internal-only failure.\n"
          "4. A maintenance window declared 5 days ahead is not an incident.\n"
          "5. Any suspected security breach is Sev1.\n"
          "6. A failure confined to the staging environment is never above Sev3.\n"
          "7. Severity is set by the first responder and may be raised by the incident commander.\n"
          "8. A failure of the status page itself is Sev2.\n"
          "9. Repeated Sev3s on the same component within 7 days are raised to Sev2.\n"
          "10. Severity is recorded at declaration and again at close.\n\n"
          "Case: At 14:02 the staging environment lost its database entirely. No production customer was affected and "
          "the status page is healthy. This is the third failure of the staging database in five days. There is no "
          "sign of a security breach and no maintenance window had been declared.",
 "field": {"id": "severity", "type": "ordinal",
           "question": "What severity should the first responder declare at 14:02?",
           "levels": [{"value": 0, "description": "Not an incident."},
                      {"value": 1, "description": "Sev3."},
                      {"value": 2, "description": "Sev2."},
                      {"value": 3, "description": "Sev1."}]},
 "target": 1, "abstention_cause": None,
 "rationale": "The failure is internal-only, so it starts at Sev3, and although rule 9 would raise a repeat to Sev2, rule 6 caps anything confined to staging at Sev3."},
{
 "key": "lp29", "domain": "healthcare_admin",
 "state": {
  "policy_name": "Cedar Valley - Referral Triage Standard",
  "rules": [
   "1. A suspected cancer referral is band 4, to be seen within 2 weeks.",
   "2. A referral carrying a red-flag symptom from Appendix 1 is band 4.",
   "3. A post-operative complication within 30 days of surgery is band 3.",
   "4. A referral showing a deteriorating measurement trend across two readings is band 3.",
   "5. A routine referral is band 1.",
   "6. A referral that has waited more than 18 weeks is raised one band, once only.",
   "7. A patient under 16 is raised one band above the referring band.",
   "8. Band 4 is the highest; a raise above it stays at band 4.",
   "9. A referral returned to the GP for missing information restarts the waiting clock.",
   "10. Bands are reviewed weekly by the triage nurse."],
  "case": {"referral": "CV-2210", "patient_age": 41, "suspected_cancer": False, "red_flags": [],
           "post_operative_within_30_days": False, "measurement_trend": "stable across two readings",
           "weeks_waiting": 21, "returned_to_gp": False}},
 "field": {"id": "triage_band", "type": "ordinal",
           "question": "Which triage band should this referral hold today?",
           "levels": [{"value": 1, "description": "Band 1, routine."},
                      {"value": 2, "description": "Band 2, raised routine."},
                      {"value": 3, "description": "Band 3, urgent."},
                      {"value": 4, "description": "Band 4, seen within two weeks."}]},
 "target": 2, "abstention_cause": None,
 "rationale": "Nothing in the case triggers bands 3 or 4, so it is routine at band 1, and the 21-week wait raises it once under rule 6 to band 2."},
{
 "key": "lp30", "domain": "education",
 "state": "Holm University - Academic Misconduct Outcomes\n"
          "1. Poor academic practice with no intent is outcome 1: a formal caution and required training.\n"
          "2. A first offence of plagiarism is outcome 2: the affected component is marked zero.\n"
          "3. A second offence, or a first offence involving contract cheating, is outcome 3: the module is failed with no reassessment.\n"
          "4. A third offence, or contract cheating in a final-year dissertation, is outcome 4: referral to the disciplinary board with possible exclusion.\n"
          "5. Collusion between students is treated as plagiarism for each student involved.\n"
          "6. An offence more than 3 years old is not counted when deciding whether an offence is a first or a second.\n"
          "7. A student who admits the offence at the first opportunity may have the outcome reduced by one tier, but never below 1.\n"
          "8. Outcomes 3 and 4 are recorded on the transcript.\n"
          "9. A caution under outcome 1 is not an offence for the purposes of rules 3 and 4.\n"
          "10. The panel records its reasoning for the tier chosen.\n\n"
          "Case: Rina received a formal caution for poor academic practice in 2023. In 2026 she submitted a "
          "second-year essay containing substantial unattributed copying from a website, and she admitted it as soon "
          "as she was asked. She has no other record and has never used a contract-cheating service.",
 "field": {"id": "outcome_tier", "type": "ordinal",
           "question": "Which outcome tier should the panel apply to Rina?",
           "levels": [{"value": 1, "description": "Outcome 1: formal caution and required training."},
                      {"value": 2, "description": "Outcome 2: the affected component is marked zero."},
                      {"value": 3, "description": "Outcome 3: the module is failed with no reassessment."},
                      {"value": 4, "description": "Outcome 4: referral to the disciplinary board."}]},
 "target": 1, "abstention_cause": None,
 "rationale": "Rule 9 keeps the 2023 caution from counting as an offence, so this is a first plagiarism offence at outcome 2, and the immediate admission reduces it one tier under rule 7."},
]
