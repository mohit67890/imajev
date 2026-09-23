"""trap: the state carries a salient distractor fact that must be ignored."""

ITEMS = [
{
 "key": "tr01", "domain": "retail",
 "state": {"policy": "Unopened homeware may be returned within 28 days with proof of purchase. Nothing else "
                     "affects the right to return; a customer's history is not a ground for refusing one.",
           "return": {"item": "ceramic casserole dish", "opened": False, "days_since_purchase": 11,
                      "proof_of_purchase": "card receipt"},
           "customer_notes": "This customer has complained about delivery twice this year and left a one-star "
                             "review in March. A store manager has flagged the account as 'difficult'.",
           "stock_position": "the line is discontinued and cannot be resold at full price"},
 "field": {"id": "accept_return", "type": "boolean",
           "question": "Should the store accept this return?"},
 "target": True, "abstention_cause": None,
 "rationale": "The dish is unopened, 11 days old and has a receipt, and the policy says a customer's history is not a ground for refusal, so the complaint record and the resale problem are beside the point."},
{
 "key": "tr02", "domain": "hr",
 "state": "The policy is that any employee with more than 26 weeks' continuous service may make one flexible "
          "working request in a 12-month period, and the request must be considered on its merits. Ruth has 3 "
          "years' service and has made no request in the last 12 months. Her line manager has written in the case "
          "notes that he finds flexible working 'disruptive to team culture' and that he would rather not receive "
          "requests at all. Nothing in the policy makes the manager's preference a ground for refusing to consider "
          "a request.",
 "field": {"id": "request_valid", "type": "boolean",
           "question": "Is Ruth entitled to have this flexible working request considered?"},
 "target": True, "abstention_cause": None,
 "rationale": "She clears the 26-week service test and has made no request in the period, and the manager's stated dislike of flexible working is not a condition in the policy."},
{
 "key": "tr03", "domain": "logistics",
 "state": {"vehicle": {"plate": "KP18 TDR", "gross_weight_limit_kg": 7500, "unladen_kg": 3100},
           "load": {"pallets": 6, "total_load_kg": 3980, "declared_value_gbp": 184000,
                    "insurance_band": "high value, band 4"},
           "rule": "A vehicle may travel if the unladen weight plus the load does not exceed the gross weight "
                   "limit. Declared value affects the insurance band and nothing else.",
           "note": "The transport clerk has queried the load because of its unusually high declared value."},
 "field": {"id": "may_travel", "type": "boolean",
           "question": "Is this vehicle within its gross weight limit?"},
 "target": True, "abstention_cause": None,
 "rationale": "3,100 kg unladen plus a 3,980 kg load is 7,080 kg, under the 7,500 kg limit, and the 184,000 GBP declared value only sets the insurance band."},
{
 "key": "tr04", "domain": "healthcare_admin",
 "state": "The screening programme invites people aged 50 to 74 on their registered date of birth, and the age "
          "criterion is the only entry criterion; clinical history does not bring anyone into the programme early. "
          "Mr Sandoval is 46. His record lists a family history of the condition, two relevant comorbidities and a "
          "previous consultant letter suggesting he should be 'kept under review'. His GP has asked whether he can "
          "be added to the screening list now.",
 "field": {"id": "invite_now", "type": "boolean",
           "question": "Does Mr Sandoval meet the criteria for a screening invitation now?"},
 "target": False, "abstention_cause": None,
 "rationale": "At 46 he is below the programme's 50-to-74 age range, which the state says is the only entry criterion, so the family history and comorbidities do not qualify him."},
{
 "key": "tr05", "domain": "finance_ops",
 "state": {"request": "PR-9920", "line_requiring_approval": {"description": "annual audit software",
                                                             "amount_eur": 3800},
           "other_lines_on_the_same_page": [{"description": "hardware refresh, approved separately last quarter",
                                             "amount_usd": 61000}],
           "threshold": "A team lead may approve a line up to 5,000 EUR. Approval is tested line by line; lines "
                        "approved in an earlier period are not aggregated with a new request.",
           "note": "The purchasing screen prints the earlier hardware line above the new one."},
 "field": {"id": "team_lead_may_approve", "type": "boolean",
           "question": "May a team lead approve the line in PR-9920 on their own?"},
 "target": True, "abstention_cause": None,
 "rationale": "The line needing approval is 3,800 EUR, under the 5,000 EUR team-lead limit, and the 61,000 USD hardware line was approved in an earlier period and is not aggregated."},
{
 "key": "tr06", "domain": "software_incidents",
 "state": "The paging rule is simple: alerts from the staging environment never page anyone, whatever their "
          "severity label, and are written to the daily digest instead. The alert 'db_replica_lag' has fired on "
          "staging. It carries the label severity=critical, it fired on production twice last month and caused a "
          "genuine incident on both occasions, and the team has a reminder in the channel topic to 'treat replica "
          "lag seriously'.",
 "field": {"id": "should_page", "type": "boolean",
           "question": "Should this alert page the on-call engineer?"},
 "target": False, "abstention_cause": None,
 "rationale": "The alert is from staging and the rule excludes staging from paging whatever the severity label, so its production history last month does not change the routing."},
{
 "key": "tr07", "domain": "travel",
 "state": {"passenger": {"name": "D. Ferrante", "frequent_flyer_number": "FF-7712094",
                         "programme_tier": "none", "lifetime_miles": 412000},
           "fare": {"code": "L", "cabin": "economy", "checked_bag_included": False},
           "rule": "A free checked bag is included for Silver tier and above, and for Flex and Business fares. "
                   "Lifetime miles do not confer tier; tier is earned each year.",
           "note": "The passenger points to the large lifetime mileage figure on the booking screen."},
 "field": {"id": "free_checked_bag", "type": "boolean",
           "question": "Does this passenger get a free checked bag?"},
 "target": False, "abstention_cause": None,
 "rationale": "The fare is a basic economy code with no bag and the programme tier is none, and the rule says lifetime miles do not confer tier."},
{
 "key": "tr08", "domain": "education",
 "state": "Entry to the MSc conversion course requires a pass in the qualifying mathematics paper, and the "
          "admissions rule states that no other module may be substituted for it. The applicant's transcript shows "
          "a distinction in advanced statistics, a first-class dissertation, a prize for the best final-year "
          "project, and a fail in the qualifying mathematics paper with no resit recorded.",
 "field": {"id": "meets_entry", "type": "boolean",
           "question": "Does this applicant meet the entry requirement for the conversion course?"},
 "target": False, "abstention_cause": None,
 "rationale": "The qualifying mathematics paper is failed with no resit and the rule forbids substituting any other module, so the distinction and the prize do not satisfy it."},
{
 "key": "tr09", "domain": "retail",
 "state": {"query": "shelf life remaining on batch LOT-88120",
           "batch_records": [{"lot": "LOT-88119", "product": "olive tapenade 180g", "best_before": "2027-01-14"},
                             {"lot": "LOT-88121", "product": "olive tapenade 180g", "best_before": "2027-01-16"}],
           "note": "Best-before dates are set per batch at production and are not interpolated between "
                   "neighbouring lot numbers; consecutive lots can be produced weeks apart.",
           "supplier_statement": "Our olive tapenade generally carries a 12-month shelf life."},
 "field": {"id": "in_date_on_2027_01_10", "type": "boolean",
           "question": "Will batch LOT-88120 still be within its best-before date on 10 January 2027?"},
 "target": None, "abstention_cause": "insufficient_evidence",
 "rationale": "No record is held for LOT-88120 itself, and the state says dates are not interpolated between neighbouring lots, so the two flanking batches and the general shelf-life statement settle nothing."},
{
 "key": "tr10", "domain": "hr",
 "state": {"employee": {"id": "E-2214", "grade": 6, "service_years": 19, "age": 58},
           "rule": "The notice an employee must give is set by grade alone: grades 1 to 4 give one month and "
                   "grades 5 and above give three months. Length of service sets what the company must give, not "
                   "what the employee must give.",
           "event": "the employee has resigned",
           "note": "The leaver's long service is displayed prominently on the exit screen."},
 "field": {"id": "three_months", "type": "boolean",
           "question": "Must this employee give three months' notice?"},
 "target": True, "abstention_cause": None,
 "rationale": "Grade 6 is at or above grade 5, which the rule sets at three months, and the 19 years of service govern only the company's side."},
{
 "key": "tr11", "domain": "finance_ops",
 "state": "Invoice INV-6610 for 4,200 GBP fell due on 12 May and was settled on 3 June. The account also carries a "
          "credit note for 4,200 GBP issued on 9 May, but that credit note was raised against a different invoice, "
          "INV-6602, and was applied to it in full on the day it was issued. The late-payment rule looks only at "
          "the settlement date of the invoice in question against its own due date.",
 "field": {"id": "paid_late", "type": "boolean",
           "question": "Was invoice INV-6610 paid late?"},
 "target": True, "abstention_cause": None,
 "rationale": "INV-6610 was due on 12 May and settled on 3 June, and the matching-value credit note belongs to a different invoice that it was already applied to."},
{
 "key": "tr12", "domain": "logistics",
 "state": {"container": "RFCU-2210", "cargo": "chilled ready meals",
           "set_point_c": 3, "permitted_range_c": [1, 5],
           "logged_temperatures_c": [2.8, 3.1, 3.0, 2.9, 3.2, 3.0],
           "ambient_temperature_c": 34, "door_open_events": 0,
           "rule": "Compliance is judged on the logged cargo temperatures against the permitted range. Ambient "
                   "temperature outside the unit is recorded for maintenance planning only."},
 "field": {"id": "temperature_compliant", "type": "boolean",
           "question": "Was this container temperature compliant for the journey?"},
 "target": True, "abstention_cause": None,
 "rationale": "Every logged cargo temperature sits between 2.8 and 3.2 degrees, inside the 1-to-5 range, and the 34-degree ambient reading is recorded for maintenance only."},
{
 "key": "tr13", "domain": "software_incidents",
 "state": {"failing_service": "notification-dispatcher",
           "service_catalogue": [{"service": "notification-dispatcher", "owning_team": "Messaging"},
                                 {"service": "user-profile", "owning_team": "Identity"},
                                 {"service": "audit-log", "owning_team": "Platform"}],
           "recent_deploys": [{"service": "notification-dispatcher", "deployed_by_team": "Platform",
                               "reason": "shared library bump on behalf of Messaging"}],
           "rule": "Ownership follows the service catalogue. The team that carried out a deploy does not become "
                   "the owner of the service."},
 "field": {"id": "owning_team", "type": "choice",
           "question": "Which team owns the failing service?",
           "options": [{"value": "Messaging"}, {"value": "Identity"}, {"value": "Platform"},
                       {"value": "no team owns it"}]},
 "target": "Messaging", "abstention_cause": None,
 "rationale": "The catalogue assigns notification-dispatcher to Messaging, and the rule says the Platform team's deploy on their behalf does not transfer ownership."},
{
 "key": "tr14", "domain": "healthcare_admin",
 "state": {"referral": {"id": "R-8812", "speciality_required": "cardiology"},
           "patient": {"registered_gp_practice": "Elmfield Surgery, which sits next door to the dermatology unit",
                       "preferred_site": "the dermatology unit, because parking is easier"},
           "clinics": [{"speciality": "cardiology", "site": "Westgate Hospital"},
                       {"speciality": "dermatology", "site": "Elmfield Annexe"},
                       {"speciality": "respiratory", "site": "Westgate Hospital"}],
           "rule": "A referral is booked to a clinic of the required speciality. Proximity to the patient's GP "
                   "practice and patient site preference do not change the speciality required."},
 "field": {"id": "booking_site", "type": "choice",
           "question": "Where should referral R-8812 be booked?",
           "options": [{"value": "cardiology at Westgate Hospital"},
                       {"value": "dermatology at Elmfield Annexe"},
                       {"value": "respiratory at Westgate Hospital"},
                       {"value": "no clinic is suitable"}]},
 "target": "cardiology at Westgate Hospital", "abstention_cause": None,
 "rationale": "The referral requires cardiology, which runs only at Westgate, and convenience of the Elmfield site is expressly not a factor."},
{
 "key": "tr15", "domain": "travel",
 "state": "Compensation under this scheme depends only on flight distance and the length of the delay. The flight "
          "covered 980 km and arrived 4 hours 10 minutes late for reasons within the carrier's control. The bands "
          "are 250 EUR for flights of 1,500 km or less and 400 EUR for longer flights, with no halving at this "
          "distance. The passenger paid 41 EUR for the ticket in a sale and has written at length about how cheap "
          "the fare was.",
 "field": {"id": "compensation", "type": "choice",
           "question": "How much compensation is due?",
           "options": [{"value": "41 EUR"}, {"value": "125 EUR"}, {"value": "250 EUR"},
                       {"value": "400 EUR"}, {"value": "no compensation"}]},
 "target": "250 EUR", "abstention_cause": None,
 "rationale": "The flight is under 1,500 km and more than three hours late, which is the 250 EUR band, and the scheme takes no account of the fare paid."},
{
 "key": "tr16", "domain": "education",
 "state": {"student": "S-4419", "final_weighted_average": 68.4,
           "classification_bands": ["first: 70 and above", "upper second: 60 to 69",
                                    "lower second: 50 to 59", "third: 40 to 49"],
           "attendance": "99 percent, the highest in the cohort",
           "tutor_comment": "outstanding engagement, clearly first-class material",
           "rule": "Classification is set by the final weighted average alone. Attendance and tutor comments are "
                   "not inputs to it, and the board has exercised no discretion in this case."},
 "field": {"id": "classification", "type": "choice",
           "question": "Which classification should this student receive?",
           "options": [{"value": "first"}, {"value": "upper second"},
                       {"value": "lower second"}, {"value": "third"}]},
 "target": "upper second", "abstention_cause": None,
 "rationale": "68.4 falls in the 60-to-69 upper second band, and the rule makes attendance and the tutor's comment irrelevant to classification."},
{
 "key": "tr17", "domain": "retail",
 "state": {"order": "N-3312", "payment_method": "credit card ending 9011, still valid",
           "customer_request": "the customer has asked, twice, for the refund to be paid in cash",
           "policy": "A refund is returned to the original payment method whenever that method can accept it. "
                     "Customer preference does not override this, and cash is only used where the original "
                     "purchase was made in cash.",
           "refund_gbp": 74.00, "purchase_channel": "in store"},
 "field": {"id": "refund_method", "type": "choice",
           "question": "How should this refund be issued?",
           "options": [{"value": "cash"}, {"value": "to the credit card ending 9011"},
                       {"value": "as store credit"}, {"value": "by bank transfer"}]},
 "target": "to the credit card ending 9011", "abstention_cause": None,
 "rationale": "The original card is still valid and the policy sends the refund back to it, with customer preference expressly not overriding that."},
{
 "key": "tr18", "domain": "finance_ops",
 "state": "Approval authority is set by the value of the request: up to 10,000 EUR a manager approves, above that a "
          "director approves. The request in front of the system is for 24,000 EUR. It was raised by the finance "
          "director herself, who noted in the comments that she had 'already reviewed and is happy with it'. The "
          "policy adds that the person who raises a request may not also approve it, whatever their level.",
 "field": {"id": "approver", "type": "choice",
           "question": "Who may approve this 24,000 EUR request?",
           "options": [{"value": "the manager"}, {"value": "the finance director who raised it"},
                       {"value": "another director"}, {"value": "nobody; the request must be cancelled"}]},
 "target": "another director", "abstention_cause": None,
 "rationale": "At 24,000 EUR the request needs a director, and the separation rule blocks the director who raised it, leaving another director."},
{
 "key": "tr19", "domain": "logistics",
 "state": "The contract with this customer fixes a 48-hour service level for all standard consignments, and says "
          "service levels are varied only by a written contract amendment. The customer's warehouse manager has "
          "sent three emails this morning describing the consignment as 'absolutely urgent' and asking for same-day "
          "delivery. No amendment has been signed, and the consignment is classified as standard on the booking.",
 "field": {"id": "service_level", "type": "choice",
           "question": "Which service level applies to this consignment?",
           "options": [{"value": "same day"}, {"value": "next day"}, {"value": "48 hours"},
                       {"value": "72 hours"}]},
 "target": "48 hours", "abstention_cause": None,
 "rationale": "The contract fixes 48 hours for standard consignments and permits variation only by written amendment, which has not been signed."},
{
 "key": "tr20", "domain": "hr",
 "state": {"new_joiner": "N-1180", "job_code": "OPS-4",
           "previous_employer_salary_gbp": 71000,
           "salary_expectation_gbp": 74000,
           "pay_bands": [{"job_code": "OPS-3", "band": "B4", "range_gbp": "42,000 to 52,000"},
                         {"job_code": "OPS-4", "band": "B5", "range_gbp": "50,000 to 62,000"},
                         {"job_code": "OPS-5", "band": "B6", "range_gbp": "60,000 to 75,000"}],
           "rule": "A joiner is placed in the band for their job code. Previous salary and salary expectation are "
                   "not inputs to banding."},
 "field": {"id": "pay_band", "type": "choice",
           "question": "Which pay band should this joiner be placed in?",
           "options": [{"value": "B4"}, {"value": "B5"}, {"value": "B6"},
                       {"value": "no band, the offer must be re-scoped"}]},
 "target": "B5", "abstention_cause": None,
 "rationale": "Job code OPS-4 maps to band B5, and the rule excludes previous salary and expectation from banding even though both point at B6."},
{
 "key": "tr21", "domain": "healthcare_admin",
 "state": "The triage standard sets priority from the clinical criteria in the referral alone: red-flag symptoms "
          "give priority 1, a deteriorating trend gives priority 2, and everything else is priority 3. The "
          "referral for this patient records no red-flag symptom and a stable trend across three readings. The "
          "referral letter also says the patient is 'extremely anxious about the wait and has telephoned the "
          "practice four times', and the referring GP has written 'please expedite'.",
 "field": {"id": "priority", "type": "choice",
           "question": "Which triage priority should this referral be given?",
           "options": [{"value": "priority 1"}, {"value": "priority 2"}, {"value": "priority 3"},
                       {"value": "return the referral to the GP"}]},
 "target": "priority 3", "abstention_cause": None,
 "rationale": "There is no red flag and the trend is stable, so the standard gives priority 3, and anxiety or a request to expedite are not clinical criteria in it."},
{
 "key": "tr22", "domain": "software_incidents",
 "state": "The rollback policy says that during a declared incident the decision belongs to the incident commander, "
          "and to nobody else, whatever their seniority. An incident is declared and Priya is the incident "
          "commander. In the incident channel, a vice president has written 'roll it back now' and the engineer who "
          "wrote the change has written 'I want to roll back'. The change is not a security patch and did not "
          "alter the database schema.",
 "field": {"id": "decision_owner", "type": "choice",
           "question": "Who owns the rollback decision here?",
           "options": [{"value": "the vice president"}, {"value": "the engineer who wrote the change"},
                       {"value": "Priya, the incident commander"}, {"value": "the security on-call"}]},
 "target": "Priya, the incident commander", "abstention_cause": None,
 "rationale": "The policy gives the decision to the incident commander during a declared incident whatever anyone's seniority, and no schema or security rule brings in another owner."},
{
 "key": "tr23", "domain": "travel",
 "state": {"booking": "FM-2204", "party": [{"name": "R. Adeyinka", "role": "lead passenger",
                                            "passport_expiry": None, "documents_scanned": False},
                                           {"name": "J. Adeyinka", "role": "travelling companion",
                                            "passport_expiry": "2031-04-18", "documents_scanned": True}],
           "destination_rule": "Entry requires a passport valid for at least six months beyond the date of "
                               "arrival. The rule is tested per traveller and no traveller's document stands in "
                               "for another's.",
           "arrival_date": "2026-12-02"},
 "field": {"id": "lead_passenger_ok", "type": "boolean",
           "question": "Does the lead passenger's passport meet the destination's validity rule?"},
 "target": None, "abstention_cause": "insufficient_evidence",
 "rationale": "No expiry date has been captured for the lead passenger and the rule is tested per traveller, so the companion's 2031 passport says nothing about hers."},
{
 "key": "tr24", "domain": "education",
 "state": "Fee status is decided by ordinary residence for the three years before the course starts, and the "
          "regulations state that nationality alone neither confers nor removes home fee status. The applicant "
          "holds citizenship of the country where the university sits, but has lived and worked abroad "
          "continuously for the last six years and has no residence in the country during the three-year period. "
          "She has attached her passport as evidence of her entitlement to home fees.",
 "field": {"id": "fee_status", "type": "choice",
           "question": "Which fee status should the applicant be assessed at?",
           "options": [{"value": "home"}, {"value": "overseas"},
                       {"value": "home, with a residence waiver"}, {"value": "fee status is decided by the funding body"}]},
 "target": "overseas", "abstention_cause": None,
 "rationale": "She has no ordinary residence in the three-year period, and the regulations say nationality alone does not confer home status, so the passport is beside the point."},
{
 "key": "tr25", "domain": "retail",
 "state": {"basket_total_gbp": 96.00, "date": "2026-11-04",
           "promotions": [{"code": "AUTUMN25", "discount": "25 percent", "valid_to": "2026-10-31", "status": "expired"},
                          {"code": "NOV10", "discount": "10 percent", "valid_from": "2026-11-01",
                           "valid_to": "2026-11-30", "status": "live"},
                          {"code": "STAFF20", "discount": "20 percent", "eligibility": "staff cards only",
                           "customer_is_staff": False}],
           "rule": "Only a live promotion for which the customer is eligible may be applied, and expired "
                   "promotions are never honoured, however recently they lapsed."},
 "field": {"id": "promotion", "type": "choice",
           "question": "Which promotion should be applied to this basket?",
           "options": [{"value": "AUTUMN25"}, {"value": "NOV10"}, {"value": "STAFF20"},
                       {"value": "no promotion applies"}]},
 "target": "NOV10", "abstention_cause": None,
 "rationale": "AUTUMN25 expired four days earlier and the customer is not staff, so the only live promotion she is eligible for is NOV10, despite the larger headline discounts."},
{
 "key": "tr26", "domain": "finance_ops",
 "state": {"transaction": "FX-5521", "amount_usd": 250000, "booking_date": "2026-08-14",
           "rule": "A foreign currency transaction is translated at the closing rate on the booking date. The "
                   "policy expressly forbids using a rate from any other date, including the prior year end.",
           "rates_available": {"2025-12-31 closing": 0.8912, "2026-08-13 closing": 0.9104,
                               "2026-08-14 closing": None},
           "note": "The 14 August closing rate has not been published to the ledger because the rate feed failed "
                   "that evening and has not been backfilled."},
 "field": {"id": "translated_band", "type": "choice",
           "question": "What is the translated value of FX-5521 in the reporting currency?",
           "options": [{"value": "222,800"}, {"value": "227,600"}, {"value": "228,900"},
                       {"value": "231,000"}]},
 "target": None, "abstention_cause": "insufficient_evidence",
 "rationale": "The policy requires the 14 August closing rate, which has not been published, and forbids substituting the 13 August or prior year-end rates that are available."},
{
 "key": "tr27", "domain": "logistics",
 "state": {"service": "TR-4410", "scheduled_arrival": "2026-01-22T11:00",
           "actual_arrival": "2026-01-22T11:20",
           "bands": ["1: under 30 minutes late", "2: 30 minutes to under 2 hours",
                     "3: 2 hours to under 4 hours", "4: 4 hours or more"],
           "weather": "an amber warning for snow was in force across the region all day",
           "rule": "The delay band records the delay actually observed. Weather affects whether a penalty is "
                   "charged, not which band is recorded."},
 "field": {"id": "delay_band", "type": "ordinal",
           "question": "Which delay band should be recorded for TR-4410?",
           "levels": [{"value": 1, "description": "Band 1, under 30 minutes late."},
                      {"value": 2, "description": "Band 2, 30 minutes to under 2 hours late."},
                      {"value": 3, "description": "Band 3, 2 hours to under 4 hours late."},
                      {"value": 4, "description": "Band 4, 4 hours or more late."}]},
 "target": 1, "abstention_cause": None,
 "rationale": "The service arrived 20 minutes late, which is band 1, and the snow warning bears on penalties rather than on the band recorded."},
{
 "key": "tr28", "domain": "software_incidents",
 "state": "Severity on this platform is defined entirely by customer impact: level 1 for internal-only effects, "
          "level 2 for a single customer, level 3 for a subset and level 4 for all customers. A build server is "
          "failing, which stops engineers from shipping but does not touch any customer traffic. The chief "
          "technology officer has joined the channel and asked for hourly updates, and two directors have described "
          "the outage as 'critical for the release'. Customer traffic is unaffected.",
 "field": {"id": "severity", "type": "ordinal",
           "question": "What severity should this incident be recorded at?",
           "levels": [{"value": 1, "description": "Level 1, internal effects only."},
                      {"value": 2, "description": "Level 2, a single customer affected."},
                      {"value": 3, "description": "Level 3, a subset of customers affected."},
                      {"value": 4, "description": "Level 4, all customers affected."}]},
 "target": 1, "abstention_cause": None,
 "rationale": "The failure is confined to the build server with no customer traffic affected, which is level 1, and executive attention is not part of the definition."},
{
 "key": "tr29", "domain": "healthcare_admin",
 "state": {"referral": "CV-7781", "clinical_findings": {"red_flags": [], "post_operative": False,
                                                        "trend": "stable", "suspected_cancer": False},
           "weeks_waiting": 4, "patient_age": 52,
           "referrer_note": "GP has ticked the 'urgent, please see within two weeks' box on the form.",
           "rule": "Bands are set from the clinical criteria and the waiting time recorded in this standard. A "
                   "referrer's urgency tick does not by itself set a band."},
 "field": {"id": "triage_band", "type": "ordinal",
           "question": "Which triage band should referral CV-7781 hold?",
           "levels": [{"value": 1, "description": "Band 1, routine."},
                      {"value": 2, "description": "Band 2, raised routine."},
                      {"value": 3, "description": "Band 3, urgent."},
                      {"value": 4, "description": "Band 4, seen within two weeks."}]},
 "target": 1, "abstention_cause": None,
 "rationale": "No clinical criterion is met and four weeks is short of any waiting uplift, so the referral is routine at band 1 whatever the GP has ticked."},
{
 "key": "tr30", "domain": "hr",
 "state": "The mid-year rating is calculated from three inputs only: objectives met, manager assessment and "
          "calibration. This employee met 5 of 5 objectives, the manager assessment is 'exceeds' and calibration "
          "confirmed 'exceeds', which maps to rating 4 on the five-point scale. A customer wrote in during the "
          "period to complain about a telephone call, which was investigated and found to be about a different "
          "employee. The complaint is not one of the three inputs in any case.",
 "field": {"id": "rating", "type": "ordinal",
           "question": "What mid-year rating should this employee receive?",
           "levels": [{"value": 1, "description": "Rating 1, below expectations."},
                      {"value": 2, "description": "Rating 2, partially meets expectations."},
                      {"value": 3, "description": "Rating 3, meets expectations."},
                      {"value": 4, "description": "Rating 4, exceeds expectations."},
                      {"value": 5, "description": "Rating 5, outstanding."}]},
 "target": 4, "abstention_cause": None,
 "rationale": "All three inputs point to 'exceeds', which maps to rating 4, and the complaint was both about another employee and outside the rating inputs."},
]
