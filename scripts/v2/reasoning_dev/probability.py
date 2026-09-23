"""probability: pick the ordinal likelihood level that the stated evidence supports."""

L5 = [{"value": 1, "description": "Very unlikely: below 10 percent."},
      {"value": 2, "description": "Unlikely: 10 to 35 percent."},
      {"value": 3, "description": "About even: above 35 and up to 65 percent."},
      {"value": 4, "description": "Likely: above 65 and up to 90 percent."},
      {"value": 5, "description": "Very likely: above 90 percent."}]
L4 = [{"value": 1, "description": "Remote: below 15 percent."},
      {"value": 2, "description": "Possible: 15 to 50 percent."},
      {"value": 3, "description": "Probable: above 50 and up to 85 percent."},
      {"value": 4, "description": "Near certain: above 85 percent."}]
L3 = [{"value": 1, "description": "Unlikely: below a third."},
      {"value": 2, "description": "About even: a third to two thirds."},
      {"value": 3, "description": "Likely: above two thirds."}]

ITEMS = [
{
 "key": "pb01", "domain": "retail",
 "state": {"supplier": "Halverden Plastics", "question_about": "the next delivery arriving on the promised day",
           "history": {"deliveries_in_last_12_months": 60, "arrived_on_promised_day": 51},
           "conditions": "The supplier's factory, route and carrier are unchanged, no industrial action is "
                         "reported, and the order size is typical for this line.",
           "instruction": "Judge the likelihood from the stated history, treating it as representative."},
 "field": {"id": "on_time_likelihood", "type": "ordinal",
           "question": "How likely is the next delivery to arrive on the promised day?", "levels": L5},
 "target": 4, "abstention_cause": None,
 "rationale": "51 of 60 deliveries is 85 percent, which falls in the 'above 65 and up to 90 percent' level."},
{
 "key": "pb02", "domain": "software_incidents",
 "state": "The alert 'cache_evictions_high' has fired 24 times in the last quarter. On 3 of those occasions it "
          "turned out to be a genuine incident that needed action; on the other 21 it resolved itself within "
          "minutes and no action was taken. Nothing about the service, its traffic pattern or its alert threshold "
          "has changed since, and the alert has just fired again with the same signature as the previous firings.",
 "field": {"id": "genuine_incident", "type": "ordinal",
           "question": "How likely is this firing to be a genuine incident?", "levels": L5},
 "target": 2, "abstention_cause": None,
 "rationale": "3 genuine incidents in 24 firings is 12.5 percent, inside the 10-to-35 percent level."},
{
 "key": "pb03", "domain": "healthcare_admin",
 "state": {"clinic": "pre-operative assessment", "question_about": "the patient attending their booked slot",
           "history": {"slots_booked_for_this_cohort": 20, "attended": 18},
           "notes": "The cohort is patients who have confirmed by text in the previous 48 hours, which this "
                    "patient has done. Reminder practice and clinic location are unchanged.",
           "instruction": "Read the likelihood off the stated attendance history for this cohort."},
 "field": {"id": "attendance_likelihood", "type": "ordinal",
           "question": "How likely is this patient to attend the booked slot?", "levels": L4},
 "target": 4, "abstention_cause": None,
 "rationale": "18 of 20 confirmed patients attended, which is 90 percent and above the 85 percent boundary for near certain."},
{
 "key": "pb04", "domain": "finance_ops",
 "state": "Of the last 9 quarters, this customer has paid within terms in 5 and has paid late in 4. Nothing about "
          "the customer's business, its finance team or the invoice size has changed, the invoice was delivered on "
          "the usual day, and no dispute has been raised on it. The credit controller is judging the chance that "
          "the current invoice is paid within terms, using the payment history as the only evidence.",
 "field": {"id": "pays_within_terms", "type": "ordinal",
           "question": "How likely is this invoice to be paid within terms?", "levels": L3},
 "target": 2, "abstention_cause": None,
 "rationale": "5 of 9 quarters is about 56 percent, which sits in the 'a third to two thirds' band."},
{
 "key": "pb05", "domain": "logistics",
 "state": {"question_about": "the consignment clearing customs on the first attempt",
           "independent_checks": [{"check": "documentation complete", "historical_pass_rate": 0.9},
                                  {"check": "commodity code accepted", "historical_pass_rate": 0.9}],
           "rule": "Both checks must pass for first-attempt clearance, and the two checks are stated to be "
                   "independent of one another.",
           "consignment": "C-4412, routine electronics, same broker as usual"},
 "field": {"id": "clears_first_time", "type": "ordinal",
           "question": "How likely is this consignment to clear customs on the first attempt?", "levels": L5},
 "target": 4, "abstention_cause": None,
 "rationale": "Two independent checks each passing 90 percent of the time both pass 81 percent of the time, which is in the 65-to-90 percent level."},
{
 "key": "pb06", "domain": "hr",
 "state": "Across the last two years, 200 candidates reached the final interview for this job family and 46 were "
          "offered the role. The candidate in front of the panel has reached the final interview. No further "
          "information about this candidate's performance is available to the person making the estimate, and the "
          "panel composition, role and hiring bar are unchanged from that period.",
 "field": {"id": "offer_likelihood", "type": "ordinal",
           "question": "How likely is this candidate to be offered the role?", "levels": L5},
 "target": 2, "abstention_cause": None,
 "rationale": "46 offers from 200 finalists is 23 percent, inside the 10-to-35 percent level."},
{
 "key": "pb07", "domain": "travel",
 "state": {"question_about": "the passenger making a 55-minute connection",
           "history": {"connections_of_this_length_at_this_airport": 400, "made_the_connection": 372},
           "today": "no weather advisory, inbound flight is showing on time, same terminal transfer",
           "instruction": "Use the stated history for connections of this length at this airport."},
 "field": {"id": "connection_made", "type": "ordinal",
           "question": "How likely is the passenger to make the connection?", "levels": L5},
 "target": 5, "abstention_cause": None,
 "rationale": "372 of 400 is 93 percent, above the 90 percent boundary for very likely."},
{
 "key": "pb08", "domain": "education",
 "state": "In each of the last six years, between 2 and 4 of the 50 students on this module failed at the first "
          "attempt, and the module, its teaching and its assessment are unchanged this year. The student in "
          "question has no attendance concerns, no missed submissions and no flag from the tutor. The question is "
          "the chance that this student fails at the first attempt, judged from the module's history alone.",
 "field": {"id": "fail_likelihood", "type": "ordinal",
           "question": "How likely is this student to fail at the first attempt?", "levels": L5},
 "target": 1, "abstention_cause": None,
 "rationale": "The failure rate runs between 2 and 4 in 50, that is 4 to 8 percent, which is below the 10 percent boundary for very unlikely."},
{
 "key": "pb09", "domain": "finance_ops",
 "state": {"question_about": "a flagged transaction turning out to be fraudulent",
           "last_month": {"transactions_reviewed": 200, "flagged_by_the_model": 40,
                          "flagged_and_fraudulent": 30, "unflagged_and_fraudulent": 2},
           "instruction": "The transaction in front of the analyst has been flagged. Judge the likelihood that it "
                          "is fraudulent using last month's outcomes, which are stated to be representative."},
 "field": {"id": "fraud_likelihood", "type": "ordinal",
           "question": "How likely is this flagged transaction to be fraudulent?", "levels": L4},
 "target": 3, "abstention_cause": None,
 "rationale": "30 of the 40 flagged transactions were fraudulent, which is 75 percent and sits in the 'above 50 and up to 85 percent' level."},
{
 "key": "pb10", "domain": "software_incidents",
 "state": "The team has deployed this kind of schema migration 11 times. On 1 of those occasions it needed a "
          "rollback; the other 10 completed without incident. The migration in front of them is the same shape, "
          "against the same database engine, with the same review process, and the pre-flight checks have all "
          "passed. The question is the chance that this migration completes without needing a rollback.",
 "field": {"id": "completes_cleanly", "type": "ordinal",
           "question": "How likely is this migration to complete without a rollback?", "levels": L5},
 "target": 5, "abstention_cause": None,
 "rationale": "10 clean completions out of 11 is 90.9 percent, just above the 90 percent boundary for very likely."},
{
 "key": "pb11", "domain": "healthcare_admin",
 "state": {"question_about": "a theatre list overrunning past 18:00",
           "history": {"lists_of_this_size_and_mix": 25, "overran_past_18_00": 13},
           "today": "the same surgeon, the same anaesthetist and the same six-case list as the historical sample",
           "instruction": "Judge the chance of an overrun from the stated history."},
 "field": {"id": "overrun_likelihood", "type": "ordinal",
           "question": "How likely is today's list to overrun past 18:00?", "levels": L5},
 "target": 3, "abstention_cause": None,
 "rationale": "13 overruns in 25 comparable lists is 52 percent, which is in the 'above 35 and up to 65 percent' level."},
{
 "key": "pb12", "domain": "retail",
 "state": "Of 500 customers who added this product to a basket and reached the payment page last quarter, 95 "
          "completed the purchase. The checkout flow, the price and the delivery promise are unchanged, and the "
          "customer now on the payment page came through the same channel as that sample. The question is the "
          "chance that this customer completes the purchase.",
 "field": {"id": "completes_purchase", "type": "ordinal",
           "question": "How likely is this customer to complete the purchase?", "levels": L5},
 "target": 2, "abstention_cause": None,
 "rationale": "95 completions from 500 is 19 percent, inside the 10-to-35 percent level."},
{
 "key": "pb13", "domain": "logistics",
 "state": {"question_about": "a pallet being damaged somewhere in the network",
           "history": {"pallets_shipped_last_year": 48000, "damage_claims": 240},
           "packaging": "unchanged", "route": "unchanged", "handling_equipment": "unchanged",
           "instruction": "Judge the chance that the next pallet is damaged, using the stated rate."},
 "field": {"id": "damage_likelihood", "type": "ordinal",
           "question": "How likely is the next pallet to be damaged in the network?", "levels": L5},
 "target": 1, "abstention_cause": None,
 "rationale": "240 claims from 48,000 pallets is 0.5 percent, far below the 10 percent boundary for very unlikely."},
{
 "key": "pb14", "domain": "travel",
 "state": "Historically, 30 percent of passengers on this route buy a seat reservation, and 60 percent of those who "
          "buy a reservation also buy priority boarding, while only 5 percent of those who do not reserve a seat "
          "buy priority boarding. The passenger in front of the agent has already bought a seat reservation. The "
          "agent is judging the chance that this passenger also buys priority boarding, using those figures alone.",
 "field": {"id": "buys_priority", "type": "ordinal",
           "question": "How likely is this passenger to buy priority boarding?", "levels": L3},
 "target": 2, "abstention_cause": None,
 "rationale": "The passenger has already reserved a seat, so the relevant figure is the 60 percent for reservation holders, which sits in the 'a third to two thirds' band."},
{
 "key": "pb15", "domain": "hr",
 "state": {"question_about": "an employee still being with the company in 12 months",
           "cohort": "employees in their second year in this job family",
           "history": {"cohort_size": 120, "still_present_after_a_further_year": 102},
           "notes": "No restructure is planned, pay review timing is unchanged, and this employee has given no "
                    "notice and raised no grievance.",
           "instruction": "Use the cohort retention figure as the estimate."},
 "field": {"id": "retention_likelihood", "type": "ordinal",
           "question": "How likely is this employee to still be with the company in 12 months?", "levels": L4},
 "target": 3, "abstention_cause": None,
 "rationale": "102 of 120 is 85 percent, which is at the top of the 'above 50 and up to 85 percent' level and not above it."},
{
 "key": "pb16", "domain": "education",
 "state": "Of the 80 applicants who were placed on the waiting list for this programme in each of the last three "
          "years, an average of 52 were eventually offered a place as others declined. The intake target, the "
          "number of offers made and the pattern of declines are all unchanged this year, and the applicant in "
          "question is on the waiting list in an ordinary position rather than at either extreme.",
 "field": {"id": "offer_from_waiting_list", "type": "ordinal",
           "question": "How likely is a waiting-list applicant to be offered a place?", "levels": L3},
 "target": 2, "abstention_cause": None,
 "rationale": "52 of 80 is 65 percent, which is below two thirds and so falls in the 'a third to two thirds' band rather than the likely one."},
{
 "key": "pb17", "domain": "finance_ops",
 "state": {"question_about": "the month-end close finishing by working day 5",
           "history": {"closes_observed": 18, "finished_by_working_day_5": 4},
           "changes": "none: the same team, the same systems and the same reconciliation backlog",
           "entity": "the UK trading company", "close_owner": "M. Rahim",
           "open_reconciling_items_at_month_end": 14, "audit_period": "the 18 months to August 2026",
           "instruction": "Judge the likelihood from the observed history, which is stated to be representative "
                          "of the current month."},
 "field": {"id": "close_on_time", "type": "ordinal",
           "question": "How likely is this month's close to finish by working day 5?", "levels": L5},
 "target": 2, "abstention_cause": None,
 "rationale": "4 of 18 closes is 22 percent, inside the 10-to-35 percent level."},
{
 "key": "pb18", "domain": "software_incidents",
 "state": {"question_about": "a customer renewing after an outage of this size",
           "available_history": {"renewals_after_outages_of_any_size": "not broken down by outage size",
                                 "overall_renewal_rate": "published only as 'healthy' in the board pack"},
           "note": "The customer success team has no figures for this outage size, no comparable cohort and no "
                   "renewal rate expressed as a number; the only published statement is the qualitative one above.",
           "instruction": "Judge the likelihood from the stated evidence and nothing else."},
 "field": {"id": "renewal_likelihood", "type": "ordinal",
           "question": "How likely is this customer to renew?", "levels": L5},
 "target": None, "abstention_cause": "insufficient_evidence",
 "rationale": "No rate is given as a number and no comparable cohort exists, so the word 'healthy' cannot be placed on the stated percentage scale."},
{
 "key": "pb19", "domain": "retail",
 "state": {"question_about": "a returned garment being resellable at full price",
           "history": {"returns_inspected": 250, "resellable_at_full_price": 205},
           "return_reason": "wrong size, the most common reason in the sample",
           "category": "womenswear knitwear", "season": "Autumn/Winter 2026",
           "inspection_standard": "unchanged since the sample was taken",
           "worn_flag": False, "tags_attached": True,
           "instruction": "Use the inspection history for this reason code, which is stated to be representative."},
 "field": {"id": "resellable", "type": "ordinal",
           "question": "How likely is this returned garment to be resellable at full price?", "levels": L4},
 "target": 3, "abstention_cause": None,
 "rationale": "205 of 250 is 82 percent, which sits in the 'above 50 and up to 85 percent' level."},
{
 "key": "pb20", "domain": "logistics",
 "state": "The depot needs both the inbound trunk to arrive before 04:00 and the night shift to be fully staffed "
          "for the morning wave to leave on time. The trunk has arrived before 04:00 on 80 percent of nights, the "
          "night shift has been fully staffed on 75 percent of nights, and the two are stated to be independent of "
          "each other. Nothing else can stop the wave leaving on time.",
 "field": {"id": "wave_on_time", "type": "ordinal",
           "question": "How likely is the morning wave to leave on time?", "levels": L5},
 "target": 3, "abstention_cause": None,
 "rationale": "Two independent conditions at 80 percent and 75 percent both hold 60 percent of the time, which is in the 'above 35 and up to 65 percent' level."},
{
 "key": "pb21", "domain": "healthcare_admin",
 "state": {"question_about": "a discharge summary reaching the GP within 24 hours",
           "history": {"discharges_audited": 300, "summary_sent_within_24_hours": 129},
           "system_changes": "none since the audit",
           "audit_period": "the twelve months to July 2026", "site": "Ridgeway General Hospital",
           "specialty_mix": "the same wards as the audit", "electronic_transfer": True,
           "instruction": "Judge the likelihood from the audit, which is stated to be representative of current "
                          "practice."},
 "field": {"id": "summary_on_time", "type": "ordinal",
           "question": "How likely is this discharge summary to reach the GP within 24 hours?", "levels": L5},
 "target": 3, "abstention_cause": None,
 "rationale": "129 of 300 is 43 percent, which falls in the 'above 35 and up to 65 percent' level."},
{
 "key": "pb22", "domain": "travel",
 "state": "On this route the airline has cancelled 6 of the last 1,000 departures. The aircraft type, the crew "
          "roster pattern and the airports are unchanged, no industrial action is notified, and the forecast is "
          "ordinary for the season. A passenger is asking how likely it is that their departure next week is "
          "cancelled, judged from that record.",
 "field": {"id": "cancellation_likelihood", "type": "ordinal",
           "question": "How likely is this departure to be cancelled?", "levels": L4},
 "target": 1, "abstention_cause": None,
 "rationale": "6 cancellations in 1,000 departures is 0.6 percent, well below the 15 percent boundary for remote."},
{
 "key": "pb23", "domain": "hr",
 "state": {"question_about": "an internal applicant passing the technical stage",
           "history": {"internal_applicants_at_this_stage": 44, "passed": 29},
           "notes": "The exercise, the markers and the pass mark are unchanged, and this applicant's background is "
                    "typical of the sample.",
           "instruction": "Use the stated pass history."},
 "field": {"id": "passes_stage", "type": "ordinal",
           "question": "How likely is this applicant to pass the technical stage?", "levels": L5},
 "target": 4, "abstention_cause": None,
 "rationale": "29 of 44 is 66 percent, just above the 65 percent boundary and so in the 'likely' level."},
{
 "key": "pb24", "domain": "education",
 "state": "Of 160 students who sat the resit for this module across the last four years, 24 failed it again. The "
          "paper format, the pass mark and the support offered before the resit are unchanged. The student in "
          "question attended the revision sessions, as most of the sample did, and there is nothing else to "
          "distinguish them from the sample.",
 "field": {"id": "fails_resit", "type": "ordinal",
           "question": "How likely is this student to fail the resit?", "levels": L5},
 "target": 2, "abstention_cause": None,
 "rationale": "24 failures from 160 resits is 15 percent, inside the 10-to-35 percent level."},
{
 "key": "pb25", "domain": "finance_ops",
 "state": {"question_about": "a disputed charge being resolved in the customer's favour",
           "history": {"disputes_of_this_type_decided": 36, "decided_for_the_customer": 10},
           "evidence_position": "the same documentary position as the sample",
           "dispute_type": "duplicate subscription charge", "amount_gbp": 119.00,
           "decision_maker": "the same adjudication team as the sample",
           "period": "the 24 months to June 2026",
           "instruction": "Judge the likelihood from the decided cases, which are stated to be representative."},
 "field": {"id": "resolved_for_customer", "type": "ordinal",
           "question": "How likely is this dispute to be resolved in the customer's favour?", "levels": L3},
 "target": 1, "abstention_cause": None,
 "rationale": "10 of 36 is 28 percent, which is below a third and so falls in the unlikely band."},
{
 "key": "pb26", "domain": "software_incidents",
 "state": {"question_about": "a flaky test failing on the next run",
           "history": {"runs_observed": 50, "failures": 6},
           "conditions": "the same branch, the same runner image and the same concurrency setting",
           "test": "OrderCheckoutIntegrationTest.concurrent_refund", "suite": "integration",
           "quarantined": False, "recent_code_changes_in_scope": False,
           "instruction": "Judge the likelihood from the observed runs, which are stated to be representative of "
                          "the next one."},
 "field": {"id": "fails_next_run", "type": "ordinal",
           "question": "How likely is this test to fail on the next run?", "levels": L5},
 "target": 2, "abstention_cause": None,
 "rationale": "6 failures in 50 runs is 12 percent, inside the 10-to-35 percent level."},
{
 "key": "pb27", "domain": "retail",
 "state": "Two things must both happen for the window display to be ready for Saturday: the props must arrive from "
          "the warehouse and the visual merchandiser must be available. The props have arrived in time on 90 "
          "percent of previous changeovers and the merchandiser has been available on 95 percent of them, and the "
          "two are stated to be independent. Nothing else is needed for the display to be ready.",
 "field": {"id": "display_ready", "type": "ordinal",
           "question": "How likely is the window display to be ready for Saturday?", "levels": L4},
 "target": 4, "abstention_cause": None,
 "rationale": "Two independent conditions at 90 percent and 95 percent both hold 85.5 percent of the time, just above the 85 percent boundary for near certain."},
{
 "key": "pb28", "domain": "logistics",
 "state": {"question_about": "a driver vacancy being filled within 30 days",
           "history": {"vacancies_posted": 22, "filled_within_30_days": 7},
           "market": "unchanged pay rate, unchanged local competition",
           "depot": "Depot Seacroft", "licence_class_required": "C+E", "shift_pattern": "nights",
           "period": "the 18 months to September 2026", "agency_cover_in_place": True,
           "instruction": "Judge the likelihood from the stated recruitment history, which is stated to be "
                          "representative of this vacancy."},
 "field": {"id": "filled_in_30_days", "type": "ordinal",
           "question": "How likely is this vacancy to be filled within 30 days?", "levels": L5},
 "target": 2, "abstention_cause": None,
 "rationale": "7 of 22 is 32 percent, which is inside the 10-to-35 percent level."},
{
 "key": "pb29", "domain": "healthcare_admin",
 "state": "Of the last 1,200 prescriptions sent electronically to this pharmacy, 1,164 were dispensed the same day. "
          "The pharmacy's opening hours, staffing and stock arrangements are unchanged, the medicine in question is "
          "one it holds routinely, and the prescription was sent at the usual time of day. The question is the "
          "chance that today's prescription is dispensed the same day.",
 "field": {"id": "same_day_dispense", "type": "ordinal",
           "question": "How likely is this prescription to be dispensed the same day?", "levels": L5},
 "target": 5, "abstention_cause": None,
 "rationale": "1,164 of 1,200 is 97 percent, above the 90 percent boundary for very likely."},
{
 "key": "pb30", "domain": "travel",
 "state": {"question_about": "an upgrade clearing at the gate",
           "history": {"requests_on_this_route_last_quarter": 260, "cleared": 39},
           "today": "an ordinary weekday departure with no group booking and the usual cabin configuration",
           "route": "Manchester to Zurich", "cabin_requested": "business", "loyalty_tier": "Silver",
           "period": "the quarter to June 2026", "aircraft_type": "unchanged",
           "instruction": "Judge the likelihood from the stated clearance history, which is stated to be "
                          "representative of this departure."},
 "field": {"id": "upgrade_clears", "type": "ordinal",
           "question": "How likely is this upgrade request to clear at the gate?", "levels": L5},
 "target": 2, "abstention_cause": None,
 "rationale": "39 of 260 is 15 percent, inside the 10-to-35 percent level."},
]
