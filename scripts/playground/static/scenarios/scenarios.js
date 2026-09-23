/* imajev scenarios — real-world decisions, one per page section.

   Each scenario is a Jev-shaped request (state + questions) over 0–2 photos, plus:
   - flips:    state fields the viewer can switch with one click (the record changes, the photo does not);
   - variants: alternative photos for one image slot (the photo changes, the record does not);
   - route(a, sure): what an application would do with the answers. It is ordinary app code over the
                     model's probabilities; `sure(answer)` applies the page's confidence threshold.
   - checks:   the expected route kind (and optionally answers) for named flip/variant combinations, as an
               array or a function returning one. scripts/playground/verify_scenarios.mjs runs them against a
               live server; only combinations that pass belong in a demo.

   Helpers (YES_NO, sureAt, buildCase) live in engine.js, shared with the wardrobe page and the checker.
   Photos and their licences: assets/attribution.json (built by scripts/playground/build_scenario_assets.py). */

window.SCENARIOS = [
  {
    id: 'listing',
    title: 'Marketplace listing check',
    kicker: 'Photo vs listing',
    problem: 'A seller uploads a photo and fills in the listing. When the two disagree, buyers return the item. Nobody can check every listing by hand.',
    solves: 'imajev reads the listing as state, looks at the photo, and names the field that is wrong, or says it cannot tell.',
    images: [{ slot: 'listing photo', src: 'assets/loafers.jpg' }],
    state: { listing: { title: "Men's suede boat shoes", color: 'beige', product_type: 'shoe' } },
    flips: [
      { path: 'listing.color', values: ['beige', 'black', 'red'] },
      { path: 'listing.product_type', values: ['shoe', 'handbag', 'watch'] },
    ],
    questions: {
      contradicted_field: {
        type: 'choice',
        instructions: 'Which field of `listing` does this photo contradict?',
        criteria: {
          'listing.color': 'the photo shows a different colour than `listing.color`',
          'listing.product_type': 'the photo shows a different kind of product than `listing.product_type`',
          'none of these': 'the photo is consistent with the listing',
        },
      },
      color_matches: {
        type: 'noul',
        instructions: 'The product in the photo matches `listing.color`.',
        criteria: YES_NO('the product is the listed colour', 'the product is clearly a different colour from the listed one'),
      },
      type_matches: {
        type: 'noul',
        instructions: 'The photo shows the kind of product given in `listing.product_type`.',
        criteria: YES_NO('the product is that kind of product', 'it is clearly a different kind of product'),
      },
    },
    route(a, sure) {
      const field = a.contradicted_field;
      if (!sure(field)) return { kind: 'human', title: 'Send to a reviewer', detail: 'The model is not sure which field, if any, is wrong.' };
      if (field.choice === 'none of these') {
        if (a.color_matches.noul < 0.5 || a.type_matches.noul < 0.5)
          return { kind: 'human', title: 'Send to a reviewer', detail: 'The field check and the per-field checks disagree.' };
        return { kind: 'ok', title: 'Publish the listing', detail: 'Photo and listing agree.' };
      }
      return { kind: 'flag', title: `Hold: ${field.choice} is wrong`, detail: 'Ask the seller to fix this field before the listing goes live.' };
    },
    checks: [
      { flips: {}, expect: 'ok' },
      { flips: { 'listing.color': 'black' }, expect: 'flag', answer: { contradicted_field: 'listing.color' } },
      { flips: { 'listing.color': 'red' }, expect: 'flag', answer: { contradicted_field: 'listing.color' } },
      { flips: { 'listing.product_type': 'handbag' }, expect: 'flag', answer: { contradicted_field: 'listing.product_type' } },
      { flips: { 'listing.product_type': 'watch' }, expect: 'flag', answer: { contradicted_field: 'listing.product_type' } },
    ],
  },

  {
    id: 'returns',
    title: 'Return check',
    kicker: 'Two photos',
    problem: 'A customer sends back an item for a refund. Is it the product you shipped, or something else in the box?',
    solves: 'imajev compares the catalogue photo with the photo taken at the returns desk and gives the refund system a probability, not a paragraph.',
    images: [
      { slot: 'what we shipped', src: 'assets/slipon-listing.jpg' },
      { slot: 'what came back', src: 'assets/slipon-return.jpg' },
    ],
    variants: {
      slot: 1,
      options: [
        { label: 'same shoe, other angle', src: 'assets/slipon-return.jpg' },
        { label: 'a different shoe', src: 'assets/loafers.jpg' },
      ],
    },
    state: { order: { id: 'ORD-20931', item: "Women's knit slip-on sneakers", colour: 'pink and black' }, return: { reason: "doesn't fit" } },
    questions: {
      same_item: {
        type: 'noul',
        instructions: 'The second photo shows the same product as the first photo.',
        criteria: YES_NO('the second photo shows the same product as the first photo', 'the second photo shows a different product'),
      },
      returned_colour: {
        type: 'choice',
        instructions: 'What is the main colour of the product in the second photo?',
        criteria: { pink: null, beige: null, black: null, white: null, blue: null },
      },
    },
    route(a, sure) {
      const p = a.same_item;
      if (!sure(p)) return { kind: 'human', title: 'Inspect by hand', detail: 'The model cannot tell whether this is the shipped product.' };
      if (p.noul >= 0.5) return { kind: 'ok', title: 'Approve the refund', detail: 'The returned item matches what was shipped.' };
      return { kind: 'flag', title: 'Hold the refund', detail: `The returned item looks different (${a.returned_colour.choice}).` };
    },
    checks: [
      { variant: 'same shoe, other angle', expect: 'ok' },
      { variant: 'a different shoe', expect: 'flag' },
    ],
  },

  {
    id: 'qc',
    title: 'Production-line QC',
    kicker: 'Reference vs part',
    problem: 'A camera photographs every part on the line. Rules-based vision breaks when parts vary, and a person cannot look at every photo.',
    solves: 'imajev compares each part with a known-good reference and says whether it is faulty and how.',
    images: [
      { slot: 'known-good reference', src: 'assets/fryum-reference.jpg' },
      { slot: 'part under inspection', src: 'assets/fryum-chipped.jpg' },
    ],
    variants: {
      slot: 1,
      options: [
        { label: 'chipped part', src: 'assets/fryum-chipped.jpg' },
        { label: 'good part', src: 'assets/fryum-good.jpg' },
      ],
    },
    state: { line: { station: 'QC-3', product: 'pipe-shaped snack', reference: 'image 1 is a known-good part' } },
    questions: {
      has_fault: {
        type: 'noul',
        instructions: 'Given the reference part in the first image, does the part in the second image have a fault?',
        criteria: YES_NO('the part in the second image is damaged or defective', 'the part in the second image is as good as the reference'),
      },
      fault_type: {
        type: 'choice',
        instructions: 'What is wrong with the part in the second image?',
        criteria: {
          'chipped or broken edge': null,
          'crack': null,
          'discoloured or burnt': null,
          'nothing is wrong': null,
        },
      },
    },
    route(a, sure) {
      const p = a.has_fault;
      if (!sure(p)) return { kind: 'human', title: 'Send to an inspector', detail: 'Not sure enough to pass or reject.' };
      if (p.noul >= 0.5) return { kind: 'flag', title: 'Reject the part', detail: `Most likely: ${a.fault_type.choice}.` };
      return { kind: 'ok', title: 'Pass', detail: 'No fault compared with the reference.' };
    },
    checks: [
      { variant: 'chipped part', expect: 'flag' },
      { variant: 'good part', expect: 'ok' },
    ],
  },

  {
    id: 'retake',
    title: '"Retake your photo"',
    kicker: 'Knowing it cannot tell',
    problem: 'A blind user photographs a can and asks what is inside. The photo shows only the lid. A normal model guesses anyway, confidently.',
    solves: 'imajev checks whether the photo can answer the question at all, and if not, why, so the app can ask for a better photo instead of guessing.',
    images: [{ slot: "user's photo", src: 'assets/can-lid.jpg' }],
    state: { user: { question: 'What kind of soup is in this can?' } },
    flips: [
      { path: 'user.question', values: ['What kind of soup is in this can?', 'What is the use-by date?', 'What shape is this object?'] },
    ],
    questions: {
      answerable: {
        type: 'noul',
        instructions: 'The question in `user.question` can be answered from this photo.',
        criteria: YES_NO('the photo shows what is needed to answer the question', 'the photo does not show what the question needs'),
      },
      problem: {
        type: 'choice',
        instructions: 'Which quality problem does this photo have?',
        criteria: {
          'bad framing': 'the subject is cut off or the important part is not in view',
          'blurry': null,
          'too dark': null,
          'too bright': null,
          'no problem': null,
        },
      },
    },
    route(a, sure) {
      const p = a.answerable;
      if (!sure(p)) return { kind: 'human', title: 'Answer with a warning', detail: 'The model is unsure the photo is enough.' };
      if (p.noul >= 0.5) return { kind: 'ok', title: 'Answer the question', detail: 'The photo shows what the question needs.' };
      const hints = {
        'bad framing': 'move the camera so the part you are asking about is in view',
        'blurry': 'hold the phone still and a little further away',
        'too dark': 'turn on a light or move closer to a window',
        'too bright': 'move out of direct light',
        'no problem': 'show the part of the object the question is about',
      };
      return { kind: 'flag', title: 'Ask for a new photo', detail: `Retake: ${hints[a.problem.choice]}.` };
    },
    checks: [
      { flips: {}, expect: 'flag' },
      { flips: { 'user.question': 'What is the use-by date?' }, expect: 'flag' },
      { flips: { 'user.question': 'What shape is this object?' }, expect: 'ok' },
    ],
  },

  {
    id: 'archive',
    title: 'Archive record audit',
    kicker: 'Photo vs catalogue',
    problem: 'Museums and photo archives hold millions of records typed by hand decades ago. Some are wrong and nobody knows which.',
    solves: 'imajev checks each catalogue record against its scan and points to the field that disagrees.',
    images: [{ slot: 'archive scan', src: 'assets/cat-on-chair.jpg' }],
    state: { record: { accession: 'PH-1904-117', subject: 'cat', medium: 'black-and-white photograph', setting: 'indoors' } },
    flips: [
      { path: 'record.subject', values: ['cat', 'dog', 'horse'] },
      { path: 'record.medium', values: ['black-and-white photograph', 'colour photograph'] },
    ],
    questions: {
      wrong_field: {
        type: 'choice',
        instructions: 'Which field of `record` does this photograph contradict?',
        criteria: {
          'record.subject': 'the animal is not the one recorded',
          'record.medium': 'the medium is wrong (e.g. the photo is black and white)',
          'record.setting': 'the scene is not the recorded setting',
          'none': 'the record is consistent with the photo',
        },
      },
      subject_matches: {
        type: 'noul',
        instructions: 'The photograph shows the animal recorded in `record.subject`.',
        criteria: YES_NO('the animal in the photo is the recorded subject', 'it is clearly a different animal'),
      },
    },
    route(a, sure) {
      const field = a.wrong_field;
      if (!sure(field)) return { kind: 'human', title: 'Queue for a cataloguer', detail: 'Not sure enough to change the record.' };
      if (field.choice === 'none') return { kind: 'ok', title: 'Record verified', detail: 'Every field agrees with the scan.' };
      return { kind: 'flag', title: `Correct ${field.choice}`, detail: 'The scan contradicts this field.' };
    },
    checks: [
      { flips: {}, expect: 'ok' },
      { flips: { 'record.subject': 'dog' }, expect: 'flag', answer: { wrong_field: 'record.subject' } },
      { flips: { 'record.subject': 'horse' }, expect: 'flag', answer: { wrong_field: 'record.subject' } },
      { flips: { 'record.medium': 'colour photograph' }, expect: 'flag', answer: { wrong_field: 'record.medium' } },
    ],
  },

  {
    id: 'audit',
    title: 'Before / after audit',
    kicker: 'What changed?',
    problem: 'Collections, rentals and stockrooms are photographed before and after. Spotting what moved between two photos is slow, tedious work.',
    solves: 'imajev compares the two photos, names the kind of change, and says which checklist field must be updated.',
    images: [
      { slot: 'before', src: 'assets/tray-before.jpg' },
      { slot: 'after', src: 'assets/tray-after.jpg' },
    ],
    variants: {
      slot: 1,
      options: [
        { label: 'labels missing', src: 'assets/tray-after.jpg' },
        { label: 'unchanged', src: 'assets/tray-before.jpg' },
      ],
    },
    state: {
      checklist: {
        subject: 'a museum specimen tray',
        all_items_still_there: 'yes, everything in the reference is still in the frame',
        no_new_items: 'yes, nothing has been added',
        colours_as_before: 'yes, every object keeps its colour',
      },
      task: { what_counts: 'treat the two photos as matching unless an item was added, removed, recoloured or hidden' },
    },
    questions: {
      what_changed: {
        type: 'choice',
        instructions: 'Image 1 is the reference, image 2 the target. Following `task.what_counts`, what changed?',
        criteria: {
          'nothing relevant changed': null,
          'an object was added': null,
          'an object was removed': null,
          'an object changed colour': null,
          'something was covered or blurred out': null,
        },
      },
      field_to_update: {
        type: 'choice',
        instructions: 'Somebody must update `checklist` after image 2 was taken. Which field needs changing? Changes that `task.what_counts` says to ignore do not count.',
        criteria: {
          'checklist.all_items_still_there': "image 2 shows that 'all items still there' no longer holds",
          'checklist.no_new_items': "image 2 shows that 'no new items' no longer holds",
          'checklist.colours_as_before': "image 2 shows that 'colours as before' no longer holds",
          'none of these': 'every field still holds for image 2',
        },
      },
    },
    route(a, sure) {
      const change = a.what_changed;
      if (!sure(change)) return { kind: 'human', title: 'Compare by hand', detail: 'The model is not sure what changed.' };
      if (change.choice === 'nothing relevant changed') return { kind: 'ok', title: 'No action', detail: 'Nothing relevant changed.' };
      return { kind: 'flag', title: 'Open an incident', detail: `${change.choice[0].toUpperCase()}${change.choice.slice(1)}; update ${a.field_to_update.choice}.` };
    },
    checks: [
      { variant: 'labels missing', expect: 'flag' },
      { variant: 'unchanged', expect: 'ok' },
    ],
  },

  {
    id: 'ticket',
    title: 'Support ticket triage',
    kicker: 'Text only, same API',
    problem: 'Every ticket needs an owner and a priority before anyone reads it properly. Keyword rules misroute; a chat model needs parsing.',
    solves: 'The same endpoint with no photo: imajev routes the ticket, flags urgency and scores frustration in one pass.',
    images: [],
    state: {
      ticket: {
        id: 'TKT-48812',
        body: 'We were billed 480 USD twice on 3 September for the same 12 seats. I have now written three times and the only reply was an automated one. Our finance close is on Wednesday and I need the duplicate reversed before then.',
        plan: 'Team (12 seats)',
      },
    },
    flips: [
      {
        path: 'ticket.body',
        label: 'ticket.body',
        values: [
          'We were billed 480 USD twice on 3 September for the same 12 seats. I have now written three times and the only reply was an automated one. Our finance close is on Wednesday and I need the duplicate reversed before then.',
          'Quick question when you have a moment: is there a way to export our dashboards to PDF? No rush at all, just planning next quarter.',
          'Since this morning every login fails with error 502 for our whole team. We cannot work at all and have a client demo in two hours.',
        ],
        names: ['double charge', 'feature question', 'outage'],
      },
    ],
    questions: {
      department: {
        type: 'choice',
        instructions: 'Which department should own this ticket?',
        criteria: {
          billing: 'payments, invoices, refunds and subscription charges',
          technical_support: 'the product is broken, erroring or unavailable',
          sales: 'new seats, upgrades, quotes and renewals',
          product_questions: 'how-to questions and feature requests',
        },
      },
      urgent: {
        type: 'noul',
        instructions: 'This ticket needs a reply today.',
        criteria: YES_NO('money is at stake, work is blocked, or the customer has a deadline within days', 'it can wait for the normal queue'),
      },
      frustration: {
        type: 'score',
        instructions: 'How frustrated does the customer sound?',
        criteria: ['calm and matter-of-fact', 'impatient: chasing a reply or repeating themselves', 'angry: threatening to escalate, churn or dispute the charge'],
      },
    },
    route(a, sure) {
      const d = a.department;
      if (!sure(d)) return { kind: 'human', title: 'Triage by hand', detail: 'No department is a clear owner.' };
      const urgent = sure(a.urgent) && a.urgent.noul >= 0.5;
      return {
        kind: urgent ? 'flag' : 'ok',
        title: `Route to ${d.choice.replace('_', ' ')}`,
        detail: urgent ? 'Priority: reply today.' : 'Priority: normal queue.',
      };
    },
    checks: [
      { flips: {}, expect: 'flag', answer: { department: 'billing' } },
      { flips: { 'ticket.body': 'feature question' }, expect: 'ok', answer: { department: 'product_questions' } },
      { flips: { 'ticket.body': 'outage' }, expect: 'flag', answer: { department: 'technical_support' } },
    ],
  },
];
