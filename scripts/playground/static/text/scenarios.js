/* Text-only decisions: the same endpoint and request shape as the photo scenarios, with no image.
   Written 2026-09-24 before any run; the first check run is the one reported (no rewording after results). */

window.SCENARIOS = [
  {
    id: 'crm',
    title: 'Email vs account record',
    kicker: 'Text vs record',
    problem: 'Customers mention changes in passing: a move, a bigger team, a different plan. The CRM stays wrong until someone notices.',
    solves: 'The text twin of the listing check: imajev reads the email against the account and names the field it contradicts.',
    images: [],
    state: {
      account: { company: 'Lindqvist Design', city: 'Berlin', seats: 12, plan: 'Team' },
      email: 'Hi, could you resend last month\'s invoice? Our accountant needs a copy. Thanks, Maja',
    },
    flips: [
      {
        path: 'email',
        label: 'email',
        values: [
          'Hi, could you resend last month\'s invoice? Our accountant needs a copy. Thanks, Maja',
          'Hi, since our move to Munich in August the invoices still show the old address. Could you update it? Thanks, Maja',
          'Hi, we have grown to 40 people this year and everyone needs access. What is the easiest way to add them? Thanks, Maja',
          'Hi, as Enterprise customers we were promised a named account manager. Who is ours? Thanks, Maja',
        ],
        names: ['invoice request', 'moved city', 'bigger team', 'different plan'],
      },
    ],
    questions: {
      contradicted_field: {
        type: 'choice',
        instructions: 'Which field of account does the email contradict?',
        criteria: {
          'account.city': null,
          'account.seats': null,
          'account.plan': null,
          'none of these': 'the email agrees with the account or says nothing about these fields',
        },
      },
    },
    route(a, sure) {
      const f = a.contradicted_field;
      if (!sure(f)) return { kind: 'human', title: 'Send to account manager', detail: 'Not sure whether the record is out of date.' };
      if (f.choice === 'none of these') return { kind: 'ok', title: 'Record is current', detail: 'Nothing in the email contradicts the account.' };
      return { kind: 'flag', title: `Update ${f.choice}`, detail: 'The email says otherwise; confirm with the customer.' };
    },
    checks: [
      { flips: {}, expect: 'ok', answer: { contradicted_field: 'none of these' } },
      { flips: { email: 'moved city' }, expect: 'flag', answer: { contradicted_field: 'account.city' } },
      { flips: { email: 'bigger team' }, expect: 'flag', answer: { contradicted_field: 'account.seats' } },
      { flips: { email: 'different plan' }, expect: 'flag', answer: { contradicted_field: 'account.plan' } },
    ],
  },
  {
    id: 'refund',
    title: 'Refund policy check',
    kicker: 'Policy vs order',
    problem: 'Refund rules are simple on paper, but agents apply them from memory, and every exception costs money or a customer.',
    solves: 'imajev reads the policy and the order together and names the rule that blocks a refund, if any.',
    images: [],
    state: {
      policy: {
        window: 'refunds within 30 days of delivery',
        opened: 'opened items can be refunded only if faulty',
        final_sale: 'final-sale items cannot be refunded',
      },
      order: { item: 'wireless headphones', days_since_delivery: 12, opened: 'no', faulty: 'no', final_sale: 'no' },
    },
    flips: [
      { path: 'order.days_since_delivery', label: 'order.days_since_delivery', values: [12, 41] },
      { path: 'order.opened', label: 'order.opened', values: ['no', 'yes'] },
      { path: 'order.final_sale', label: 'order.final_sale', values: ['no', 'yes'] },
    ],
    questions: {
      blocking_rule: {
        type: 'choice',
        instructions: 'Which rule of policy blocks a refund for this order?',
        criteria: {
          'policy.window': null,
          'policy.opened': null,
          'policy.final_sale': null,
          'none: the refund is allowed': null,
        },
      },
    },
    route(a, sure) {
      const r = a.blocking_rule;
      if (!sure(r)) return { kind: 'human', title: 'Ask a team lead', detail: 'Not sure how the policy applies.' };
      if (r.choice.startsWith('none')) return { kind: 'ok', title: 'Approve the refund', detail: 'No rule blocks it.' };
      return { kind: 'flag', title: 'Decline politely', detail: `Blocked by ${r.choice}.` };
    },
    checks: [
      { flips: {}, expect: 'ok', answer: { blocking_rule: 'none: the refund is allowed' } },
      { flips: { 'order.days_since_delivery': 41 }, expect: 'flag', answer: { blocking_rule: 'policy.window' } },
      { flips: { 'order.opened': 'yes' }, expect: 'flag', answer: { blocking_rule: 'policy.opened' } },
      { flips: { 'order.final_sale': 'yes' }, expect: 'flag', answer: { blocking_rule: 'policy.final_sale' } },
    ],
  },
  {
    id: 'invoice',
    title: 'Is the invoice paid?',
    kicker: 'Knowing when it cannot tell',
    problem: 'Automations that guess are worse than ones that stop. A reminder sent for a paid invoice costs goodwill.',
    solves: 'When the record says nothing about payment, imajev puts its weight on unknown and the app asks a person.',
    images: [],
    state: {
      invoice: { id: 'INV-2291', customer: 'Harbour Coffee Ltd', amount: '1,240 EUR', due: '1 September 2026', payment_note: '' },
    },
    flips: [
      {
        path: 'invoice.payment_note',
        label: 'invoice.payment_note',
        values: ['', 'Paid in full by bank transfer on 29 August.', 'Reminder sent 5 September; no payment received yet.'],
        names: ['nothing recorded', 'paid', 'not paid'],
      },
    ],
    questions: {
      paid: {
        type: 'noul',
        instructions: 'The customer has paid this invoice.',
        criteria: YES_NO('the record shows the payment arrived', 'the record shows it has not been paid'),
      },
    },
    route(a, sure) {
      const p = a.paid;
      if (!sure(p)) return { kind: 'human', title: 'Ask accounts', detail: 'The record does not say whether it was paid.' };
      return p.noul >= 0.5
        ? { kind: 'ok', title: 'Close the invoice', detail: 'Payment recorded.' }
        : { kind: 'flag', title: 'Send a second reminder', detail: 'No payment yet.' };
    },
    checks: [
      { flips: {}, expect: 'human' },
      { flips: { 'invoice.payment_note': 'paid' }, expect: 'ok', answer: { paid: 'yes' } },
      { flips: { 'invoice.payment_note': 'not paid' }, expect: 'flag', answer: { paid: 'no' } },
    ],
  },
  {
    id: 'moderation',
    title: 'Community post review',
    kicker: 'Moderation',
    problem: 'A forum gets thousands of posts a day. Most are fine; the few that are not need to come down fast.',
    solves: 'One choice over your own policy categories, with a probability the app can threshold.',
    images: [],
    state: {
      forum: 'a neighbourhood gardening forum',
      post: 'Tip for anyone fighting slugs: a ring of crushed eggshells around the lettuce worked better for me than beer traps.',
    },
    flips: [
      {
        path: 'post',
        label: 'post',
        values: [
          'Tip for anyone fighting slugs: a ring of crushed eggshells around the lettuce worked better for me than beer traps.',
          'BEST PRICES on garden furniture!!! 70% off today only, order now at cheap-patio-deals dot com',
          'Dave on Elm Street is a thief and a liar. Everyone should go and tell him what they think of him.',
          'The man who took my hose lives at 14 Elm Street, his phone is 0151 555 0199, call him day and night.',
        ],
        names: ['gardening tip', 'spam', 'harassment', 'private details'],
      },
    ],
    questions: {
      verdict: {
        type: 'choice',
        instructions: 'Under the forum rules, how should this post be handled?',
        criteria: {
          publish: 'on topic and respectful',
          'remove: spam or advertising': null,
          'remove: harassment of a person': null,
          "remove: shares someone's private details": null,
        },
      },
    },
    route(a, sure) {
      const v = a.verdict;
      if (!sure(v)) return { kind: 'human', title: 'Send to a moderator', detail: 'Not sure enough to act.' };
      return v.choice === 'publish'
        ? { kind: 'ok', title: 'Publish', detail: 'No rule applies.' }
        : { kind: 'flag', title: 'Take it down', detail: v.choice.replace('remove: ', '') };
    },
    checks: [
      { flips: {}, expect: 'ok', answer: { verdict: 'publish' } },
      { flips: { post: 'spam' }, expect: 'flag', answer: { verdict: 'remove: spam or advertising' } },
      { flips: { post: 'harassment' }, expect: 'flag', answer: { verdict: 'remove: harassment of a person' } },
      { flips: { post: 'private details' }, expect: 'flag', answer: { verdict: "remove: shares someone's private details" } },
    ],
  },
  {
    id: 'review',
    title: 'Review routing',
    kicker: 'Score + choice',
    problem: 'Reviews pile up faster than anyone reads them. The bad ones should reach the team that can fix the cause.',
    solves: 'A score for how unhappy the customer is and a choice for what went wrong, in one request.',
    images: [],
    state: {
      review: 'Arrived two days early and the jacket is even warmer than I hoped. Stitching is neat. Would buy again.',
    },
    flips: [
      {
        path: 'review',
        label: 'review',
        values: [
          'Arrived two days early and the jacket is even warmer than I hoped. Stitching is neat. Would buy again.',
          'The jacket itself is lovely, but it took three weeks to arrive and the tracking link never worked.',
          'After two wears the zip broke and a seam on the sleeve came apart. Very disappointed with the quality.',
        ],
        names: ['happy', 'late delivery', 'broke'],
      },
    ],
    questions: {
      satisfaction: {
        type: 'score',
        instructions: 'How satisfied is the customer?',
        criteria: ['very unhappy', 'unhappy', 'mixed', 'happy', 'very happy'],
      },
      problem_area: {
        type: 'choice',
        instructions: 'Which part of the purchase does the review complain about?',
        criteria: {
          delivery: 'shipping time, tracking, packaging',
          'product quality': 'the item is faulty, broke or is badly made',
          price: null,
          'no complaint': null,
        },
      },
    },
    route(a, sure) {
      const p = a.problem_area;
      if (!sure(p)) return { kind: 'human', title: 'Read by hand', detail: 'The complaint is unclear.' };
      if (p.choice === 'no complaint') return { kind: 'ok', title: 'Thank the customer', detail: `Satisfaction ${a.satisfaction.score.toFixed(1)} of 4.` };
      return { kind: 'flag', title: `Send to the ${p.choice} team`, detail: `Satisfaction ${a.satisfaction.score.toFixed(1)} of 4.` };
    },
    checks: [
      { flips: {}, expect: 'ok', answer: { problem_area: 'no complaint' } },
      { flips: { review: 'late delivery' }, expect: 'flag', answer: { problem_area: 'delivery' } },
      { flips: { review: 'broke' }, expect: 'flag', answer: { problem_area: 'product quality' } },
    ],
  },
  {
    id: 'inbox',
    title: 'Does this email need action?',
    kicker: 'Inbox triage',
    problem: 'Most email is FYI. The few that ask for something by a date get buried with the rest.',
    solves: 'A yes/no on whether the sender asks for action, and a choice for how soon.',
    images: [],
    state: {
      today: 'Thursday 24 September 2026',
      email: 'FYI: the office will be closed on 3 October for the public holiday. No action needed.',
    },
    flips: [
      {
        path: 'email',
        label: 'email',
        values: [
          'FYI: the office will be closed on 3 October for the public holiday. No action needed.',
          'Could you sign the attached supplier contract and send it back by tomorrow morning? Legal needs it before the board meeting.',
          'When you get a chance in the next few weeks, could you review the draft onboarding guide and leave comments?',
        ],
        names: ['FYI', 'sign by tomorrow', 'review in weeks'],
      },
    ],
    questions: {
      needs_action: {
        type: 'noul',
        instructions: 'The sender asks the reader to do something.',
      },
      deadline: {
        type: 'choice',
        instructions: 'By when does the reader need to act?',
        criteria: { 'today or tomorrow': null, 'within a week': null, 'no deadline or later': null },
      },
    },
    route(a, sure) {
      const n = a.needs_action;
      if (!sure(n)) return { kind: 'human', title: 'Leave in inbox', detail: 'Not sure it needs anything.' };
      if (n.noul < 0.5) return { kind: 'ok', title: 'Archive', detail: 'Nothing to do.' };
      const d = a.deadline;
      return { kind: 'flag', title: 'Add to tasks', detail: sure(d) ? `Due: ${d.choice}.` : 'Due date unclear.' };
    },
    checks: [
      { flips: {}, expect: 'ok', answer: { needs_action: 'no' } },
      { flips: { email: 'sign by tomorrow' }, expect: 'flag', answer: { needs_action: 'yes', deadline: 'today or tomorrow' } },
      { flips: { email: 'review in weeks' }, expect: 'flag', answer: { needs_action: 'yes', deadline: 'no deadline or later' } },
    ],
  },
];
