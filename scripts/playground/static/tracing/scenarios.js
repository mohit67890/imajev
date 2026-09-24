/* imajev tracing pad — the request, the app rule and the checks, shared by the page (index.html) and the
   checker (node scripts/playground/verify_scenarios.mjs --pack tracing).

   imajev sees only the child's strokes (never the dotted guide, which would give the answer away) and is not
   told which character was asked for; it answers which character was written. The app compares that with the
   task, measures how much of each guide line the strokes cover, and turns both into kid-friendly feedback. Checked on generated sample tracings
   (build_tracing_samples.py): a good trace, a half-finished one, the wrong character, a scribble. */
(function (root) {
  const CHARS = ['A', 'C', 'E', 'H', 'L', 'O', 'T', '1', '4', '7'];
  const SCRIBBLE = 'a scribble, not a letter or number';
  const WRONG = { A: 'H', C: 'O', E: 'L', H: 'A', L: 'E', O: 'C', T: '7', 1: '7', 4: 'A', 7: '1' };
  const isDigit = c => /\d/.test(c);
  const noun = c => (isDigit(c) ? `the number ${c}` : `the letter ${c}`);

  /* imajev is not told which character was asked for: a task in the record primes it to "see" that character. */
  function questions() {
    return {
      drawn: {
        type: 'choice',
        instructions: 'This is a child\u2019s handwriting practice. Which letter or number is written in the picture?',
        criteria: { ...Object.fromEntries(CHARS.map(c => [c, isDigit(c) ? `the number ${c}` : `the capital letter ${c}`])), [SCRIBBLE]: null },
      },
    };
  }

  function state() { return {}; }

  /* Kid-facing feedback. imajev says what was written; the app compares it with the task and measures how much
     of the weakest guide line the strokes cover (coverage, 0..1; null in the checker, which tests imajev's part only). */
  function route(a, sure, char, coverage = null) {
    const drawn = a.drawn.choice;
    if (!sure(a.drawn)) return { kind: 'human', title: 'Nice try!', detail: 'Let\u2019s do it once more, slowly along the dots.' };
    if (drawn === SCRIBBLE) return { kind: 'flag', title: 'Let\u2019s try again', detail: `Follow the dotted lines to make ${noun(char)}.` };
    if (drawn !== char) return { kind: 'flag', title: `That looks like ${isDigit(drawn) ? 'a' : /^[AEHLO]/.test(drawn) ? 'an' : 'a'} ${drawn}!`, detail: `Now try ${noun(char)}.` };
    if (coverage != null && coverage < COVERED) return { kind: 'flag', title: 'Almost there!', detail: `Finish all the dotted lines of your ${char}.` };
    return { kind: 'ok', title: 'Well done! \u2b50', detail: `That is a great ${char}.` };
  }
  const COVERED = 0.8;    // share of every guide line the strokes must pass over (app geometry, not the model)

  const SCENARIOS = CHARS.map(char => ({
    id: `trace-${char}`,
    title: `Trace ${char}`,
    kicker: 'Tracing pad',
    images: [{ slot: 'the child’s tracing', src: `samples/${char}-good.png` }],
    variants: {
      slot: 0,
      options: ['good', 'half', 'wrong', 'scribble'].map(kind => ({ label: kind, src: `samples/${char}-${kind}.png` })),
    },
    state: state(),
    questions: questions(),
    route: (a, sure) => route(a, sure, char),
    // Completeness of the half-finished sample is the app's coverage check, so only imajev's reading is checked here.
    checks: [
      { variant: 'good', expect: 'ok', answer: { drawn: char } },
      { variant: 'wrong', expect: 'flag', answer: { drawn: WRONG[char] } },
      { variant: 'scribble', expect: ['flag', 'human'], answer: { drawn: [SCRIBBLE] } },
    ],
  }));

  root.TRACING = { CHARS, SCRIBBLE, COVERED, questions, state, route, noun, isDigit };
  root.SCENARIOS = SCENARIOS;
})(typeof window !== 'undefined' ? window : globalThis);
