/* Shared by every scenario page (scenarios/, wardrobe/) and scripts/playground/verify_scenarios.mjs.
   Load before the page's scenario data. */

window.YES_NO = (yes, no) => ({ true: yes, false: no });

/* The one confidence rule: an answer is "sure" when it did not abstain and its top outcome has at
   least probability t (for noul, the larger of P(yes) and P(no)). Below that, routes send the case
   to a person. The checker applies the same rule at DEFAULT_THRESHOLD. */
window.DEFAULT_THRESHOLD = 0.8;
window.sureAt = t => answer => {
  if (!answer || answer.abstained) return false;
  if (answer.type === 'noul') return Math.max(answer.noul, 1 - answer.noul) >= t;
  return Math.max(...Object.values(answer.probabilities || { x: 0 })) >= t;
};

/* Build the request for one scenario under a set of flips and a photo variant; the page and the
   checker both use it, so a checked combination is exactly what the page sends.
   `flips` maps a state path to a value, or to one of that flip's `names`. */
window.buildCase = (scenario, { flips = {}, variant = null } = {}) => {
  const state = JSON.parse(JSON.stringify(scenario.state));
  for (const [path, wanted] of Object.entries(flips)) {
    const flip = (scenario.flips || []).find(f => f.path === path);
    const byName = flip && flip.names ? flip.names.indexOf(wanted) : -1;
    const value = byName >= 0 ? flip.values[byName] : wanted;
    const keys = path.split('.');
    let node = state;
    for (const key of keys.slice(0, -1)) node = node[key];
    node[keys[keys.length - 1]] = value;
  }
  const images = scenario.images.map(image => image.src);
  if (scenario.variants && variant != null) {
    const option = scenario.variants.options.find(o => o.label === variant);
    if (option) images[scenario.variants.slot] = option.src;
  }
  return { state, questions: scenario.questions, images };
};

/* Every flip value and every photo variant, crossed: the one-click combinations a viewer can reach. */
window.allCombinations = scenario => {
  let combos = [{ flips: {}, variant: scenario.variants ? scenario.variants.options[0].label : null }];
  for (const flip of scenario.flips || []) {
    combos = combos.flatMap(c => flip.values.map((v, i) => ({ ...c, flips: { ...c.flips, [flip.path]: flip.names ? flip.names[i] : v } })));
  }
  if (scenario.variants) combos = combos.flatMap(c => scenario.variants.options.map(o => ({ ...c, variant: o.label })));
  return combos;
};

window.checksOf = scenario => (typeof scenario.checks === 'function' ? scenario.checks() : scenario.checks);
