"""State-grounded question templates for decision-v2.

Same conventions as `scripts/v2/templates/photo.py`: every public function here is pure -- it takes
an `inputs` mapping describing one photo plus a seeded `random.Random`, and returns a question
(and its state) in the shape `vision_decision.contracts.Request` validates.  Nothing is read from
disk, nothing is written, and no answer is ever produced: decision-v2 targets come from the 9B
teacher.

What is different from `photo.py` is *where the question lives*.  In `photo.py` the state is
decoration -- aligned, irrelevant or contradictory flavour text -- and every question can be
answered from the pixels alone.  Here the STATE makes a specific, checkable claim and the question
cannot be answered without reading it: the same photograph, with a different state, has a
different right answer.  That is the gap the testers found ("the 2B does not relate the text in the
state with the image") and the measured failure on ABO listing material (3/12 altered materials
detected) and on true listings (26% called a mismatch).

`inputs` keys
-------------
    id          str   -- stable id of the photo, used only for reproducible wording
    source      str   -- which image source the photo came from (wording only)
    subject     str   -- a noun phrase for the main subject ("a pair of earrings", "a dog")
    present     list  -- noun phrases a human verified (Open Images Confidence==1) or a caption
                         claimed are in the photo
    absent      list  -- noun phrases verified/believed NOT in the photo.  These are the best
                         false claims available: they are specific, plausible and really wrong.
    attrs       dict  -- ground attributes of the subject where the source has structured fields
                         (ABO): colour, material, product_type, pattern, shape, finish
    hidden      dict  -- fields the source carries that NO photograph can settle (model_number,
                         item_weight, item_dimensions, model_year, fabric_type)
    peers       list  -- plausible alternative product types / subjects for a false category claim
    title       str   -- upstream title, ONLY when the converter has checked it cannot answer the
                         question by itself; "" otherwise.  A caption is never passed in here.

Every template carries at least eight paraphrases, and about a third of the questions attach a
short criterion to each option (Jev's "option -> what it means" style).  Questions refer to the
state with backticks, exactly as Jev's own instructions do (`claim`, `listing.material`), so the
model has to resolve a path into the state before it can look at the photograph.  The `unknown`
option is added by the loader and is never listed here.
"""
from __future__ import annotations

FAMILIES = (
    "claim_supported",
    "which_field_conflicts",
    "count_matches",
    "label_text_matches",
    "condition_report",
    "instruction_changes_answer",
)
PAIRED_FAMILIES = ("instruction_changes_answer",)

# Share of questions that carry Jev-style "option -> what it means" criteria.
CRITERIA_SHARE = 0.30

# --------------------------------------------------------------------------- vocabularies
COLOURS = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white",
    "grey", "beige", "gold", "silver", "turquoise", "navy", "cream", "burgundy",
]
COLOUR_CRITERIA = {
    "red": "clear red, including crimson and scarlet",
    "orange": "orange, between red and yellow",
    "yellow": "yellow, including mustard and lemon",
    "green": "any green, from olive to emerald",
    "blue": "mid blue, not navy and not turquoise",
    "purple": "purple or violet, including mauve",
    "pink": "pink, including rose and magenta",
    "brown": "brown, including tan and chocolate",
    "black": "black or near-black",
    "white": "white or off-white",
    "grey": "neutral grey between black and white",
    "beige": "pale sandy neutral, darker than white",
    "gold": "metallic yellow, like brass or gilt",
    "silver": "metallic grey, like steel or chrome",
    "turquoise": "blue-green, like teal or aqua",
    "navy": "very dark blue",
    "cream": "warm off-white, paler than beige",
    "burgundy": "dark red with a purple cast",
}
# A swapped colour has to be clearly different in a photograph but still a value a real listing
# would carry: "black" -> "white" is a fair test, "black" -> "jet" is not a test at all.
COLOUR_CONFUSABLE = {
    "red": ["burgundy", "orange", "pink", "brown"],
    "orange": ["red", "yellow", "brown", "gold"],
    "yellow": ["gold", "orange", "cream", "beige"],
    "green": ["turquoise", "blue", "grey", "navy"],
    "blue": ["navy", "turquoise", "purple", "grey"],
    "purple": ["navy", "pink", "burgundy", "blue"],
    "pink": ["red", "purple", "orange", "beige"],
    "brown": ["beige", "black", "burgundy", "grey"],
    "black": ["navy", "grey", "brown", "white"],
    "white": ["cream", "beige", "grey", "silver"],
    "grey": ["silver", "beige", "black", "navy"],
    "beige": ["cream", "brown", "grey", "white"],
    "gold": ["silver", "yellow", "brown", "bronze"],
    "silver": ["grey", "gold", "white", "black"],
    "turquoise": ["blue", "green", "navy", "grey"],
    "navy": ["black", "blue", "purple", "grey"],
    "cream": ["white", "beige", "yellow", "grey"],
    "burgundy": ["red", "brown", "purple", "black"],
    "multicoloured": ["black", "white", "blue", "red"],
    "transparent": ["white", "black", "grey", "blue"],
    "ivory": ["cream", "beige", "white", "grey"],
    "off-white": ["cream", "grey", "beige", "white"],
    "tan": ["brown", "beige", "cream", "grey"],
    "navy blue": ["black", "blue", "grey", "green"],
    "bronze": ["gold", "brown", "silver", "black"],
    "light blue": ["navy", "turquoise", "grey", "white"],
}

MATERIALS = [
    "wood", "metal", "plastic", "glass", "fabric", "paper", "stone", "ceramic", "leather",
    "rubber", "silicone", "canvas", "steel", "concrete", "foam", "wicker",
]
MATERIAL_CRITERIA = {
    "wood": "visible grain, planks or a natural timber surface",
    "metal": "a hard specular surface, screws, welds or bare steel/aluminium",
    "plastic": "a moulded, uniform, often glossy synthetic surface",
    "glass": "transparent or translucent, with reflections or a visible edge",
    "fabric": "woven or knitted, with folds, weave or stitching",
    "paper": "a printed sheet, corrugation or a folded carton",
    "stone": "mineral, rough or cut, with grain, aggregate or veining",
    "ceramic": "glazed, hard and usually white or coloured clay",
    "leather": "a supple hide with grain, seams or worn edges",
    "rubber": "matte, flexible and usually black, like a tyre or seal",
    "silicone": "soft, slightly rubbery moulded plastic, often matte and flexible",
    "canvas": "heavy plain-woven cloth, usually with a visible coarse weave",
    "steel": "bright or brushed hard metal, often with a machined edge",
    "concrete": "grey cast mineral with a slightly rough, porous face",
    "foam": "soft, porous, compressible padding",
    "wicker": "woven cane, rattan or willow with visible strands",
}
# Same idea: a swapped material must look different in a photograph.
MATERIAL_CONFUSABLE = {
    "wood": ["metal", "plastic", "stone", "ceramic"],
    "engineered wood": ["metal", "glass", "leather", "stone"],
    "metal": ["plastic", "wood", "ceramic", "leather"],
    "stainless steel": ["plastic", "wood", "ceramic", "leather"],
    "steel": ["plastic", "wood", "ceramic", "fabric"],
    "aluminium": ["plastic", "wood", "leather", "ceramic"],
    "iron": ["plastic", "wood", "fabric", "glass"],
    "plastic": ["metal", "wood", "glass", "leather"],
    "silicone": ["leather", "metal", "wood", "glass"],
    "polypropylene": ["metal", "wood", "leather", "glass"],
    "vinyl": ["metal", "wood", "ceramic", "glass"],
    "glass": ["metal", "wood", "ceramic", "plastic"],
    "ceramic": ["metal", "wood", "plastic", "fabric"],
    "stoneware": ["metal", "wood", "plastic", "fabric"],
    "porcelain": ["metal", "wood", "plastic", "fabric"],
    "fabric": ["leather", "metal", "wood", "plastic"],
    "polyester": ["leather", "metal", "wood", "glass"],
    "cotton": ["leather", "metal", "wood", "glass"],
    "wool": ["leather", "metal", "plastic", "glass"],
    "linen": ["leather", "metal", "plastic", "glass"],
    "velvet": ["metal", "wood", "plastic", "glass"],
    "mesh": ["leather", "wood", "ceramic", "glass"],
    "nylon": ["leather", "wood", "ceramic", "glass"],
    "jute": ["metal", "plastic", "glass", "leather"],
    "canvas": ["leather", "metal", "glass", "ceramic"],
    "leather": ["fabric", "metal", "wood", "plastic"],
    "faux leather": ["fabric", "metal", "wood", "glass"],
    "suede": ["metal", "glass", "plastic", "wood"],
    "paper": ["metal", "wood", "leather", "glass"],
    "stone": ["wood", "plastic", "fabric", "glass"],
    "marble": ["wood", "plastic", "fabric", "metal"],
    "concrete": ["wood", "plastic", "fabric", "glass"],
    "rubber": ["metal", "wood", "glass", "ceramic"],
    "foam": ["metal", "wood", "glass", "ceramic"],
    "wicker": ["metal", "glass", "plastic", "fabric"],
    "bamboo": ["metal", "glass", "plastic", "fabric"],
    "acrylic": ["wood", "metal", "fabric", "stone"],
}

PATTERNS = ["solid", "striped", "checked", "floral", "geometric", "polka dot", "plaid",
            "animal print", "paisley", "textured"]
PATTERN_CONFUSABLE = {
    "solid": ["striped", "floral", "geometric", "checked"],
    "striped": ["solid", "checked", "plaid", "geometric"],
    "geometric": ["solid", "floral", "striped", "paisley"],
    "textured": ["solid", "striped", "floral", "checked"],
    "floral": ["solid", "geometric", "striped", "paisley"],
    "checked": ["solid", "striped", "floral", "geometric"],
    "plaid": ["solid", "floral", "polka dot", "geometric"],
    "colour block": ["solid", "floral", "striped", "paisley"],
    "animal print": ["solid", "striped", "geometric", "floral"],
    "polka dot": ["solid", "striped", "plaid", "floral"],
}

SHAPES = ["round", "square", "rectangular", "oval", "triangular", "hexagonal", "heart",
          "cylindrical", "teardrop"]
SHAPE_CONFUSABLE = {
    "rectangular": ["round", "oval", "square", "triangular"],
    "round": ["square", "rectangular", "triangular", "hexagonal"],
    "square": ["round", "oval", "triangular", "hexagonal"],
    "oval": ["square", "rectangular", "triangular", "hexagonal"],
    "drum": ["square", "rectangular", "triangular", "heart"],
    "runner": ["round", "square", "oval", "hexagonal"],
    "teardrop": ["square", "rectangular", "round", "hexagonal"],
    "l-shaped": ["round", "oval", "square", "triangular"],
    "heart": ["square", "rectangular", "round", "triangular"],
    "cylindrical": ["square", "rectangular", "triangular", "heart"],
    "triangular": ["round", "square", "oval", "heart"],
    "hexagonal": ["round", "square", "oval", "heart"],
    "octagonal": ["round", "square", "oval", "heart"],
    "trapezoid": ["round", "oval", "heart", "cylindrical"],
    "u-shaped": ["round", "square", "oval", "triangular"],
}

FINISHES = ["matte", "polished", "brushed", "glossy", "antique", "chrome", "painted", "satin"]
FINISH_CONFUSABLE = {
    "brushed": ["polished", "glossy", "painted", "antique"],
    "matte": ["glossy", "polished", "chrome", "satin"],
    "oil-rubbed": ["polished", "chrome", "glossy", "painted"],
    "polished": ["matte", "brushed", "antique", "painted"],
    "antique": ["polished", "chrome", "glossy", "matte"],
    "satin": ["glossy", "chrome", "antique", "painted"],
    "chrome": ["matte", "painted", "antique", "brushed"],
    "painted": ["polished", "chrome", "brushed", "glossy"],
    "enamelled": ["matte", "brushed", "antique", "painted"],
    "glossy": ["matte", "brushed", "antique", "painted"],
    "distressed": ["polished", "chrome", "glossy", "painted"],
    "natural": ["painted", "chrome", "glossy", "polished"],
    "laminated": ["brushed", "antique", "chrome", "matte"],
    "mirrored": ["matte", "brushed", "painted", "antique"],
}

ATTRIBUTE_POOL = {
    "colour": (COLOURS, COLOUR_CONFUSABLE, COLOUR_CRITERIA),
    "material": (MATERIALS, MATERIAL_CONFUSABLE, MATERIAL_CRITERIA),
    "pattern": (PATTERNS, PATTERN_CONFUSABLE, None),
    "shape": (SHAPES, SHAPE_CONFUSABLE, None),
    "finish": (FINISHES, FINISH_CONFUSABLE, None),
}
ATTRIBUTE_NOUN = {"colour": "colour", "material": "material", "pattern": "pattern",
                  "shape": "shape", "finish": "finish", "product_type": "product type",
                  "count": "quantity", "text_on_label": "text printed on the item",
                  "condition": "condition", "category": "category", "setting": "setting"}

# Conditions a photograph can settle.  Used by `condition_report` and by damage claims.
CONDITIONS = [
    ("a crack across the surface", "a split or fracture line running across a face of the item"),
    ("a deep scratch", "a single long score mark cut into the surface"),
    ("a dent", "a pushed-in area that deforms the outline of the item"),
    ("a chipped edge", "a piece broken away from a corner or rim"),
    ("a torn seam", "stitching that has come apart, leaving an open edge"),
    ("a stain or discolouration", "a patch of a different colour that is not part of the design"),
    ("rust or corrosion", "orange-brown pitting on a metal surface"),
    ("a broken or missing part", "a component that has snapped off or is not there at all"),
    ("water damage", "swelling, staining or warping from having got wet"),
    ("a bent or buckled frame", "a structural part visibly out of straight"),
    ("peeling paint or coating", "the surface finish lifting away in flakes"),
    ("a shattered screen or glass", "glass broken into a web of cracks"),
    ("heavy wear on the surface", "the finish rubbed through from ordinary use"),
    ("a burn mark", "a scorched dark patch with a damaged surface"),
    ("mould growth", "fuzzy dark or greenish patches spreading over the surface"),
]

# Fields no photograph can settle.  A claim built from one of these is honestly unverifiable, and
# the teacher is expected to answer `unknown` -- which is the behaviour the abstention story needs.
UNVERIFIABLE_CLAIM_FORMS = [
    "the item weighs {weight}",
    "the item was manufactured in {year}",
    "the item is model number {model}",
    "the item costs {price}",
    "the item ships from {place}",
    "the item comes with a {years}-year guarantee",
    "the photograph was taken on {date}",
    "the photograph was taken by {person}",
    "the item is covered by batch reference {batch}",
    "the item has been returned once before",
    "the item is the last one left in stock",
    "the item was cleaned the day before the photograph was taken",
    "the seller has held this stock since {year}",
    "the serial number of the item ends in {digits}",
    "the item's packaging is stored in warehouse {place}",
    "the item was inspected by two different people",
]
UNVERIFIABLE_FILL = {
    "weight": ["0.4 kg", "1.2 kg", "860 g", "3.5 kg", "220 g", "12 kg"],
    "year": ["2019", "2020", "2021", "2022", "2023", "2024"],
    "model": ["TX-4410", "QP2288", "AL-77B", "9920-C", "KD1345", "MR-6001"],
    "price": ["£18.99", "£42.00", "£7.50", "£129.00", "£64.95", "£23.40"],
    "place": ["Leeds", "Rotterdam", "Zaragoza", "Gdansk", "Bristol", "Turin"],
    "years": ["1", "2", "3", "5"],
    "date": ["4 March", "11 June", "22 October", "7 January", "30 August"],
    "person": ["the warehouse team", "an external photographer", "the supplier", "a field agent"],
    "batch": ["B-2290", "L4471", "K-88231", "R0092", "C-1174"],
    "digits": ["17", "42", "08", "93", "60"],
}

# Plausible printed strings for the label family.
LABEL_PREFIXES = ["SN", "REF", "LOT", "BATCH", "MOD", "PN", "ID", "CODE"]
LABEL_WORDS = ["CAUTION", "FRAGILE", "EXIT", "STOP", "NO ENTRY", "OPEN", "CLOSED", "RESERVED",
               "PRIVATE", "KEEP CLEAR", "STAFF ONLY", "FIRE EXIT", "WAY OUT", "SALE"]

COUNT_ITEM_FALLBACK = ["items", "units", "pieces"]

COUNT_LEVELS = [
    (0, "the photograph shows fewer than the expected number"),
    (1, "the photograph shows exactly the expected number"),
    (2, "the photograph shows more than the expected number"),
]
COUNT_LEVELS_WIDE = [
    (0, "none at all, although the state expects some"),
    (1, "fewer than the state expects, but not none"),
    (2, "exactly as many as the state expects"),
    (3, "more than the state expects"),
]
COUNT_BANDS = [
    ("none", "not one of them is in the photograph"),
    ("exactly one", "a single one is visible"),
    ("two or three", "two or three are visible"),
    ("four to six", "between four and six are visible"),
    ("more than six", "seven or more are visible"),
]

NONE_OF_THESE = "none of these"
NONE_OF_THESE_CRITERION = "every field listed in the state agrees with the photograph"

CONDITION_OPTIONS = [
    ("the reported condition is visible",
     "the photograph shows the damage the report describes"),
    ("a different condition is visible",
     "the photograph shows damage, but not the damage the report describes"),
    ("no visible damage",
     "the item in the photograph looks intact; nothing is wrong with it"),
]
CONDITION_OPTION_EXTRA = (
    "the photograph does not show the item described",
    "whatever is in the frame, it is not the item the report is about")

# --------------------------------------------------------------------------- paraphrase banks
CLAIM_QUESTIONS = [
    "Does the photograph support the claim recorded at `{p}`?",
    "Compare `{p}` with the picture. Is the claim true of this photograph?",
    "Somebody has written `{p}` about this photo. Is that borne out by the image?",
    "Read `{p}`, then look at the photograph. Does the picture back the claim up?",
    "Is the statement stored at `{p}` consistent with what the photograph shows?",
    "Judging from the image alone, is `{p}` correct?",
    "The state carries an assertion at `{p}`. Does the photograph confirm it?",
    "Answer yes or no: the photograph supports `{p}`.",
    "Check `{p}` against the picture. Does the picture agree?",
    "Using only the photograph, would you say `{p}` is accurate?",
    "`{p}` is on file for this image. Does the image bear it out?",
    "Does what you can see in the photo match the claim at `{p}`?",
]
CLAIM_YES = [
    "the photograph clearly shows what the claim says",
    "the picture bears the claim out",
    "what is visible agrees with the recorded claim",
]
CLAIM_NO = [
    "the photograph shows something that contradicts the claim",
    "the picture disagrees with the recorded claim",
    "what is visible is not what the claim says",
]

CONFLICT_QUESTIONS = [
    "Which field of `{r}` does the photograph contradict?",
    "Compare `{r}` with the picture. Which entry does not match?",
    "One entry in `{r}` may be wrong. Using the photograph, which one?",
    "Check the photograph against `{r}` and name the field that is incorrect.",
    "Which listed field of `{r}` disagrees with the image?",
    "Somebody has to correct `{r}` from the photograph. Which field needs changing?",
    "Looking at the picture, which value in `{r}` is wrong?",
    "The record at `{r}` was filled in from memory. Which field does the photo show is mistaken?",
    "Which single entry of `{r}` is not supported by the photograph?",
    "Read `{r}`, then the photo. Which field is contradicted?",
    "Point to the field of `{r}` that the photograph proves wrong.",
]

COUNT_BOOL_QUESTIONS = [
    "The state expects {n} {item} (`{p}`). Does the photograph show that many?",
    "`{p}` says {n}. Are there exactly {n} {item} in the picture?",
    "Count the {item} in the photo. Does the total match `{p}`?",
    "Does the photograph show the {n} {item} the state records at `{p}`?",
    "Somebody counted {n} {item} and wrote it at `{p}`. Does the photo agree?",
    "Check the number at `{p}` against the picture: is it right?",
    "Answer yes or no: the photograph shows exactly the number of {item} recorded at `{p}`.",
    "Is `{p}` ({n} {item}) borne out by what you can count in the image?",
    "The packing note at `{p}` says {n} {item}. Does the photograph show {n}?",
    "Looking at the photo, are there {n} {item} as `{p}` claims?",
]
COUNT_ORDINAL_QUESTIONS = [
    "`{p}` expects {n} {item}. Does the photograph show fewer, exactly that, or more?",
    "Compare the number of {item} in the picture with the {n} recorded at `{p}`.",
    "The state says {n} {item} (`{p}`). How does the photograph compare?",
    "Count the {item} in the image and place the total against the {n} at `{p}`.",
    "Is the number of {item} in the photo below, equal to, or above `{p}`?",
    "Somebody expected {n} {item}. Judging by the picture, were they short, right, or over?",
    "Against the expected {n} at `{p}`, what does the photograph actually show?",
    "How does the count of {item} in this photograph stand against `{p}`?",
    "Grade the photograph against the {n} {item} recorded at `{p}`.",
    "Place the photo's count of {item} relative to the expected {n} in `{p}`.",
]
COUNT_BAND_QUESTIONS = [
    "Following the rule at `{p}`, how many {item} should be counted in this photograph?",
    "`{p}` says which {item} to count. How many are there?",
    "Apply `{p}` to the picture: how many {item} does it come to?",
    "Read `{p}`, then count. What is the total for this photograph?",
    "How many {item} does the photograph show, once `{p}` has been applied?",
    "Using the counting rule at `{p}`, give the number of {item} in the image.",
    "Somebody has to fill in a count for this photo under `{p}`. What is it?",
    "Count the {item} that `{p}` asks for and pick the matching band.",
    "With `{p}` in force, how many {item} are in the frame?",
    "What number would you record for {item} here, obeying `{p}`?",
]

LABEL_QUESTIONS = [
    "`{p}` gives the text that should be printed on the item. Does the visible text match?",
    "Does the photograph show the text recorded at `{p}`?",
    "Compare the lettering in the picture with `{p}`. Do they agree?",
    "The expected marking is at `{p}`. Is that what the photograph shows?",
    "Somebody recorded the printed text as `{p}`. Does the image bear that out?",
    "Check `{p}` against any writing in the photograph. Is it the same?",
    "Answer yes or no: the text visible in this photograph matches `{p}`.",
    "Is the wording at `{p}` the wording you can actually read in the picture?",
    "Does any label, sign or marking in the photo read as `{p}` says it should?",
    "`{p}` is the expected print. Does the photograph confirm it?",
]
LABEL_YES = [
    "the text in the photograph reads exactly as the state says",
    "the visible lettering matches the expected text",
]
LABEL_NO = [
    "the text in the photograph is different, or there is no such text to read",
    "the visible lettering does not match the expected text",
]

CONDITION_QUESTIONS = [
    "`{p}` describes the damage that was reported. What does the photograph actually show?",
    "Compare the condition report at `{p}` with the picture.",
    "A report at `{p}` says the item is damaged. Judging by the photo, which is it?",
    "Read the damage note at `{p}`, then look at the item. What is the position?",
    "Does the photograph show the damage recorded at `{p}`, something else, or nothing?",
    "Assess the item in the picture against the condition report at `{p}`.",
    "Somebody has claimed the damage at `{p}`. What does the photograph support?",
    "Settle the condition claim at `{p}` from the photograph.",
    "The claim at `{p}` has to be checked against this photograph. What is the outcome?",
    "Look at the item and grade it against the report stored at `{p}`.",
]

PAIR_SUBJECT_QUESTIONS = [
    "Report the {noun} of the thing named at `{p}`, as it appears in this photograph.",
    "`{p}` says which thing to look at. What {noun} is it in the picture?",
    "What {noun} would you record for the item named at `{p}`?",
    "Look only at what `{p}` names. Which {noun} does it have here?",
    "Fill in the {noun} field for the subject given at `{p}`.",
    "For the item at `{p}`, and for nothing else in the frame, what is the {noun}?",
    "Following `{p}`, which {noun} should be recorded from this photograph?",
    "The state names a subject at `{p}`. Give its {noun} from the image.",
    "Judging by the photo, the thing named at `{p}` is which {noun}?",
    "Record the {noun} of whatever `{p}` points to.",
]
PAIR_CATALOGUE_QUESTIONS = [
    "Which of the items listed at `{p}` appears in this photograph?",
    "`{p}` is a checklist. Which entry on it can you see in the picture?",
    "Go through the list at `{p}`. Which one is in the frame?",
    "Which entry of `{p}` does the photograph show?",
    "Somebody has to tick one entry of `{p}` from this photo. Which?",
    "Looking at the image, which item on the list at `{p}` is present?",
    "Pick the entry of `{p}` that the photograph actually contains.",
    "Of the things named at `{p}`, which is visible here?",
    "Search the photograph for the items at `{p}` and say which you find.",
    "Which item from the list at `{p}` turns up in this picture?",
]
CATALOGUE_NONE = "none of them"
CATALOGUE_NONE_CRITERION = "not one of the listed items is anywhere in the photograph"

# --------------------------------------------------------------------------- state shapes
# Each shape is (name, builder).  A builder returns (state, dotted path to the claim/record).
# Filler values are ordinary-looking record metadata, never anything that answers the question.
REFERENCES = ["REQ-{n}", "CASE-{n}", "TCK-{n}", "SUB-{n}", "CHK-{n}", "AUD-{n}"]
PEOPLE = ["a field agent", "the warehouse team", "the seller", "a customer", "the inspector",
          "an automated checker", "the supplier", "the returns desk"]
CHANNELS = ["email", "mobile app", "web form", "phone", "in person", "scanner"]


def _ref(rng):
    return rng.choice(REFERENCES).format(n=rng.randrange(1000, 99999))


def _claim_state(rng, claim, as_string_allowed=True):
    """-> (state, path, shape_name).  Six object shapes plus a free-text one.

    The free-text shape is drawn twice as often as each object shape, so roughly a quarter of the
    states in this source are plain strings rather than JSON, as `docs/decision-v2-pseudolabel-
    spec.md` asks for across decision-v2.  It cannot be the spec's 40% here: four of the six
    families put the claim in a *named field* the question then quotes, and a free-text state has
    no field to quote.
    """
    shape = min(rng.randrange(8 if as_string_allowed else 6), 6)
    if shape == 0:
        return {"claim": claim}, "claim", "claim"
    if shape == 1:
        return ({"submitted_claim": {"text": claim, "submitted_by": rng.choice(PEOPLE)}},
                "submitted_claim.text", "submitted_claim")
    if shape == 2:
        return ({"report": {"reference": _ref(rng), "assertion": claim,
                            "channel": rng.choice(CHANNELS)}},
                "report.assertion", "report")
    if shape == 3:
        return ({"ticket": {"id": _ref(rng), "body": claim, "priority": rng.choice(["P1", "P2", "P3"])}},
                "ticket.body", "ticket")
    if shape == 4:
        return ({"verification": {"statement": claim, "checked": False,
                                  "source": rng.choice(PEOPLE)}},
                "verification.statement", "verification")
    if shape == 5:
        return ({"case": {"reference": _ref(rng), "details": {"claim": claim}}},
                "case.details.claim", "case")
    # free text: the claim is the whole state
    return (f"Claim on file ({_ref(rng)}): {claim}", "the claim in the state", "free_text")


def _record_state(rng, fields, kind, title="", noise_fields=None):
    """A listing / ticket / inspection record with 3-6 fields.

    -> (state, root reference, {field key: dotted path}).  Some shapes are flat, some nested, and
    a couple carry unrelated real-looking metadata so the model cannot treat "every key in the
    state is an option" as a rule.  `noise_fields` are real fields of the record that NO photograph
    can settle (a model number, a weight); they stay in the state and are never options, so
    "a key I cannot check must be the wrong one" is not a shortcut either.
    """
    shape = rng.randrange(6)
    noise = dict(noise_fields or {})
    if rng.random() < 0.45:
        noise.update({"reference": _ref(rng), "recorded_by": rng.choice(PEOPLE)})
        if rng.random() < 0.5:
            noise["channel"] = rng.choice(CHANNELS)
    body = dict(fields)
    if title:
        body["title"] = title
    if shape == 0:
        state = {kind: {**noise, **body}}
        return state, kind, {k: f"{kind}.{k}" for k in fields}
    if shape == 1:
        state = {kind: {**noise, "attributes": body}}
        return state, f"{kind}.attributes", {k: f"{kind}.attributes.{k}" for k in fields}
    if shape == 2:
        state = {"record": {"type": kind, **noise, "values": body}}
        return state, "record.values", {k: f"record.values.{k}" for k in fields}
    if shape == 3:
        state = {**noise, **body}
        return state, "the state", {k: k for k in fields}
    if shape == 4:
        state = {"case": {"reference": _ref(rng), kind: body}}
        return state, f"case.{kind}", {k: f"case.{kind}.{k}" for k in fields}
    state = {kind: body, "meta": noise or {"reference": _ref(rng)}}
    return state, kind, {k: f"{kind}.{k}" for k in fields}


def _count_state(rng, item, quantity):
    shape = min(rng.randrange(7), 5)
    if shape == 0:
        return {"expected": {"item": item, "quantity": quantity}}, "expected.quantity"
    if shape == 1:
        return ({"packing_note": {"reference": _ref(rng), "item": item, "expected_units": quantity}},
                "packing_note.expected_units")
    if shape == 2:
        return {"expected_count": quantity, "counted_thing": item}, "expected_count"
    if shape == 3:
        return ({"stock_check": {"sku": _ref(rng), "description": item, "on_paper": quantity}},
                "stock_check.on_paper")
    if shape == 4:
        return ({"order": {"reference": _ref(rng), "lines": [{"item": item, "quantity": quantity}]}},
                "order.lines")
    return (f"The paperwork for this photograph records {quantity} {item}.",
            "the number in the state")


def _label_state(rng, text):
    shape = min(rng.randrange(7), 5)
    if shape == 0:
        return {"expected_text": text}, "expected_text"
    if shape == 1:
        return ({"label": {"printed": text, "position": rng.choice(
            ["on the front", "on the underside", "on the side panel", "on the tag"])}},
            "label.printed")
    if shape == 2:
        return ({"marking": {"reference": _ref(rng), "should_read": text}}, "marking.should_read")
    if shape == 3:
        return ({"asset": {"tag": text, "registered_by": rng.choice(PEOPLE)}}, "asset.tag")
    if shape == 4:
        return ({"check": {"field": "printed text", "expected": text}}, "check.expected")
    return (f"The item in this photograph should carry the printed text \"{text}\".",
            "the expected text in the state")


def _condition_state(rng, damage, item="the item", when=None):
    """A damage report.  The report names the item it is about, so the question is specific:
    "a deep scratch on the wooden chair", not "a deep scratch" on an unnamed thing."""
    shape = min(rng.randrange(7), 5)
    when = when or rng.choice(["yesterday", "on delivery", "last week", "at the depot",
                               "before dispatch", "on arrival"])
    if shape == 0:
        return {"condition_report": {"item": item, "reported_damage": damage, "reported": when}}, \
            "condition_report.reported_damage"
    if shape == 1:
        return ({"claim": {"reference": _ref(rng), "item": item, "damage": damage,
                           "raised_by": rng.choice(PEOPLE)}}, "claim.damage")
    if shape == 2:
        return ({"inspection": {"subject": item, "finding": damage, "inspected": when,
                                "passed": False}}, "inspection.finding")
    if shape == 3:
        return ({"return": {"reference": _ref(rng), "item": item, "reason": damage}},
                "return.reason")
    if shape == 4:
        return ({"warranty": {"case": _ref(rng), "item": item, "fault_described": damage,
                              "customer_says": "it arrived like this"}},
                "warranty.fault_described")
    return (f"A damage report ({_ref(rng)}) says {item} in this photograph has {damage}.",
            "the damage described in the state")


def _rule_state(rng, instruction, key="rule"):
    shape = rng.randrange(5)
    if shape == 0:
        return {key: {"instruction": instruction}}, f"{key}.instruction"
    if shape == 1:
        return {key: instruction}, key
    if shape == 2:
        return ({"task": {"reference": _ref(rng), "how_to_count": instruction}},
                "task.how_to_count")
    if shape == 3:
        return ({"policy": {"version": rng.randrange(1, 9), "applies_to": "this photograph",
                            "rule": instruction}}, "policy.rule")
    return ({"job": {"id": _ref(rng), "instruction": instruction}}, "job.instruction")


def _subject_state(rng, subject, key="focus"):
    shape = rng.randrange(5)
    if shape == 0:
        return {key: {"subject": subject}}, f"{key}.subject"
    if shape == 1:
        return {"look_at": subject}, "look_at"
    if shape == 2:
        return ({"task": {"reference": _ref(rng), "subject": subject,
                          "note": "answer about this and nothing else in the frame"}},
                "task.subject")
    if shape == 3:
        return ({"row": {"id": _ref(rng), "item_being_catalogued": subject}},
                "row.item_being_catalogued")
    return ({"worksheet": {"line": rng.randrange(1, 40), "describes": subject}},
            "worksheet.describes")


def _catalogue_state(rng, items):
    shape = rng.randrange(4)
    if shape == 0:
        return {"checklist": list(items)}, "checklist"
    if shape == 1:
        return ({"survey": {"reference": _ref(rng), "looking_for": list(items)}},
                "survey.looking_for")
    if shape == 2:
        return ({"inventory": {"expected_items": list(items),
                               "site": rng.choice(["site A", "site B", "the depot", "unit 4"])}},
                "inventory.expected_items")
    return ({"watchlist": {"entries": list(items), "opened": _ref(rng)}}, "watchlist.entries")


# --------------------------------------------------------------------------- helpers
def _pick(rng, bank):
    """-> (index, item); the index becomes part of the template id so wording is auditable."""
    i = rng.randrange(len(bank))
    return i, bank[i]


def _fmt(template, p, **kw):
    """Fill a question template.  A dotted path keeps Jev's backticks (`listing.colour`); a
    free-text state has no path to quote, so the prose phrase replaces the quoted reference."""
    text = template.format(p=p, **kw)
    return text.replace(f"`{p}`", p) if " " in str(p) else text


def _options(values, rng, criteria=None, describe_share=CRITERIA_SHARE, shuffle=True):
    """Contract options, de-duplicated, shuffled, with Jev-style criteria on ~30% of questions."""
    values = list(dict.fromkeys(values))
    if shuffle:
        rng.shuffle(values)
    with_desc = criteria is not None and rng.random() < describe_share
    out = []
    for value in values:
        option = {"value": str(value)[:128]}
        if with_desc and criteria.get(value):
            option["description"] = criteria[value][:2000]
        out.append(option)
    return out


def _levels(levels):
    return [{"value": value, "description": text} for value, text in levels]


def _field(kind, question, **rest):
    return {"id": "answer", "type": kind, "question": question[:2000], **rest}


def _maybe_bool_criteria(field, rng, yes_bank, no_bank):
    if rng.random() < CRITERIA_SHARE:
        field["yes_description"] = rng.choice(yes_bank)
        field["no_description"] = rng.choice(no_bank)
    return field


def _fill(rng, pool, taken, count):
    available = [x for x in pool if x not in taken]
    rng.shuffle(available)
    return available[:count]


def swap_value(kind, value, rng, pool=None):
    """A clearly different value of the same attribute -- plausible, never nonsense.

    -> the altered value, or None when nothing sensible is available.
    """
    values, confusable, _ = ATTRIBUTE_POOL[kind]
    options = [v for v in confusable.get(str(value).lower(), []) if v != value]
    if not options:
        options = [v for v in (pool or values) if v != value]
    return rng.choice(options) if options else None


def swap_subject(subject, rng, peers=(), absent=()):
    """A different but plausible subject: a verified-absent thing first (really wrong, really
    specific), otherwise a peer product type from the same catalogue group."""
    pool = [x for x in absent if x != subject]
    if pool and (not peers or rng.random() < 0.6):
        return rng.choice(pool)
    pool = [x for x in peers if x != subject]
    return rng.choice(pool) if pool else None


# Upstream labels that read as mass nouns: "a clothing" and "a food" are wrong.  Same list as
# `scripts/v2/templates/photo.py:MASS_LIKE`, which meets the same Open Images labels.
MASS_LIKE = {
    "food", "water", "hair", "fur", "skin", "wood", "metal", "plastic", "glass", "paper", "snow",
    "ice", "grass", "sand", "soil", "milk", "meat", "cheese", "bread", "rice", "pasta", "sugar",
    "salt", "coffee", "tea", "beer", "wine", "juice", "smoke", "fire", "dust", "clothing",
    "furniture", "packaging", "money", "cash", "jewellery", "jewelry", "hardware", "equipment",
    "machinery", "traffic", "vegetation", "foliage", "produce", "seafood", "fast food",
    "junk food", "street food", "baked goods", "dairy", "cutlery", "crockery", "luggage",
    "footwear", "drink", "drinkware", "tableware", "stationery", "bedding", "outerwear",
    "sportswear", "lighting", "flooring", "textile", "fabric", "leather", "fruit", "vegetable",
    "produce", "seating", "storage", "wildlife", "livestock", "poultry", "jewellery",
}
# Mass nouns that take "some X are visible" rather than a number: never counted.
UNCOUNTABLE = MASS_LIKE

# Living things and people.  A damage report about them is nonsense, and so is a material
# question, so both families steer round these.
ANIMATE = {
    "animal", "person", "people", "man", "woman", "girl", "boy", "child", "baby", "human",
    "human body", "human face", "human hand", "human head", "dog", "cat", "bird", "horse", "cow",
    "sheep", "pig", "goat", "fish", "insect", "butterfly", "bee", "spider", "duck", "chicken",
    "plant", "tree", "flower", "grass", "leaf", "mushroom", "fungus", "wildlife", "livestock",
    "crowd", "family", "bride", "athlete", "player", "worker", "soldier", "toddler",
}
# Scenes and places: "a dent on a supermarket" is not a condition report.  A photograph filed
# under one of these has no single item a damage claim could be about.
SCENE_LIKE = {
    "street_scene", "kitchen", "supermarket", "restaurant_interior", "living_room", "bedroom",
    "bathroom", "office_interior", "construction_site", "laboratory", "storefront", "harbour",
    "market_stall", "garden_or_park", "snow_scene", "mountain", "coast_or_beach", "river_or_lake",
    "forest", "landscape", "people_group", "railway", "industrial", "sport", "city", "airport",
}


def is_item_like(phrase) -> bool:
    """True when a damage report or a material question about this thing makes sense."""
    word = str(phrase).strip().lower()
    for article in ("a pair of ", "an ", "a ", "the "):
        if word.startswith(article):
            word = word[len(article):]
            break
    if not word:
        return False
    return not (word in ANIMATE or word.replace(" ", "_") in SCENE_LIKE
                or word.split()[-1] in ANIMATE)


def _a(text):
    """Leave a phrase that already carries its determiner alone; otherwise add one.

    A mass noun ("clothing", "food") takes no article at all: "a clothing" is the sort of phrase
    that tells a model the question was machine-made rather than asked by a person.
    """
    text = str(text).strip()
    if not text:
        return "the item"
    lowered = text.lower()
    first = lowered.split()[0]
    if first in {"a", "an", "the", "some", "two", "three", "four", "five", "people"}:
        return text
    if lowered in MASS_LIKE or lowered.split()[-1] in MASS_LIKE:
        return text
    if text.endswith("s") and not text.endswith(("ss", "us", "is")):
        return text
    return ("an " if text[0].lower() in "aeiou" else "a ") + text


def is_countable(phrase) -> bool:
    """False for a mass noun: nothing may ask 'how many clothings'."""
    word = str(phrase).strip().lower()
    for article in ("a pair of ", "an ", "a ", "the "):
        if word.startswith(article):
            word = word[len(article):]
            break
    return not (word in UNCOUNTABLE or (word.split() and word.split()[-1] in UNCOUNTABLE))


def _plural(phrase):
    word = str(phrase)
    for article in ("a pair of ", "an ", "a ", "the "):
        if word.startswith(article):
            word = word[len(article):]
            break
    if not is_countable(word):
        return word
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _label_text(rng, kind=None):
    """An expected printed string.

    `code` is a serial-style reference: a real photograph almost never carries that exact string,
    so the honest answer is "no" or "there is no readable text here".  `word` is ordinary signage
    wording, which a shop front, a road sign or a packet really might show -- so the family is not
    a family of no's, and the teacher has something to settle.
    """
    kind = kind or ("code" if rng.random() < 0.55 else "word")
    if kind == "code":
        return f"{rng.choice(LABEL_PREFIXES)}-{rng.randrange(1000, 99999)}"
    word = rng.choice(LABEL_WORDS)
    return word if rng.random() < 0.6 else f"{word} {rng.randrange(1, 40)}"


def _unverifiable_claim(rng):
    form = rng.choice(UNVERIFIABLE_CLAIM_FORMS)
    fill = {k: rng.choice(v) for k, v in UNVERIFIABLE_FILL.items()}
    return form.format(**fill)


# --------------------------------------------------------------------------- claim building
CLAIM_KINDS = ("object", "attribute", "count", "text", "condition", "category")

OBJECT_CLAIM_FORMS = [
    "the photograph shows {thing}",
    "there is {thing} in the picture",
    "{thing} is visible in this photograph",
    "this photo contains {thing}",
    "{thing} appears somewhere in the frame",
    "you can see {thing} in this image",
    "the image includes {thing}",
    "{thing} is in shot",
]
ATTRIBUTE_CLAIM_FORMS = [
    "the {noun} of {thing} is {value}",
    "{thing} in this photograph is {value}",
    "{thing} shown here has a {value} {noun}",
    "the {thing} pictured is {value}",
    "{thing} is recorded as {value}",
    "this is {value} {thing}",
    "the item's {noun} is {value}",
    "{thing} in the image is {value} in {noun}",
]
COUNT_CLAIM_FORMS = [
    "there are {n} {things} in the photograph",
    "the photograph shows {n} {things}",
    "{n} {things} are visible in this image",
    "you can count {n} {things} here",
    "this picture contains {n} {things}",
    "the frame holds {n} {things}",
    "{n} separate {things} appear in the photo",
    "the count of {things} in this image is {n}",
]
TEXT_CLAIM_FORMS = [
    "the text printed on the item reads \"{text}\"",
    "\"{text}\" is printed somewhere in this photograph",
    "the label in the picture says \"{text}\"",
    "the marking visible here reads \"{text}\"",
    "the item carries the printing \"{text}\"",
    "there is a sign in the photograph reading \"{text}\"",
    "the wording shown in the image is \"{text}\"",
    "the visible lettering spells \"{text}\"",
]
CONDITION_CLAIM_FORMS = [
    "{thing} in the photograph has {damage}",
    "{damage} can be seen on {thing} in this picture",
    "the picture shows {damage} on {thing}",
    "{thing} shown here is damaged: {damage}",
    "there is {damage} on {thing} in shot",
    "the photograph records {damage} on {thing}",
    "{thing} arrived with {damage}",
    "{damage} is visible on {thing} in this image",
]
CATEGORY_CLAIM_FORMS = [
    "the product in the photograph is {thing}",
    "what is being sold here is {thing}",
    "this listing's photo shows {thing}",
    "the item pictured is {thing}",
    "the photograph is of {thing}",
    "the goods in this image are {thing}",
    "the product type shown is {thing}",
    "this is a photograph of {thing}",
]


def build_claim(inputs, rng, intent, kind=None):
    """-> (claim text, basis dict) -- a specific, checkable assertion about the photograph.

    `intent` is what the converter *wants* the claim to be ("true", "false", "unverifiable").  It
    is an intent, not a label: the teacher settles every record, and a claim built as "true" from
    a PD12M caption can still turn out false.  `basis` records exactly what was used and what was
    altered, so the source README and the tests can audit the mix.
    """
    attrs = dict(inputs.get("attrs") or {})
    present = [p for p in (inputs.get("present") or []) if p]
    absent = [a for a in (inputs.get("absent") or []) if a]
    subject = inputs.get("subject") or (present[0] if present else "the main subject")
    peers = list(inputs.get("peers") or [])

    if intent == "unverifiable":
        hidden = inputs.get("hidden") or {}
        if hidden and rng.random() < 0.45:
            key = rng.choice(sorted(hidden))
            nouns = {"model_number": "model number", "item_weight": "weight",
                     "item_dimensions": "measurements", "model_year": "model year",
                     "fabric_type": "fibre composition"}
            claim = f"the item's {nouns.get(key, key)} is {hidden[key]}"
            return claim, {"kind": "hidden_field", "field": key, "value": str(hidden[key])}
        return _unverifiable_claim(rng), {"kind": "unverifiable_form"}

    wants_true = intent == "true"
    choices = []
    if present or absent:
        choices.append("object")
    if attrs:
        choices.append("attribute")
        if "product_type" in attrs:
            choices.append("category")
    if present or absent:
        choices.append("count")
    choices.append("text")
    if condition_subject(inputs) is not None:
        choices.append("condition")
    # Attribute claims are what the 2B is worst at (it caught 3 of 12 altered ABO materials), so
    # a photo with structured attributes gets attribute claims three times as often as any other
    # kind, and category claims twice as often.
    weights = [3.0 if c == "attribute" else 2.0 if c == "category" else 1.0 for c in choices]
    kind = kind or rng.choices(choices, weights=weights)[0]
    if kind == "condition" and condition_subject(inputs) is None:
        kind = "object" if (present or absent) else "text"

    if kind == "object":
        if wants_true and present:
            thing = rng.choice(present)
            basis = {"kind": "object", "thing": thing, "from": "present"}
        elif not wants_true and absent:
            thing = rng.choice(absent)
            basis = {"kind": "object", "thing": thing, "from": "absent"}
        elif present:
            thing = rng.choice(present)
            basis = {"kind": "object", "thing": thing, "from": "present"}
        elif absent:
            thing = rng.choice(absent)
            basis = {"kind": "object", "thing": thing, "from": "absent"}
        else:
            thing = subject
            basis = {"kind": "object", "thing": thing, "from": "subject"}
        return rng.choice(OBJECT_CLAIM_FORMS).format(thing=_a(thing)), basis

    if kind == "category":
        gold = attrs.get("product_type") or subject
        if wants_true:
            value, altered = gold, None
        else:
            altered = swap_subject(gold, rng, peers=peers, absent=absent)
            value = altered or gold
        return rng.choice(CATEGORY_CLAIM_FORMS).format(thing=_a(value)), \
            {"kind": "category", "gold": gold, "stated": value, "altered": altered is not None}

    if kind == "attribute":
        keys = [k for k in attrs if k in ATTRIBUTE_POOL] or ["colour"]
        key = rng.choices(keys, weights=[2.0 if k == "material" else 1.0 for k in keys])[0]
        gold = attrs.get(key, "black")
        if wants_true:
            value, altered = gold, False
        else:
            swapped = swap_value(key, gold, rng)
            value, altered = (swapped or gold), swapped is not None
        thing = _a(attrs.get("product_type") or subject)
        return rng.choice(ATTRIBUTE_CLAIM_FORMS).format(
            noun=ATTRIBUTE_NOUN[key], thing=thing, value=value), \
            {"kind": "attribute", "field": key, "gold": gold, "stated": value, "altered": altered}

    if kind == "count":
        countable_present = [p for p in present if is_countable(p)]
        countable_absent = [a for a in absent if is_countable(a)]
        if wants_true and countable_present:
            thing, n = rng.choice(countable_present), 1
            basis = {"kind": "count", "thing": thing, "stated": n, "from": "present_one"}
        elif not wants_true and countable_absent:
            thing, n = rng.choice(countable_absent), rng.randint(1, 4)
            basis = {"kind": "count", "thing": thing, "stated": n, "from": "absent"}
        elif countable_present or countable_absent:
            thing = (countable_present or countable_absent)[0]
            n = rng.randint(2, 6)
            basis = {"kind": "count", "thing": thing, "stated": n, "from": "guess"}
        else:   # nothing countable in this photo: fall back to an object claim
            return build_claim(inputs, rng, intent, kind="object")
        return rng.choice(COUNT_CLAIM_FORMS).format(n=n, things=_plural(thing)), basis

    if kind == "text":
        text = _label_text(rng)
        return rng.choice(TEXT_CLAIM_FORMS).format(text=text), \
            {"kind": "text", "stated": text}

    damage, _ = rng.choice(CONDITIONS)
    thing = condition_subject(inputs) or _a(subject)
    return rng.choice(CONDITION_CLAIM_FORMS).format(damage=damage, thing=thing), \
        {"kind": "condition", "stated": damage, "thing": thing}


# --------------------------------------------------------------------------- families
def make_claim_supported(inputs, rng, intent="true", kind=None):
    """boolean.  The state carries one assertion about the photograph; is it borne out?

    `kind` pins the sort of claim (object / attribute / count / text / condition / category); the
    probe builder uses it so that a claim's truth follows from verified metadata alone.
    """
    claim, basis = build_claim(inputs, rng, intent, kind=kind)
    state, path, shape = _claim_state(rng, claim)
    i, template = _pick(rng, CLAIM_QUESTIONS)
    field = _maybe_bool_criteria(_field("boolean", _fmt(template, path)), rng,
                                 CLAIM_YES, CLAIM_NO)
    return {
        "state": state, "fields": [field],
        "template_id": f"claim_supported/{i:02d}/{shape}/{basis['kind']}/{intent}",
        "intent": intent, "state_kind": shape, "basis": basis, "state_path": path,
    }


CONDITION_OK = ["as new, no damage", "unused, no visible damage", "undamaged",
                "no faults recorded", "intact"]
SIGHTING_KEYS = ["main_subject", "also_visible", "also_present", "third_item", "fourth_item"]


def make_which_field_conflicts(inputs, rng, intent="one_conflict"):
    """choice.  A listing / ticket / report with 3-6 photo-checkable fields, exactly one of which
    contradicts the photograph -- or none, in which case `none of these` is the honest answer.

    Every field's value is a ground fact of the source (an ABO structured attribute, or a label a
    human verified is in the photograph), so `none of these` really is right when nothing was
    altered.  A source that cannot supply enough ground facts returns None and is skipped -- the
    converter never invents a value it could not stand behind.
    """
    attrs = dict(inputs.get("attrs") or {})
    present = [p for p in (inputs.get("present") or []) if p]
    absent = [a for a in (inputs.get("absent") or []) if a]
    peers = list(inputs.get("peers") or [])

    fields: dict[str, str] = {}
    checkable: list[str] = []
    if attrs:
        # catalogue-listing shaped: the real structured attributes of the listing
        for key in ("product_type", "colour", "material", "pattern", "shape", "finish"):
            if key in attrs and len(fields) < 6:
                fields[key] = str(attrs[key])
                checkable.append(key)
        if len(checkable) < 4 and inputs.get("condition_is_new"):
            fields["condition"] = rng.choice(CONDITION_OK)
            checkable.append("condition")
        kind = "listing"
    else:
        # sighting/inspection shaped: several things a human verified are in this photograph
        if len(present) < 3:
            return None
        names = present[:rng.randint(3, min(5, len(present)))]
        for key, name in zip(SIGHTING_KEYS, names):
            fields[key] = _a(name)
            checkable.append(key)
        kind = rng.choice(["report", "sighting", "inspection", "ticket"])
    if len(checkable) < 2:
        return None

    swapped_key, swapped_from = None, None
    if intent == "one_conflict":
        # weighted shuffle: material and colour are swapped most often, because those are the
        # fields the 2B misses (material 3/12, and a fair share of colour) -- product type it
        # already catches (21/24)
        weight = {"material": 4.0, "colour": 2.5, "pattern": 1.5, "shape": 1.5, "finish": 1.5}
        order = sorted(checkable, key=lambda k: -rng.random() ** (1.0 / weight.get(k, 1.0)))
        for key in order:
            if key in ATTRIBUTE_POOL:
                new = swap_value(key, fields[key], rng)
            elif key == "product_type":
                new = swap_subject(fields[key], rng, peers=peers, absent=absent)
            elif key in SIGHTING_KEYS:
                taken = {v for v in fields.values()}
                pool = [x for x in absent if _a(x) not in taken]
                new = _a(rng.choice(pool)) if pool else None
            elif key == "condition":
                new = rng.choice(CONDITIONS)[0]
            else:
                new = None
            if new and str(new) != fields[key]:
                swapped_key, swapped_from = key, fields[key]
                fields[key] = str(new)
                break
        if swapped_key is None:
            intent = "none"

    title = inputs.get("title") or ""
    noise = dict(inputs.get("hidden") or {}) if rng.random() < 0.40 else {}
    state, root, paths = _record_state(rng, fields, kind, title=title, noise_fields=noise)

    # 2-7 options: the listed photo-checkable fields (never the noise fields) plus `none of these`
    # as an ordinary option, exactly as decision-v1's `contradicted field` form does.
    values = [paths[k] for k in checkable if k in paths]
    values.append(NONE_OF_THESE)
    criteria = {paths[k]: f"the photograph disagrees with the {ATTRIBUTE_NOUN.get(k, k.replace('_', ' '))} recorded there"
                for k in checkable if k in paths}
    criteria[NONE_OF_THESE] = NONE_OF_THESE_CRITERION
    i, template = _pick(rng, CONFLICT_QUESTIONS)
    field = _field("choice", _fmt(template.replace("{r}", "{p}"), root),
                   options=_options(values, rng, criteria))
    return {
        "state": state, "fields": [field],
        "template_id": f"which_field_conflicts/{i:02d}/{kind}/{intent}",
        "intent": intent, "state_kind": kind,
        "basis": {"kind": "record", "fields": sorted(fields), "swapped": swapped_key,
                  "swapped_from": swapped_from,
                  "swapped_to": fields.get(swapped_key) if swapped_key else None,
                  # the option value that names the altered field -- what a correct answer is
                  "swapped_path": paths.get(swapped_key) if swapped_key else NONE_OF_THESE,
                  "paths": dict(paths)},
        "state_path": root,
    }


def make_count_matches(inputs, rng, intent="match", form=None):
    """boolean or ordinal.  The state records an expected number; the photograph is the check."""
    present = [p for p in (inputs.get("present") or []) if p and is_countable(p)]
    absent = [a for a in (inputs.get("absent") or []) if a and is_countable(a)]
    if not (present or absent or is_countable(inputs.get("subject") or "")):
        return None       # nothing in this photo that a person would count
    if intent == "fewer" and absent:
        thing, n = rng.choice(absent), rng.randint(1, 5)
        basis_from = "absent"
    elif intent == "more" and present:
        thing, n = rng.choice(present), 0
        basis_from = "present_zero"
    elif present:
        thing, n = rng.choice(present), (1 if intent == "match" else rng.randint(1, 5))
        basis_from = "present"
    elif absent:
        thing, n = rng.choice(absent), rng.randint(1, 5)
        basis_from = "absent"
    else:
        thing, n = inputs.get("subject") or "items", rng.randint(1, 4)
        basis_from = "subject"
    item = _plural(thing)
    state, path = _count_state(rng, item, n)
    form = form or ("ordinal" if rng.random() < 0.45 else "boolean")
    if form == "boolean":
        i, template = _pick(rng, COUNT_BOOL_QUESTIONS)
        field = _maybe_bool_criteria(
            _field("boolean", _fmt(template, path, n=n, item=item)), rng,
            [f"the photograph shows exactly {n} {item}"],
            [f"the photograph shows some number of {item} other than {n}"])
    else:
        i, template = _pick(rng, COUNT_ORDINAL_QUESTIONS)
        levels = COUNT_LEVELS if rng.random() < 0.7 else COUNT_LEVELS_WIDE
        field = _field("ordinal", _fmt(template, path, n=n, item=item), levels=_levels(levels))
    return {
        "state": state, "fields": [field],
        "template_id": f"count_matches/{i:02d}/{form}/{intent}",
        "intent": intent, "state_kind": "count",
        "basis": {"kind": "count", "thing": thing, "expected": n, "from": basis_from},
        "state_path": path,
    }


def make_label_text_matches(inputs, rng, intent="code"):
    """boolean.  The state gives the text that should be printed on the item.

    No source here carries OCR, so the expected text is authored rather than read off the photo:
    `word` states ordinary signage wording a photograph plausibly shows, `code` states a
    serial-style reference it almost certainly does not.  Which it is, is recorded in the record
    and in the source README; the teacher answers every one of them from the pixels.
    """
    text = _label_text(rng, kind=intent if intent in ("word", "code") else None)
    state, path = _label_state(rng, text)
    i, template = _pick(rng, LABEL_QUESTIONS)
    field = _maybe_bool_criteria(_field("boolean", _fmt(template, path)), rng,
                                 LABEL_YES, LABEL_NO)
    return {
        "state": state, "fields": [field],
        "template_id": f"label_text_matches/{i:02d}/{intent}",
        "intent": intent, "state_kind": "label",
        "basis": {"kind": "text", "stated": text, "text_kind": intent},
        "state_path": path,
    }


def condition_subject(inputs):
    """-> the phrase a damage report can name, or None when this photo has no such item."""
    attrs = inputs.get("attrs") or {}
    if attrs.get("product_type"):
        return _a(attrs["product_type"])
    for candidate in (inputs.get("present") or []):
        if is_item_like(candidate):
            return _a(candidate)
    subject = inputs.get("subject") or ""
    return _a(subject) if subject and is_item_like(subject) else None


def make_condition_report(inputs, rng, intent="reported"):
    """choice.  A damage report in the state; does the photo show that, something else, or
    nothing wrong?"""
    damage, criterion = rng.choice(CONDITIONS)
    item = condition_subject(inputs)
    if item is None:
        return None          # a scene or a living thing: no item for a damage report to be about
    state, path = _condition_state(rng, damage, item=item)
    pairs = list(CONDITION_OPTIONS)
    if rng.random() < 0.35:
        pairs.append(CONDITION_OPTION_EXTRA)
    criteria = {value: text for value, text in pairs}
    criteria[CONDITION_OPTIONS[0][0]] = f"the photograph shows {damage}"
    i, template = _pick(rng, CONDITION_QUESTIONS)
    field = _field("choice", _fmt(template, path),
                   options=_options([v for v, _ in pairs], rng, criteria))
    return {
        "state": state, "fields": [field],
        "template_id": f"condition_report/{i:02d}/{len(pairs)}/{intent}",
        "intent": intent, "state_kind": "condition",
        "basis": {"kind": "condition", "stated": damage, "criterion": criterion},
        "state_path": path,
    }


# ------------------------------------------------- instruction_changes_answer (paired records)
PAIR_CONSTRUCTIONS = ("subject_switch", "count_scope", "catalogue_switch")


def make_instruction_pair(inputs, rng, construction=None):
    """-> [record_a, record_b]: the SAME photograph with two states that legitimately change the
    right answer.  Both members share the wording of the question; only the state differs, so a
    model that ignores the state must get one of the two wrong.
    """
    present = [p for p in (inputs.get("present") or []) if p]
    absent = [a for a in (inputs.get("absent") or []) if a]
    attrs = dict(inputs.get("attrs") or {})

    usable = []
    if len(present) >= 2 or (present and absent):
        usable.append("subject_switch")
    if any(is_countable(p) for p in present):
        usable.append("count_scope")
    if present and len(absent) >= 2:
        usable.append("catalogue_switch")
    if not usable:
        return None
    construction = construction if construction in usable else rng.choice(usable)

    if construction == "subject_switch":
        pool = present + absent
        first = present[0]
        second = next((x for x in pool[1:] if x != first), None)
        if second is None:
            return None
        noun = "material" if (rng.random() < 0.35 and is_item_like(first)
                              and is_item_like(second)) else "colour"
        values, _, criteria = ATTRIBUTE_POOL[noun]
        chosen = _fill(rng, values, set(), rng.randint(4, 8))
        i, template = _pick(rng, PAIR_SUBJECT_QUESTIONS)
        out = []
        for role, thing in (("a", first), ("b", second)):
            state, path = _subject_state(rng, _a(thing))
            field = _field("choice", _fmt(template, path, noun=noun),
                           options=_options(chosen, rng, criteria))
            out.append({
                "state": state, "fields": [field],
                "template_id": f"instruction_changes_answer/{i:02d}/subject_switch/{noun}/{role}",
                "intent": "pair", "state_kind": "focus",
                "basis": {"kind": "subject_switch", "subject": thing, "attribute": noun,
                          "role": role},
                "state_path": path, "pair_role": role,
            })
        return out

    if construction == "count_scope":
        thing = rng.choice([p for p in present if is_countable(p)])
        colour = rng.choice(COLOURS)
        item = _plural(thing)
        i, template = _pick(rng, COUNT_BAND_QUESTIONS)
        criteria = {value: text for value, text in COUNT_BANDS}
        rules = [("a", f"count only the {colour} {item}"),
                 ("b", f"count every one of the {item}, whatever colour it is")]
        out = []
        for role, instruction in rules:
            state, path = _rule_state(rng, instruction)
            field = _field("choice", _fmt(template, path, item=item),
                           options=_options([v for v, _ in COUNT_BANDS], rng, criteria))
            out.append({
                "state": state, "fields": [field],
                "template_id": f"instruction_changes_answer/{i:02d}/count_scope/{role}",
                "intent": "pair", "state_kind": "rule",
                "basis": {"kind": "count_scope", "thing": thing, "colour": colour,
                          "instruction": instruction, "role": role},
                "state_path": path, "pair_role": role,
            })
        return out

    # catalogue_switch: list A contains something that is in the photo, list B does not
    hit = rng.choice(present)
    misses = [x for x in absent if x != hit]
    rng.shuffle(misses)
    n = rng.randint(2, min(4, len(misses)))
    list_a = [_a(hit)] + [_a(x) for x in misses[:n]]
    list_b = [_a(x) for x in misses[:n + 1]]
    if len(list_b) < 2:
        return None
    rng.shuffle(list_a)
    rng.shuffle(list_b)
    i, template = _pick(rng, PAIR_CATALOGUE_QUESTIONS)
    out = []
    for role, entries in (("a", list_a), ("b", list_b)):
        state, path = _catalogue_state(rng, entries)
        values = list(entries) + [CATALOGUE_NONE]
        criteria = {v: f"{v} is the item you can see in the photograph" for v in entries}
        criteria[CATALOGUE_NONE] = CATALOGUE_NONE_CRITERION
        field = _field("choice", _fmt(template, path), options=_options(values, rng, criteria))
        out.append({
            "state": state, "fields": [field],
            "template_id": f"instruction_changes_answer/{i:02d}/catalogue_switch/{role}",
            "intent": "pair", "state_kind": "catalogue",
            "basis": {"kind": "catalogue_switch", "hit": hit, "entries": entries, "role": role},
            "state_path": path, "pair_role": role,
        })
    return out


MAKERS = {
    "claim_supported": make_claim_supported,
    "which_field_conflicts": make_which_field_conflicts,
    "count_matches": make_count_matches,
    "label_text_matches": make_label_text_matches,
    "condition_report": make_condition_report,
}

# What the converter is allowed to aim for per family.  These are intents, not labels: the teacher
# decides every answer, and `claim_supported` is deliberately aimed at roughly 35/35/30.
INTENTS = {
    "claim_supported": ("true", "false", "unverifiable"),
    "which_field_conflicts": ("one_conflict", "none"),
    "count_matches": ("match", "fewer", "more"),
    "label_text_matches": ("word", "code"),
    # One construction: the report names a specific, visible kind of damage and the photograph
    # settles it.  The three ANSWERS (reported / different / none) are the option set, not intents.
    "condition_report": ("reported",),
}


def make(family, inputs, rng, intent=None, **kw):
    """The single entry point the converter uses for the five single-record families."""
    maker = MAKERS[family]
    if intent is None:
        intent = rng.choice(INTENTS[family])
    return maker(inputs, rng, intent=intent, **kw)


# --------------------------------------------------------------------------- leak guard
def state_text(state) -> str:
    """Every string a state carries, flattened -- what the leak checks read."""
    if isinstance(state, str):
        return state
    out = []

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                out.append(str(k))
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        else:
            out.append(str(value))

    walk(state)
    return " ".join(out)


def shares_long_ngram(a: str, b: str, n: int = 6) -> bool:
    """True when two texts share a run of `n` words -- the caption-leak check.

    `a` is a state, `b` a source caption or title.  Six words is long enough that ordinary
    vocabulary overlap ("the item in the photograph") does not trip it, and short enough that a
    copied caption clause cannot slip through.
    """
    def grams(text):
        words = [w for w in "".join(c.lower() if c.isalnum() else " " for c in str(text)).split()]
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    return bool(grams(a) & grams(b))
