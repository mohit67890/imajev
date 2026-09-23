"""Read a PD12M synthetic caption and decide three things, with no network and no model.

PD12M ships one machine-written caption per image ("The image shows a yellow and black wasp
sitting on top of a white cup. ...").  That caption is the only description we have, and it is
used for exactly three jobs, all of them about *choosing and shaping the question*:

    reject(caption)   -> why this image is not a photograph we want (artwork, scan, engraving,
                         logo, map, ...), or None to keep it.
    bucket(caption)   -> one of BUCKETS, the coarse subject used to balance the selection and to
                         seed plausible option sets (`scripts/v2/templates/photo.py`).
    objects(caption)  -> ([things the caption claims are in the frame],
                          [things that are concrete, plausible in general, and NOT claimed here]).

The caption is NEVER written into a record's state and NEVER becomes a target.  The teacher
answers from the pixels; a caption that is wrong only costs us a question the teacher will settle
differently.  `scripts/v2/convert_photos.py` passes `title=""` for this source precisely so that no
caption text can leak into the state through `photo.make_state`.

Both `present` and `absent` are *claims*, not verified labels:
  * present  -- the caption said so.  Machine-written, so it can be wrong.
  * absent   -- the caption did not say so, the thing is not typical of the bucket, and it is not
                in `OFTEN_UNSAID` (things photographs contain without a caption mentioning them).
                This is weaker than Open Images' human-verified `Confidence == 0`, and the source
                README says so.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- rejection
# Captions whose subject is a picture of a picture, a scan, or a graphic.  We want photographs.
REJECT = [
    ("artwork", r"\b(painting|painted portrait|oil on canvas|watercolou?r|drawing|drawn|sketch|"
                r"sketched|etching|engraving|engraved|lithograph|woodcut|linocut|print of|"
                r"illustration|illustrated|cartoon|comic strip|caricature|fresco|mural|"
                r"stained glass window depicting|tapestry depicting|icon depicting)\b"),
    ("scan_or_page", r"\b(manuscript|handwritten|handwriting|calligraphy|title page|book page|"
                     r"page of a book|newspaper page|scanned|scan of|sheet music|musical score|"
                     r"ledger|certificate|diploma|postcard|envelope|letterhead)\b"),
    ("graphic", r"\b(logo|emblem|coat of arms|heraldic|crest of|monogram|trademark|"
                r"clip ?art|vector graphic|infographic|diagram|schematic|blueprint|floor plan|"
                r"screenshot|3d render|computer[- ]generated)\b"),
    ("map", r"\b(map of|a map|nautical chart|atlas|cartograph)\w*\b"),
    ("numismatic", r"\b(coin|medal|medallion|banknote|bank note|postage stamp|wax seal|"
                   r"commemorative plaque bearing)\b"),
    ("framed_work", r"\b(framed in a (photo|picture) frame|in an ornate frame|picture frame "
                    r"containing|on display in a museum case)\b"),
    ("poster", r"\b(poster|advertisement|advertising sign for|magazine cover|book cover|"
               r"album cover|flyer|leaflet)\b"),
    ("no_subject", r"^\s*the image (shows )?(is )?(a )?(blank|empty|solid colou?r|plain)\b"),
]
_REJECT = [(name, re.compile(pattern, re.I)) for name, pattern in REJECT]


def reject(caption: str) -> str | None:
    """-> the reason this caption does not look like a photograph, or None to keep it."""
    text = str(caption or "")
    if len(text) < 40:
        return "caption_too_short"
    for name, pattern in _REJECT:
        if pattern.search(text):
            return name
    return None


# --------------------------------------------------------------------------- subject buckets
# The rule whose keyword appears EARLIEST in the caption wins (see `bucket` below); the order
# here is the tie-break, most specific first, for two keywords at the same position.  The names
# are the seed categories `scripts/v2/templates/photo.py` already knows (CATEGORY_OBJECT /
# CATEGORY_SCENES / CATEGORY_ASSOCIATED), so pd12m photos flow through the same templates, and
# the same option banks, as the other two image sources.
BUCKET_RULES: list[tuple[str, str]] = [
    # -- specialised subjects
    ("medical_device", r"\b(stethoscope|syringe|scalpel|wheelchair|hospital bed|x-?ray|"
                       r"dental chair|prosthe\w+|crutch\w*|surgical instrument)\b"),
    ("laboratory", r"\b(laborator\w+|test tube|beaker|petri dish|microscope|pipette|"
                   r"bunsen burner|specimen jar)\b"),
    ("musical_instrument", r"\b(guitar|piano|violin|cello|trumpet|trombone|saxophone|accordion|"
                           r"drum kit|drums|flute|clarinet|harp|banjo|organ pipes|mandolin)\b"),
    ("sport", r"\b(football|soccer|tennis|basketball|baseball|cricket match|golf|rugby|"
              r"ice hockey|athlete|stadium|skier|skiing|snowboard|surfer|surfboard|"
              r"running race|marathon|wrestl\w+|boxing)\b"),
    # -- vehicles
    ("aircraft", r"\b(aeroplane|airplane|aircraft|biplane|helicopter|glider|jet fighter|"
                 r"hot air balloon|airship)\b"),
    ("train", r"\b(locomotive|steam train|a train|the train|tram|streetcar|railcar|carriages "
              r"of a train)\b"),
    ("railway", r"\b(railway|railroad|rail tracks|railway tracks|train station|railway station|"
                r"railway platform|level crossing)\b"),
    ("boat", r"\b(boat|ship|sailboat|sailing vessel|yacht|canoe|kayak|ferry|barge|dinghy|"
             r"rowing boat|trawler)\b"),
    ("harbour", r"\b(harbou?r|marina|quay|jetty|pier|dockyard|moorings?)\b"),
    ("bus", r"\b(bus|double[- ]decker|coach parked)\b"),
    ("truck", r"\b(truck|lorry|van parked|pickup truck|tractor|bulldozer|excavator|"
              r"fire engine|ambulance)\b"),
    ("motorcycle", r"\b(motorcycle|motorbike|moped|scooter)\b"),
    ("bicycle", r"\b(bicycle|bike|cyclist)\b"),
    ("car", r"\b(car|automobile|sedan|hatchback|estate car|taxi|jeep|convertible)\b"),
    # -- work, tools, industry
    ("power_tool", r"\b(power drill|electric drill|chainsaw|circular saw|angle grinder|"
                   r"jackhammer|nail gun)\b"),
    ("hand_tool", r"\b(hammer|screwdriver|wrench|spanner|pliers|chisel|axe|handsaw|hand saw|"
                  r"shovel|spade|rake|pitchfork|trowel|scythe)\b"),
    ("construction_site", r"\b(construction site|scaffolding|building under construction|"
                          r"cement mixer|crane lift\w*|construction work\w*)\b"),
    ("machine", r"\b(machine|machinery|engine|turbine|generator|gears?|mechanism|"
                r"conveyor|windmill|water wheel|printing press)\b"),
    ("laboratory", r"\bscientific (apparatus|instrument)\b"),
    ("damaged_item", r"\b(rusty|rusted|rusting|broken|damaged|cracked|shattered|ruined|ruins|"
                     r"derelict|dilapidated|abandoned|crumbling|decay\w*|worn[- ]out|"
                     r"peeling paint|burnt[- ]out|wreck\w*)\b"),
    ("waste_container", r"\b(rubbish|garbage|trash|dumpster|wheelie bin|waste bin|litter)\b"),
    # -- signage and print in the world
    ("road_sign", r"\b(road sign|street sign|traffic sign|stop sign|signpost|traffic light|"
                  r"speed limit sign|direction sign)\b"),
    ("sign", r"\b(sign|signage|plaque|billboard|noticeboard|notice board|nameplate|"
             r"information board|banner)\b"),
    ("document", r"\b(document|newspaper|piece of paper with|sheet of paper with|printed text|"
                 r"receipt|form filled|typed page)\b"),
    ("book", r"\b(book|books|bookshelf|bookcase|library shelf)\b"),
    # -- commerce and interiors
    ("storefront", r"\b(storefront|shop ?front|store ?front|shop window|display window)\b"),
    ("supermarket", r"\b(supermarket|grocery store|shopping trolley|shopping cart|"
                    r"checkout counter|shop aisle)\b"),
    ("market_stall", r"\b(market stall|market place|marketplace|street market|farmers.? market|"
                     r"bazaar|vendor selling)\b"),
    ("restaurant_interior", r"\b(restaurant|caf[eé]|bistro|pub interior|bar counter|"
                            r"dining room|tea room)\b"),
    ("kitchen", r"\b(kitchen|worktop|countertop with|cooker|stove|oven)\b"),
    ("bathroom", r"\b(bathroom|bathtub|shower|washbasin|wash basin|toilet)\b"),
    ("bedroom", r"\b(bedroom|a bed|the bed|bunk bed|headboard)\b"),
    ("living_room", r"\b(living room|sitting room|lounge|sofa|couch|armchair)\b"),
    ("office_interior", r"\b(office|desk with|classroom|lecture hall|meeting room|"
                        r"reception desk)\b"),
    # -- food and drink
    ("prepared_food", r"\b(meal|dish of|plate of food|sandwich|pizza|cake|pie|bread|loaf|soup|"
                      r"pastry|pasta|noodles|salad|breakfast|dessert|cookie|biscuit)\b"),
    ("fruit", r"\b(apple|orange|banana|grapes|berries|strawberr\w+|lemon|pear|cherr\w+|melon|"
              r"peach|plum|fruit)\b"),
    ("vegetable", r"\b(vegetable|tomato|potato|carrot|onion|cabbage|pumpkin|cucumber|pepper|"
                  r"lettuce|beans|corn cob)\b"),
    ("beverage", r"\b(coffee|tea in|glass of|bottle of wine|beer|cocktail|juice|drink)\b"),
    ("kitchenware", r"\b(cup|mug|teapot|kettle|saucepan|frying pan|cutlery|fork|spoon|knife|"
                    r"plate|bowl|jug|tableware)\b"),
    # -- products and materials
    ("computer_hardware", r"\b(computer|laptop|keyboard|monitor|server rack|circuit board|"
                          r"motherboard|hard drive)\b"),
    ("electronics", r"\b(television|radio set|telephone|camera|loudspeaker|amplifier|"
                    r"electronic device|gramophone|typewriter)\b"),
    ("clothing", r"\b(dress|coat|jacket|shirt|uniform|trousers|skirt|hat|cap|scarf|gloves|"
                 r"costume|clothing)\b"),
    ("footwear", r"\b(shoes?|boots?|sandals?|slippers?|clogs)\b"),
    ("jewellery", r"\b(jewell?ery|necklace|bracelet|earrings?|brooch|pendant|ring set with)\b"),
    ("textile", r"\b(fabric|textile|cloth|rug|carpet|quilt|embroider\w+|lace|woven|knitted|"
                r"curtain)\b"),
    ("toy", r"\b(toy|doll|teddy bear|puppet|model train set|rocking horse|board game)\b"),
    ("packaging", r"\b(cardboard box|packaging|crate|barrel|sack|canister|tin can|jar of)\b"),
    ("furniture", r"\b(chair|table|cupboard|cabinet|wardrobe|chest of drawers|bench|stool|"
                  r"furniture)\b"),
    # -- living things
    ("insect", r"\b(insect|beetle|butterfly|moth|bee|wasp|hornet|dragonfly|damselfly|ant|"
               r"spider|grasshopper|cricket perched|caterpillar|ladybird|ladybug|fly perched|"
               r"cicada|mantis|bug)\b"),
    ("fish_or_marine", r"\b(fish|coral|jellyfish|crab|starfish|octopus|squid|seashell|"
                       r"sea ?shell|mollusc|anemone|urchin)\b"),
    ("bird", r"\b(bird|sparrow|pigeon|gull|seagull|duck|goose|swan|eagle|hawk|owl|heron|"
             r"robin|finch|crow|raven|parrot|penguin|stork|woodpecker)\b"),
    ("dog", r"\b(dog|puppy|terrier|retriever|spaniel|hound)\b"),
    ("cat", r"\b(cat|kitten|feline)\b"),
    ("farm_animal", r"\b(cow|cattle|bull|calf|sheep|lamb|goat|pig|piglet|horse|pony|donkey|"
                    r"chicken|hen|rooster|turkey|farm animal)\b"),
    ("wild_animal", r"\b(deer|fox|badger|bear|wolf|lizard|snake|frog|toad|squirrel|monkey|"
                    r"elephant|lion|tiger|giraffe|zebra|rabbit|hare|hedgehog|seal|whale|"
                    r"dolphin|turtle|tortoise|bat hanging)\b"),
    ("fungus", r"\b(mushroom|fungus|fungi|toadstool|lichen|mould|mold growing)\b"),
    ("flower", r"\b(flower|blossom|petals?|bloom|orchid|daisy|rose|tulip|lily|sunflower|"
               r"daffodil|poppy|iris|wildflower)\b"),
    ("houseplant", r"\b(potted plant|houseplant|plant in a pot|window box)\b"),
    ("tree", r"\b(tree|trunk of|branch|branches|foliage|leaves|shrub|bush|hedge)\b"),
    # -- places
    ("street_scene", r"\b(street|road|avenue|alley|lane lined|pavement|sidewalk|"
                     r"pedestrian crossing|town square|city centre|city center|intersection)\b"),
    ("church_building", r"\b(church|cathedral|chapel|abbey|mosque|temple|synagogue|monastery|"
                        r"bell tower|steeple|minaret)\b"),
    ("statue_or_monument", r"\b(statue|monument|memorial|obelisk|sculpture|bust of|"
                           r"war memorial|gravestone|tombstone)\b"),
    ("bridge", r"\b(bridge|viaduct|aqueduct|footbridge)\b"),
    ("industrial", r"\b(factory|industrial|warehouse|power station|power plant|refinery|"
                   r"chimney|smokestack|silo|quarry|mine shaft)\b"),
    ("building_exterior", r"\b(building|house|cottage|castle|palace|tower|barn|farmhouse|"
                          r"facade|fa[cç]ade|mansion|hut|cabin|architecture)\b"),
    ("architectural_detail", r"\b(window|door|doorway|staircase|stairs|archway|arch|column|"
                             r"pillar|roof|balcony|gate|ceiling|fireplace)\b"),
    ("garden_or_park", r"\b(garden|park|lawn|flowerbed|flower bed|greenhouse|orchard|"
                       r"allotment|courtyard)\b"),
    ("snow_scene", r"\b(snow|snowy|snow-covered|frost|icicle|frozen|glacier|blizzard)\b"),
    ("mountain", r"\b(mountain|mountains|peak|summit|cliff|cliffs|valley|canyon|volcano|"
                 r"ridge|hillside|crag)\b"),
    ("coast_or_beach", r"\b(beach|coast|coastline|shore|shoreline|sea|ocean|waves|sand dune|"
                       r"cove|bay)\b"),
    ("river_or_lake", r"\b(river|lake|pond|stream|brook|waterfall|canal|reservoir|"
                      r"riverbank|lagoon)\b"),
    ("forest", r"\b(forest|woodland|woods|jungle|rainforest|grove|copse)\b"),
    ("landscape", r"\b(landscape|field|meadow|countryside|hills|prairie|desert|moorland|"
                  r"pasture|farmland|horizon)\b"),
    ("people_group", r"\b(group of people|crowd|people standing|people sitting|men and women|"
                     r"children playing|family posing|soldiers|workers)\b"),
]
BUCKETS = tuple(dict.fromkeys(name for name, _ in BUCKET_RULES))

_PLAIN_WORD = re.compile(r"^[a-z][a-z-]*$")
_ALTERNATION = re.compile(r"\(([^()]*\|[^()]*)\)")


def _pluralise(token: str) -> str:
    """`cup` -> `cup(?:s)?`, `bush` -> `bush(?:es)?`.

    PD12M captions are descriptive, so plurals are everywhere ("trees", "windows", "flowers").
    Anything that is not a plain lower-case word -- a phrase like `trunk of`, or a fragment that
    already carries regex syntax like `decay\\w*` -- is left exactly as it was written.
    """
    if not _PLAIN_WORD.match(token) or token.endswith("s") or len(token) < 3:
        return token
    return token + ("(?:es)?" if token.endswith(("x", "z", "ch", "sh")) else "(?:s)?")


def _expand(pattern: str) -> str:
    """Pluralise a rule's alternatives and make its groups non-capturing.

    Non-capturing matters: the bucket rules are joined into ONE regex whose only capturing groups
    are the per-rule named ones, so `match.lastindex` says which rule fired.
    """
    pattern = _ALTERNATION.sub(
        lambda m: "(?:" + "|".join(_pluralise(t) for t in m.group(1).split("|")) + ")", pattern)
    return re.sub(r"\((?!\?)", "(?:", pattern)   # safety net: no stray capturing group survives


# One regex, not 73.  Python's alternation is leftmost-first, so a single `search` returns the
# rule whose keyword appears EARLIEST in the caption, ties broken by the order of BUCKET_RULES.
# That is the behaviour we want: a synthetic caption names its subject first ("a wasp sitting on
# top of a white cup" is about the wasp, not the cup), and matching on position rather than on
# rule order stops a generic rule stealing a photo from a specific one.
_COMBINED = re.compile(
    "|".join(f"(?P<b{i}>{_expand(pattern)})" for i, (_, pattern) in enumerate(BUCKET_RULES)),
    re.I)
_BY_INDEX = {i + 1: name for i, (name, _) in enumerate(BUCKET_RULES)}
assert _COMBINED.groups == len(BUCKET_RULES), "a bucket rule leaked a capturing group"


def bucket(caption: str) -> str | None:
    """-> the coarse subject bucket for this caption, or None when nothing matches."""
    match = _COMBINED.search(str(caption or ""))
    return None if match is None else _BY_INDEX[match.lastindex]


# --------------------------------------------------------------------------- objects
# Concrete things a template may ask about.  Each entry is (phrase as it reads in a question,
# regex that spots it in a caption).  Phrases carry their own article, so
# `photo._thing_phrase` passes them through untouched.
OBJECT_LEXICON: list[tuple[str, str]] = [
    ("a car", r"\b(car|automobile|sedan|taxi)\b"),
    ("a bus", r"\bbus\b"),
    ("a truck", r"\b(truck|lorry)\b"),
    ("a bicycle", r"\b(bicycle|bike)\b"),
    ("a motorcycle", r"\b(motorcycle|motorbike|moped)\b"),
    ("a train", r"\b(train|locomotive|tram)\b"),
    ("a boat", r"\b(boat|ship|yacht|canoe|ferry)\b"),
    ("an aeroplane", r"\b(aeroplane|airplane|aircraft|biplane)\b"),
    ("a horse", r"\b(horse|pony)\b"),
    ("a cow", r"\b(cow|cattle|bull|calf)\b"),
    ("a sheep", r"\b(sheep|lamb)\b"),
    ("a dog", r"\b(dog|puppy)\b"),
    ("a cat", r"\b(cat|kitten)\b"),
    ("a bird", r"\b(bird|sparrow|pigeon|gull|duck|goose|swan|eagle|owl|heron|crow|parrot)\b"),
    ("a fish", r"\bfish\b"),
    ("an insect", r"\b(insect|beetle|butterfly|moth|bee|wasp|dragonfly|ant|grasshopper|"
                  r"caterpillar|ladybird)\b"),
    ("a flower", r"\b(flower|blossom|petals?|bloom|orchid|daisy|rose|tulip|lily)\b"),
    ("a tree", r"\b(tree|trunk|branch|branches)\b"),
    ("a potted plant", r"\b(potted plant|houseplant|plant in a pot|window box)\b"),
    ("a mushroom", r"\b(mushroom|toadstool|fungus)\b"),
    ("a chair", r"\b(chair|armchair|stool)\b"),
    ("a table", r"\b(table|desk)\b"),
    ("a bed", r"\bbed\b"),
    ("a sofa", r"\b(sofa|couch)\b"),
    ("a lamp", r"\b(lamp|lantern|light fitting)\b"),
    ("a mirror", r"\bmirror\b"),
    ("a clock", r"\b(clock|sundial)\b"),
    ("a window", r"\bwindows?\b"),
    ("a door", r"\b(door|doorway|gate)\b"),
    ("a staircase", r"\b(staircase|stairs|steps leading)\b"),
    ("a fence", r"\b(fence|railing|hedge)\b"),
    ("a bench", r"\bbench\b"),
    ("a bridge", r"\b(bridge|viaduct)\b"),
    ("a building", r"\b(building|house|cottage|castle|tower|barn)\b"),
    ("a statue", r"\b(statue|sculpture|monument|bust)\b"),
    ("a street sign", r"\b(street sign|road sign|signpost|traffic sign)\b"),
    ("a traffic light", r"\btraffic light\b"),
    ("a rubbish bin", r"\b(rubbish bin|dustbin|waste bin|trash can|litter bin)\b"),
    ("a laptop", r"\b(laptop|computer)\b"),
    ("a mobile phone", r"\b(mobile phone|smartphone|cell phone)\b"),
    ("a television", r"\b(television|tv set)\b"),
    ("a camera", r"\bcamera\b"),
    ("a book", r"\bbooks?\b"),
    ("a newspaper", r"\bnewspaper\b"),
    ("a plate", r"\bplates?\b"),
    ("a bowl", r"\bbowls?\b"),
    ("a cup", r"\b(cup|mug|teacup)\b"),
    ("a bottle", r"\bbottles?\b"),
    ("a glass", r"\b(drinking glass|wine glass|glass of)\b"),
    ("a knife", r"\bknife\b"),
    ("a fork", r"\bforks?\b"),
    ("a spoon", r"\bspoons?\b"),
    ("a kettle", r"\b(kettle|teapot)\b"),
    ("a sink", r"\b(sink|washbasin|basin)\b"),
    ("an oven", r"\b(oven|stove|cooker)\b"),
    ("a refrigerator", r"\b(refrigerator|fridge)\b"),
    ("a handbag", r"\b(handbag|purse)\b"),
    ("a backpack", r"\b(backpack|rucksack)\b"),
    ("an umbrella", r"\bumbrella\b"),
    ("a pair of shoes", r"\b(shoes|boots|sandals)\b"),
    ("a hat", r"\b(hat|cap|helmet)\b"),
    ("a jacket", r"\b(jacket|coat)\b"),
    ("a hammer", r"\bhammer\b"),
    ("a screwdriver", r"\bscrewdriver\b"),
    ("a ladder", r"\bladder\b"),
    ("a bucket", r"\b(bucket|pail)\b"),
    ("a guitar", r"\bguitar\b"),
    ("a piano", r"\bpiano\b"),
    ("a ball", r"\bball\b"),
    ("a suitcase", r"\b(suitcase|trunk case|luggage)\b"),
    ("a cardboard box", r"\b(cardboard box|carton)\b"),
    ("a price label", r"\b(price (label|tag)|price list)\b"),
    ("a shopping trolley", r"\b(shopping trolley|shopping cart)\b"),
    ("a wheelchair", r"\bwheelchair\b"),
    ("a stethoscope", r"\bstethoscope\b"),
    ("a microscope", r"\bmicroscope\b"),
    ("a fire extinguisher", r"\bfire extinguisher\b"),
]
_LEXICON = [(phrase, re.compile(_expand(pattern), re.I)) for phrase, pattern in OBJECT_LEXICON]
LEXICON_PHRASES = tuple(phrase for phrase, _ in OBJECT_LEXICON)

# Things a photograph very often contains without the caption saying so.  They are kept out of the
# "probably not in this photo" pool, because "the caption did not mention it" is a much weaker
# claim for these than for, say, a fire extinguisher.
OFTEN_UNSAID = frozenset({
    "a window", "a door", "a fence", "a tree", "a building", "a chair", "a table", "a bench",
    "a staircase", "a lamp", "a bird", "a flower", "a potted plant", "a bottle", "a cup",
    "a plate", "a hat", "a jacket", "a pair of shoes", "a car",
})

MAX_PRESENT = 8
MAX_ABSENT = 12


def objects(caption: str, *, associated=(), rng=None,
            limit_present: int = MAX_PRESENT,
            limit_absent: int = MAX_ABSENT) -> tuple[list[str], list[str]]:
    """-> (claimed present, plausibly absent).

    `associated` is the bucket's typical-contents list (`photo.CATEGORY_ASSOCIATED`); anything in
    it is withheld from the absent list even when the caption is silent about it.  `rng` (a seeded
    `random.Random`) rotates the absent pool so that every photo does not get the same first
    twelve lexicon entries; without it the order is lexicon order.
    """
    text = str(caption or "")
    present, absent = [], []
    withheld = set(associated) | OFTEN_UNSAID
    for phrase, pattern in _LEXICON:
        if pattern.search(text):
            present.append(phrase)
        elif phrase not in withheld:
            absent.append(phrase)
    if rng is not None:
        rng.shuffle(present)
        rng.shuffle(absent)
    return present[:limit_present], absent[:limit_absent]
