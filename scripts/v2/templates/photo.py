"""Photo question templates for decision-v2.

Every public function here is pure: it takes an `inputs` mapping describing one photo (or a pair of
photos) plus a seeded `random.Random`, and returns a list of request fields in the shape
`vision_decision.contracts.Request` validates.  Nothing is read from disk, nothing is written, and
no answer is ever produced -- decision-v2 targets come from the 9B teacher, so a template's only
job is to ask something a photograph could honestly settle (or honestly fail to settle).

`inputs` keys
-------------
    id                 str   -- stable id of the photo, used only for reproducible wording
    category           str   -- coarse category of the photo (Commons seed label, or an Open Images
                                verified label).  NEVER used as, or turned into, a target.
    present            list  -- things a human verified as being in the photo (Open Images
                                Confidence==1), or [] when the source has no such annotation
    absent             list  -- things a human verified as NOT being in the photo (Confidence==0)
    title              str   -- the upstream title/filename text, used to colour the state
    second             dict  -- for the two-image families: the same keys for the other photo
    same_category      bool  -- for the two-image families: whether the pair shares a category

Every family carries at least eight paraphrases, and about a third of the questions attach a short
criterion to each option (Jev's "option -> what it means" style).  The `unknown` option is added by
the loader and is never listed here.
"""
from __future__ import annotations

FAMILIES = (
    "scene_type", "main_object", "object_present", "count_bucket", "photo_quality",
    "colour_of_x", "text_visible", "material", "indoor_outdoor",
    "two_image_same_subject", "two_image_which",
)
SINGLE_IMAGE_FAMILIES = FAMILIES[:9]
TWO_IMAGE_FAMILIES = FAMILIES[9:]

# --------------------------------------------------------------------------- option banks
SCENES = [
    "a residential kitchen", "a living room", "a bedroom", "a bathroom", "an office interior",
    "a restaurant or cafe interior", "a supermarket or shop interior", "a shop front from the street",
    "a city street", "a suburban street", "a rural road", "a park or garden", "a beach or coastline",
    "a forest or woodland", "a farm or field", "a construction site", "a factory or workshop",
    "a laboratory", "a hospital or clinic room", "a classroom", "a warehouse", "a car park",
    "a railway station or platform", "an airport terminal", "a sports field or court",
    "a gym or fitness room", "a museum or gallery", "a place of worship", "a market stall",
    "a close-up of a single object on a plain background", "a close-up of a single object in use",
    "a hotel room", "a garage or driveway", "a stairwell or corridor", "a balcony or terrace",
    "an open field or meadow", "a harbour or marina", "a bridge or overpass", "a building site office",
    "a studio product shot",
]

SCENE_CRITERIA = {'a residential kitchen': 'a kitchen in a home: domestic worktops, a cooker, a sink',
    'a living room': 'a room for sitting: sofa or armchairs, usually a television',
    'a bedroom': 'a room built around a bed',
    'a bathroom': 'a room with a bath, shower, basin or lavatory',
    'an office interior': 'desks, screens and office chairs indoors',
    'a restaurant or cafe interior': 'tables laid for customers, a counter or bar',
    'a supermarket or shop interior': 'aisles, shelving and priced goods, seen from inside',
    'a shop front from the street': 'the outside of a shop: window, fascia, entrance',
    'a city street': 'a built-up street with traffic, pavements and buildings',
    'a suburban street': 'houses with gardens or driveways along a quieter road',
    'a rural road': 'a road through fields, woods or open country',
    'a park or garden': 'planted, tended outdoor ground with grass, beds or paths',
    'a beach or coastline': 'sand, shingle or rocks meeting the sea',
    'a forest or woodland': 'standing trees close enough to form a canopy',
    'a farm or field': 'worked agricultural land, livestock or farm buildings',
    'a construction site': 'a building under construction: scaffolding, plant, materials',
    'a factory or workshop': 'a place where things are made or repaired: benches, machines',
    'a laboratory': 'benches with scientific apparatus, glassware or instruments',
    'a hospital or clinic room': 'a clinical room: beds, medical equipment, wipe-clean surfaces',
    'a classroom': "desks facing a board or a teacher's position", 'a warehouse': 'bulk storage: racking, pallets, wide floors',
    'a car park': 'marked bays for parking, indoors or out',
    'a railway station or platform': 'a platform, track and station furniture',
    'an airport terminal': 'a passenger terminal: gates, check-in, concourse',
    'a sports field or court': 'a marked playing surface for a sport',
    'a gym or fitness room': 'exercise machines or free weights indoors',
    'a museum or gallery': 'exhibits displayed with labels or cases',
    'a place of worship': 'a church, mosque, temple, synagogue or similar',
    'a market stall': 'goods laid out for sale on a stall or trestle',
    'a close-up of a single object on a plain background': 'one object filling the frame against a blank backdrop',
    'a close-up of a single object in use': 'one object filling the frame while being used or held',
    'a hotel room': 'a made-up room of a kind let to guests',
    'a garage or driveway': 'a vehicle space at a house: garage, hardstanding or drive',
    'a stairwell or corridor': 'a passage or staircase inside a building',
    'a balcony or terrace': 'an outdoor platform attached to a building',
    'an open field or meadow': 'open grass or crop land with no buildings close by',
    'a harbour or marina': 'moored boats, quays or pontoons',
    'a bridge or overpass': 'a span carrying a road, path or railway over something',
    'a building site office': 'a temporary cabin or office at a work site',
    'a studio product shot': 'a lit product photograph on a seamless background'}

COLOUR_CRITERIA = {'red': 'clear red, including crimson and scarlet',
    'orange': 'orange, between red and yellow',
    'yellow': 'yellow, including mustard and lemon',
    'green': 'any green, from olive to emerald',
    'blue': 'mid blue, not navy and not turquoise',
    'purple': 'purple or violet, including mauve',
    'pink': 'pink, including rose and magenta',
    'brown': 'brown, including tan and chocolate',
    'black': 'black or near-black',
    'white': 'white or off-white',
    'grey': 'neutral grey between black and white',
    'beige': 'pale sandy neutral, darker than white',
    'gold': 'metallic yellow, like brass or gilt',
    'silver': 'metallic grey, like steel or chrome',
    'turquoise': 'blue-green, like teal or aqua',
    'navy': 'very dark blue'}

MATERIALS = [
    "wood", "metal", "plastic", "glass", "fabric or textile", "paper or cardboard",
    "stone or concrete", "ceramic or porcelain", "leather", "rubber", "foam", "a painted surface",
]
MATERIAL_CRITERIA = {
    "wood": "visible grain, planks or a natural timber surface",
    "metal": "a hard specular surface, screws, welds or bare steel/aluminium",
    "plastic": "a moulded, uniform, often glossy synthetic surface",
    "glass": "transparent or translucent, with reflections or a visible edge",
    "fabric or textile": "woven or knitted, with folds, weave or stitching",
    "paper or cardboard": "printed sheet, corrugation or a folded carton",
    "stone or concrete": "mineral, rough or cast, with aggregate or veining",
    "ceramic or porcelain": "glazed, hard and usually white or coloured clay",
    "leather": "a supple hide with grain, seams or worn edges",
    "rubber": "matte, flexible and usually black, like a tyre or seal",
    "foam": "soft, porous, compressible padding",
    "a painted surface": "an opaque coat of paint that hides what is underneath",
}

COLOURS = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white",
    "grey", "beige", "gold", "silver", "turquoise", "navy",
]

INDOOR_OPTIONS = [
    ("indoors", "the photo is taken inside a building, with walls or a ceiling around the subject"),
    ("outdoors", "the photo is taken in the open air"),
    ("partly covered", "a porch, a covered market, an open garage or a vehicle interior"),
    ("cannot be placed", "the frame is too tight or too plain to show where this is"),
]

QUALITY_LEVELS = [
    (0, "unusable: too blurred, too dark or too washed out to make anything out"),
    (1, "poor: the subject is recognisable but details are lost"),
    (2, "acceptable: ordinary snapshot quality, the subject reads clearly"),
    (3, "good: sharp, well exposed, the subject is easy to inspect"),
    (4, "excellent: crisp and evenly lit, small details are legible"),
]
QUALITY_LEVELS_SHORT = [
    (0, "unusable for any judgement about the subject"),
    (1, "poor but the subject is recognisable"),
    (2, "ordinary snapshot quality"),
    (3, "sharp and easy to inspect"),
]

COUNT_LEVELS = [
    (0, "none at all"),
    (1, "exactly one"),
    (2, "two or three"),
    (3, "four to six"),
    (4, "more than six"),
]

# Everyday things used to top up option sets and to ask about objects that are not there.
COMMON_OBJECTS = [
    "a chair", "a table", "a bed", "a sofa", "a lamp", "a mirror", "a clock", "a window", "a door",
    "a bicycle", "a car", "a bus", "a truck", "a motorcycle", "a boat", "an aeroplane", "a train",
    "a dog", "a cat", "a bird", "a horse", "a cow", "a sheep", "a fish", "a potted plant",
    "a laptop", "a mobile phone", "a television", "a keyboard", "a computer mouse", "a camera",
    "a printer", "a microwave", "an oven", "a refrigerator", "a kettle", "a toaster", "a sink",
    "a knife", "a fork", "a spoon", "a plate", "a bowl", "a cup", "a bottle", "a glass",
    "a handbag", "a backpack", "an umbrella", "a pair of shoes", "a hat", "a jacket", "a watch",
    "a book", "a newspaper", "a pen", "a pair of scissors", "a hammer", "a screwdriver", "a drill",
    "a ladder", "a bucket", "a broom", "a fire extinguisher", "a traffic light", "a street sign",
    "a rubbish bin", "a bench", "a fence", "a staircase", "a suitcase", "a guitar", "a piano",
    "a ball", "a helmet", "a wheelchair", "a stethoscope", "a syringe", "a microscope",
    "a shopping trolley", "a cash register", "a price label", "a cardboard box", "a pallet",
]

# Commons seed label -> a few scenes that are plausible for it.  Used ONLY to make the option set
# plausible; the correct option is never marked, and the teacher may well answer `unknown`.
CATEGORY_SCENES = {
    "street_scene": ["a city street", "a suburban street", "a shop front from the street"],
    "kitchen": ["a residential kitchen", "a restaurant or cafe interior"],
    "car": ["a car park", "a garage or driveway", "a city street"],
    "bicycle": ["a city street", "a park or garden", "a garage or driveway"],
    "bus": ["a city street", "a railway station or platform", "a car park"],
    "train": ["a railway station or platform", "a bridge or overpass"],
    "boat": ["a harbour or marina", "a beach or coastline"],
    "aircraft": ["an airport terminal", "an open field or meadow"],
    "dog": ["a park or garden", "a living room"],
    "cat": ["a living room", "a bedroom"],
    "bird": ["a park or garden", "a forest or woodland"],
    "farm_animal": ["a farm or field", "an open field or meadow"],
    "prepared_food": ["a residential kitchen", "a restaurant or cafe interior", "a studio product shot"],
    "vegetable": ["a market stall", "a residential kitchen", "a studio product shot"],
    "fruit": ["a market stall", "a studio product shot"],
    "beverage": ["a restaurant or cafe interior", "a studio product shot"],
    "hand_tool": ["a factory or workshop", "a studio product shot"],
    "power_tool": ["a factory or workshop", "a construction site"],
    "sign": ["a city street", "a shop front from the street"],
    "road_sign": ["a city street", "a rural road"],
    "document": ["a studio product shot", "an office interior"],
    "storefront": ["a shop front from the street", "a city street"],
    "supermarket": ["a supermarket or shop interior", "a warehouse"],
    "restaurant_interior": ["a restaurant or cafe interior", "a hotel room"],
    "clothing": ["a studio product shot", "a supermarket or shop interior"],
    "footwear": ["a studio product shot", "a supermarket or shop interior"],
    "electronics": ["a studio product shot", "an office interior"],
    "computer_hardware": ["a studio product shot", "an office interior"],
    "furniture": ["a living room", "a studio product shot"],
    "living_room": ["a living room", "a hotel room"],
    "bedroom": ["a bedroom", "a hotel room"],
    "bathroom": ["a bathroom", "a hotel room"],
    "office_interior": ["an office interior", "a classroom"],
    "sport": ["a sports field or court", "a gym or fitness room"],
    "medical_device": ["a hospital or clinic room", "a laboratory"],
    "laboratory": ["a laboratory", "a factory or workshop"],
    "damaged_item": ["a close-up of a single object on a plain background", "a garage or driveway"],
    "construction_site": ["a construction site", "a building site office"],
    "machine": ["a factory or workshop", "a warehouse"],
    "musical_instrument": ["a studio product shot", "a museum or gallery"],
    "kitchenware": ["a residential kitchen", "a studio product shot"],
    "toy": ["a studio product shot", "a bedroom"],
    "houseplant": ["a living room", "a balcony or terrace"],
    "book": ["a studio product shot", "a museum or gallery"],
    "packaging": ["a studio product shot", "a warehouse"],
    "waste_container": ["a city street", "a car park"],
}

# What a Commons file filed under each seed category is most likely a picture of.  This is a
# curator-supplied CLAIM (the category a human filed the file under), not a verified label, and it
# is used only to make an option set plausible.  The teacher still decides every answer.
CATEGORY_OBJECT = {
    "street_scene": "a street scene", "kitchen": "a kitchen", "car": "a car", "bicycle": "a bicycle",
    "bus": "a bus", "train": "a train", "boat": "a boat", "aircraft": "an aircraft", "dog": "a dog",
    "cat": "a cat", "bird": "a bird", "farm_animal": "a farm animal",
    "prepared_food": "a prepared dish of food", "vegetable": "a vegetable", "fruit": "a fruit",
    "beverage": "a drink", "hand_tool": "a hand tool", "power_tool": "a power tool", "sign": "a sign",
    "road_sign": "a road sign", "document": "a printed document", "storefront": "a shop front",
    "supermarket": "a supermarket interior", "restaurant_interior": "a restaurant interior",
    "clothing": "a piece of clothing", "footwear": "a shoe", "electronics": "an electronic device",
    "computer_hardware": "a piece of computer hardware", "furniture": "a piece of furniture",
    "living_room": "a living room", "bedroom": "a bedroom", "bathroom": "a bathroom",
    "office_interior": "an office interior", "sport": "people playing a sport",
    "medical_device": "a medical device", "laboratory": "a laboratory",
    "damaged_item": "a damaged object", "construction_site": "a construction site",
    "machine": "a machine", "musical_instrument": "a musical instrument",
    "kitchenware": "a piece of kitchenware", "toy": "a toy", "houseplant": "a houseplant",
    "book": "a book", "packaging": "packaging", "waste_container": "a waste container",
}

# Everyday objects that are *likely* to be in a photo filed under each Commons seed category.  They
# are held out of the "things that are probably not there" pool, so a question built to be honestly
# unanswerable is less likely to name something that is in fact in the frame.  Commons gives no
# per-file object labels, so this is the best that can be done there and the source README says so.
CATEGORY_ASSOCIATED = {
    "street_scene": ["a car", "a bus", "a street sign", "a traffic light", "a bench", "a bicycle", "a rubbish bin", "a fence"],
    "kitchen": ["a sink", "an oven", "a microwave", "a refrigerator", "a kettle", "a toaster", "a plate", "a bowl", "a cup", "a knife", "a table", "a chair"],
    "car": ["a car", "a fence", "a street sign"],
    "bicycle": ["a bicycle", "a fence", "a helmet"],
    "bus": ["a bus", "a street sign", "a traffic light", "a car"],
    "train": ["a train", "a fence", "a staircase"],
    "boat": ["a boat"],
    "aircraft": ["an aeroplane", "a suitcase"],
    "dog": ["a dog", "a fence", "a bench"],
    "cat": ["a cat", "a chair", "a sofa", "a bed", "a window"],
    "bird": ["a bird", "a fence"],
    "farm_animal": ["a cow", "a sheep", "a horse", "a fence"],
    "prepared_food": ["a plate", "a bowl", "a fork", "a knife", "a spoon", "a cup", "a glass", "a table"],
    "vegetable": ["a bowl", "a plate", "a knife", "a cardboard box", "a price label"],
    "fruit": ["a bowl", "a plate", "a knife", "a price label"],
    "beverage": ["a glass", "a bottle", "a cup", "a table"],
    "hand_tool": ["a hammer", "a screwdriver", "a pair of scissors", "a knife"],
    "power_tool": ["a drill", "a screwdriver", "a hammer", "a ladder"],
    "sign": ["a street sign", "a traffic light", "a price label"],
    "road_sign": ["a street sign", "a traffic light", "a car", "a fence"],
    "document": ["a book", "a newspaper", "a pen", "a price label"],
    "storefront": ["a street sign", "a car", "a bench", "a rubbish bin", "a price label", "a bicycle"],
    "supermarket": ["a shopping trolley", "a price label", "a cash register", "a cardboard box", "a bottle"],
    "restaurant_interior": ["a table", "a chair", "a plate", "a glass", "a cup", "a lamp", "a window"],
    "clothing": ["a jacket", "a hat", "a pair of shoes", "a handbag"],
    "footwear": ["a pair of shoes", "a price label"],
    "electronics": ["a television", "a laptop", "a mobile phone", "a camera", "a keyboard", "a computer mouse"],
    "computer_hardware": ["a laptop", "a keyboard", "a computer mouse", "a printer", "a television"],
    "furniture": ["a chair", "a table", "a bed", "a sofa", "a lamp", "a mirror"],
    "living_room": ["a sofa", "a chair", "a table", "a lamp", "a television", "a mirror", "a window", "a potted plant", "a clock"],
    "bedroom": ["a bed", "a lamp", "a mirror", "a window", "a chair", "a clock"],
    "bathroom": ["a sink", "a mirror", "a window", "a door"],
    "office_interior": ["a chair", "a table", "a laptop", "a keyboard", "a computer mouse", "a printer", "a lamp", "a clock", "a window"],
    "sport": ["a ball", "a helmet", "a bench", "a fence"],
    "medical_device": ["a stethoscope", "a syringe", "a wheelchair", "a microscope"],
    "laboratory": ["a microscope", "a syringe", "a bottle", "a glass"],
    "damaged_item": ["a car", "a window", "a door", "a fence"],
    "construction_site": ["a ladder", "a bucket", "a helmet", "a fence", "a pallet", "a truck"],
    "machine": ["a machine", "a ladder", "a bucket"],
    "musical_instrument": ["a guitar", "a piano"],
    "kitchenware": ["a plate", "a bowl", "a cup", "a knife", "a fork", "a spoon", "a kettle"],
    "toy": ["a ball", "a car", "a doll"],
    "houseplant": ["a potted plant", "a window", "a table"],
    "book": ["a book", "a newspaper", "a pen"],
    "packaging": ["a cardboard box", "a bottle", "a price label", "a pallet"],
    "waste_container": ["a rubbish bin", "a fence", "a car"],
}

# --------------------------------------------------------------------------- pd12m buckets
# Added 23 Sept 2026 for the `pd12m` source.  PD12M is captioned free text rather than filed under
# a curator's category, so its subject buckets reach into landscape, nature and architecture
# subjects the Commons category crawl never produced.  This block is purely ADDITIVE: it only adds
# new keys, and `commons_photos` / `openimages_v2` never emit any of them, so nothing those two
# sources do changes.  Scene values are drawn from `SCENES` above, object phrases carry their own
# article (so `_thing_phrase` leaves them alone), and the associated lists use `COMMON_OBJECTS`
# wording so they really do withhold those items from the "probably not there" pool.
PD12M_CATEGORY_OBJECT = {
    "railway": "a railway track", "harbour": "a harbour", "truck": "a truck",
    "motorcycle": "a motorcycle", "industrial": "an industrial building",
    "market_stall": "a market stall", "jewellery": "a piece of jewellery",
    "textile": "a piece of fabric", "insect": "an insect",
    "fish_or_marine": "a fish or sea creature", "wild_animal": "a wild animal",
    "fungus": "a mushroom", "flower": "a flower", "tree": "a tree",
    "church_building": "a church or other place of worship",
    "statue_or_monument": "a statue or monument", "bridge": "a bridge",
    "building_exterior": "the outside of a building",
    "architectural_detail": "part of a building", "garden_or_park": "a garden or park",
    "snow_scene": "a snowy scene", "mountain": "a mountain",
    "coast_or_beach": "a beach or coastline", "river_or_lake": "a river or lake",
    "forest": "a forest", "landscape": "an open landscape",
    "people_group": "a group of people",
}
PD12M_CATEGORY_SCENES = {
    "railway": ["a railway station or platform", "a bridge or overpass"],
    "harbour": ["a harbour or marina", "a beach or coastline"],
    "truck": ["a city street", "a construction site", "a car park"],
    "motorcycle": ["a city street", "a garage or driveway", "a rural road"],
    "industrial": ["a factory or workshop", "a warehouse"],
    "market_stall": ["a market stall", "a city street"],
    "jewellery": ["a studio product shot", "a museum or gallery"],
    "textile": ["a studio product shot", "a close-up of a single object on a plain background"],
    "insect": ["a park or garden", "an open field or meadow", "a forest or woodland"],
    "fish_or_marine": ["a beach or coastline", "a harbour or marina"],
    "wild_animal": ["a forest or woodland", "an open field or meadow", "a park or garden"],
    "fungus": ["a forest or woodland", "a park or garden"],
    "flower": ["a park or garden", "an open field or meadow", "a forest or woodland"],
    "tree": ["a forest or woodland", "a park or garden", "a rural road"],
    "church_building": ["a place of worship", "a city street"],
    "statue_or_monument": ["a park or garden", "a museum or gallery", "a city street"],
    "bridge": ["a bridge or overpass", "a city street"],
    "building_exterior": ["a suburban street", "a city street", "a rural road"],
    "architectural_detail": ["a stairwell or corridor", "a balcony or terrace",
                             "a shop front from the street"],
    "garden_or_park": ["a park or garden", "an open field or meadow"],
    "snow_scene": ["an open field or meadow", "a rural road", "a forest or woodland"],
    "mountain": ["an open field or meadow", "a forest or woodland", "a rural road"],
    "coast_or_beach": ["a beach or coastline", "a harbour or marina"],
    "river_or_lake": ["a park or garden", "a bridge or overpass", "a forest or woodland"],
    "forest": ["a forest or woodland", "a park or garden"],
    "landscape": ["an open field or meadow", "a farm or field", "a rural road"],
    "people_group": ["a city street", "a park or garden", "a market stall"],
}
PD12M_CATEGORY_ASSOCIATED = {
    "railway": ["a train", "a fence", "a staircase", "a street sign", "a bench"],
    "harbour": ["a boat", "a fence", "a bench"],
    "truck": ["a truck", "a car", "a fence", "a street sign"],
    "motorcycle": ["a motorcycle", "a car", "a street sign", "a helmet", "a fence"],
    "industrial": ["a ladder", "a pallet", "a fence", "a truck", "a bucket"],
    "market_stall": ["a price label", "a cardboard box", "a bench", "a table", "a fence"],
    "jewellery": ["a mirror", "a price label"],
    "textile": ["a chair", "a bed", "a sofa"],
    "insect": ["a potted plant", "a fence"],
    "fish_or_marine": ["a boat", "a fence"],
    "wild_animal": ["a fence", "a bird"],
    "fungus": ["a fence"],
    "flower": ["a potted plant", "a fence", "a bench"],
    "tree": ["a bench", "a fence", "a bird", "a bicycle"],
    "church_building": ["a bench", "a door", "a window", "a staircase", "a clock", "a fence"],
    "statue_or_monument": ["a bench", "a fence", "a potted plant"],
    "bridge": ["a car", "a fence", "a boat", "a street sign", "a bicycle"],
    "building_exterior": ["a window", "a door", "a staircase", "a fence", "a car",
                          "a street sign", "a bench", "a potted plant"],
    "architectural_detail": ["a window", "a door", "a staircase", "a mirror", "a lamp", "a fence"],
    "garden_or_park": ["a bench", "a fence", "a potted plant", "a rubbish bin", "a bicycle", "a dog"],
    "snow_scene": ["a fence", "a car", "a bench", "a hat", "a jacket"],
    "mountain": ["a fence", "a bench"],
    "coast_or_beach": ["a boat", "a fence", "a bench", "an umbrella"],
    "river_or_lake": ["a boat", "a fence", "a bench", "a bicycle"],
    "forest": ["a fence", "a bench", "a bird", "a dog"],
    "landscape": ["a fence", "a bench", "a cow", "a sheep", "a horse"],
    "people_group": ["a chair", "a table", "a bench", "a handbag", "a hat", "a jacket",
                     "a backpack", "a bicycle", "a car"],
}
CATEGORY_OBJECT.update(PD12M_CATEGORY_OBJECT)
CATEGORY_SCENES.update(PD12M_CATEGORY_SCENES)
CATEGORY_ASSOCIATED.update(PD12M_CATEGORY_ASSOCIATED)

# --------------------------------------------------------------------------- paraphrase banks
SCENE_QUESTIONS = [
    "Look at the photograph. Which of these best describes the place it was taken?",
    "What kind of place does this photo show? Pick the closest description.",
    "Judging only by the picture, where was this taken?",
    "Choose the setting that matches the photograph.",
    "Which description fits the scene in the image?",
    "Somebody has to file this photo by setting. Which one is right?",
    "From what is visible, what sort of location is this?",
    "Pick the option that describes the surroundings shown in the photo.",
    "Classify the scene in the photograph.",
    "If you had to tell a colleague where this photo was taken, which of these would you say?",
]
MAIN_OBJECT_QUESTIONS = [
    "What is the main thing this photograph is of?",
    "Which of these is the subject of the picture?",
    "The photo is mostly about one thing. Which one?",
    "Pick the item that the photograph is showing.",
    "Which of the following actually appears as the subject of this image?",
    "Somebody needs a one-line caption. Which of these is the subject?",
    "Looking at the picture, which listed item is the main subject?",
    "Choose the option that names what is shown.",
    "What does the photograph depict, out of these choices?",
    "Which of these things is the photo's subject?",
]
OBJECT_PRESENT_QUESTIONS = [
    "Is there {thing} visible anywhere in this photograph?",
    "Does the photo show {thing}?",
    "Can you see {thing} in the image?",
    "Look carefully: does {thing} appear in this picture?",
    "Is {thing} present in the photograph?",
    "Somebody claims this photo contains {thing}. Is that true?",
    "Does {thing} appear anywhere in the frame?",
    "Answer yes or no: the photograph shows {thing}.",
    "Would you say {thing} is in this image?",
    "Is {thing} part of what this photo shows?",
]
COUNT_QUESTIONS = [
    "How many {thing} can you count in this photograph?",
    "Count the {thing} that are visible in the image.",
    "How many {thing} appear in the picture?",
    "Give the number of {thing} you can see in this photo.",
    "Looking at the image, how many {thing} are there?",
    "How many separate {thing} does the photograph show?",
    "Count how many {thing} are in the frame.",
    "Pick the band that matches the number of {thing} in the photo.",
    "How many {thing} would you say are visible here?",
    "Estimate the number of {thing} shown in the image.",
]
QUALITY_QUESTIONS = [
    "How good is this photograph for judging what it shows?",
    "Rate the usable quality of this image.",
    "Could someone inspect the subject in this photo? Rate how well.",
    "How clear and well exposed is this picture?",
    "Judge the technical quality of the photograph.",
    "How easy is it to see detail in this image?",
    "Rate how usable this photo is as evidence about its subject.",
    "If this photo were submitted with a claim, how good would it be?",
    "Score the sharpness and exposure of this image.",
    "How well does this photograph show its subject?",
]
COLOUR_QUESTIONS = [
    "What colour is {thing} in this photograph?",
    "Looking at the image, which colour best describes {thing}?",
    "Pick the colour of {thing} as it appears in the photo.",
    "Which of these colours does {thing} have here?",
    "What is the dominant colour of {thing} in this picture?",
    "Somebody is filling in a colour field for {thing}. Which one?",
    "From the photograph, what colour is {thing}?",
    "Choose the colour that matches {thing} in the image.",
    "Which colour would you record for {thing} shown here?",
    "Judging by the photo, {thing} is which colour?",
]
TEXT_VISIBLE_QUESTIONS = [
    "Is there any readable printed or written text in this photograph?",
    "Does the image contain words that could actually be read?",
    "Can you make out any writing, lettering or numbers in the picture?",
    "Is legible text visible anywhere in this photo?",
    "Does the photograph show any text a person could read?",
    "Somebody wants to transcribe this photo. Is there readable text in it?",
    "Are there any readable labels, signs or captions in the image?",
    "Answer yes or no: readable text appears in this photograph.",
    "Is any writing in the frame clear enough to read?",
    "Does this picture include legible words or numbers?",
]
MATERIAL_QUESTIONS = [
    "What is the main subject of this photograph mostly made of?",
    "Which material does the main object in the image appear to be?",
    "Judging by the surface in the photo, what is the subject made from?",
    "Pick the material of the main thing shown.",
    "What does the subject of this picture seem to be made of?",
    "Somebody is filling in a material field. Which one fits the photo?",
    "From the texture visible here, which material is it?",
    "Which of these materials best matches the subject of the image?",
    "What material would you record for the object in this photo?",
    "Looking at the photograph, the main object is made of what?",
]
MATERIAL_OF_X_QUESTIONS = [
    "What is {thing} in this photograph made of?",
    "Which material is {thing} shown here made from?",
    "Looking at the picture, what is {thing} made of?",
    "Pick the material of {thing} in this image.",
    "Somebody is filling in a material field for {thing}. Which one fits the photo?",
    "From the photograph, which material does {thing} appear to be?",
    "What would you record as the material of {thing} here?",
    "Judging by the image, {thing} is made of which material?",
    "Which of these materials matches {thing} in the photograph?",
    "The photo is supposed to show {thing}. What is it made of?",
]

INDOOR_QUESTIONS = [
    "Was this photograph taken indoors or outdoors?",
    "Is the scene in this picture inside or outside?",
    "Judging by the image, where is this: inside a building, or out in the open?",
    "Pick whether the photo is an indoor or an outdoor shot.",
    "Does this photo show an indoor setting or an outdoor one?",
    "Somebody has to tag this image indoor/outdoor. Which is it?",
    "From the photograph, is the subject inside or outside?",
    "Classify this picture as indoors or outdoors.",
    "Which describes the setting of the photograph?",
    "Is this an interior shot or an exterior one?",
]
SAME_SUBJECT_QUESTIONS = [
    "Image 1 is the reference and image 2 is the new photo. Do the two show the same kind of subject?",
    "Do these two photographs show the same sort of thing?",
    "Compare the two images. Is the subject of image 2 the same kind of thing as in image 1?",
    "Image 1 and image 2 were filed together. Do they really show the same kind of subject?",
    "Are both photographs pictures of the same kind of thing?",
    "Somebody paired these two images. Is the pairing right -- is it the same kind of subject?",
    "Looking at both photos, would you say they show the same sort of subject?",
    "Does image 2 show the same kind of subject as the reference image 1?",
    "Answer yes or no: the two images show the same kind of thing.",
    "Is the thing in image 2 of the same kind as the thing in image 1?",
]
WHICH_IMAGE_QUESTIONS = [
    "Which image shows {thing}?",
    "One of these photographs is supposed to contain {thing}. Which one does?",
    "Looking at image 1 and image 2, where does {thing} appear?",
    "In which of the two pictures can you see {thing}?",
    "Pick the image that contains {thing}.",
    "Somebody is looking for {thing}. Which image should they use?",
    "Where is {thing}: image 1, image 2, both, or neither?",
    "Which of the two photographs shows {thing}?",
    "Decide which image contains {thing}.",
    "{thing} -- which image is it in?",
]

WHICH_OPTIONS = [
    ("image 1", "only the first photograph shows it"),
    ("image 2", "only the second photograph shows it"),
    ("both images", "it appears in each of the two photographs"),
    ("neither image", "it does not appear in either photograph"),
]


# Every phrase written in this file already reads as English on its own.
AUTHORED_PHRASES = frozenset(COMMON_OBJECTS) | frozenset(CATEGORY_OBJECT.values())

# --------------------------------------------------------------------------- helpers
def _pick(rng, bank):
    """-> (index, item); the index becomes part of the template id so wording is auditable."""
    i = rng.randrange(len(bank))
    return i, bank[i]


def _options(values, rng, criteria=None, describe_share=0.33):
    """Turn a list of option strings into contract options, shuffled, sometimes with criteria."""
    values = list(dict.fromkeys(values))
    rng.shuffle(values)
    with_desc = criteria is not None and rng.random() < describe_share
    out = []
    for value in values:
        option = {"value": value[:128]}
        if with_desc and criteria.get(value):
            option["description"] = criteria[value]
        out.append(option)
    return out


def _levels(levels, rng, describe_share=1.0):
    del rng, describe_share  # ordinal levels always carry their description: the contract requires one
    return [{"value": value, "description": text} for value, text in levels]


def _field(kind, question, **rest):
    return {"id": "answer", "type": kind, "question": question, **rest}


# Upstream labels that read as mass nouns: "a food" and "a water" are wrong, "food" is right.
MASS_LIKE = {
    "food", "water", "hair", "fur", "skin", "wood", "metal", "plastic", "glass", "paper", "snow",
    "ice", "grass", "sand", "soil", "milk", "meat", "cheese", "bread", "rice", "pasta", "sugar",
    "salt", "coffee", "tea", "beer", "wine", "juice", "smoke", "fire", "dust", "clothing",
    "furniture", "packaging", "money", "cash", "jewellery", "jewelry", "hardware", "equipment",
    "machinery", "traffic", "vegetation", "foliage", "produce", "seafood", "fast food", "junk food",
    "street food", "baked goods", "dairy", "cutlery", "crockery", "luggage", "footwear",
}
_VOWEL = "aeiou"


def _thing_phrase(name, authored=None):
    """Open Images gives bare nouns ('Coffee table'); this makes them read as English.

    Phrases this file wrote already carry their determiner ("a chair", "people playing a sport")
    and pass through untouched; an upstream label gets "a"/"an" unless it is plural or a mass noun.
    """
    text = str(name).strip()
    if not text:
        return "the subject"
    if text[:1].isupper() and not text.isupper():
        text = text[0].lower() + text[1:]
    if text in (AUTHORED_PHRASES if authored is None else authored):
        return text
    first = text.split()[0]
    if first in {"a", "an", "the", "some", "people", "two", "three"}:
        return text
    if text in MASS_LIKE or text.split()[-1] in MASS_LIKE:
        return text
    if text.endswith("s") and not text.endswith(("ss", "us", "is")):
        return text  # already plural
    return ("an " if text[0] in _VOWEL else "a ") + text


def _plural(phrase):
    """A crude but readable plural for the counting questions ('a chair' -> 'chairs')."""
    word = phrase
    for article in ("a pair of ", "an ", "a ", "the "):
        if word.startswith(article):
            word = word[len(article):]
            break
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _fill(rng, pool, taken, count):
    """Take `count` fresh items from `pool` that are not already in `taken`."""
    available = [x for x in pool if x not in taken]
    rng.shuffle(available)
    return available[:count]


def _candidate_objects(inputs, rng, want, exclude=()):
    """Plausible object names: verified absences first (they are real, and really absent), then
    everyday objects.  Verified absences make the hardest honest distractors Open Images can give."""
    taken = set(exclude) | set(CATEGORY_ASSOCIATED.get(inputs.get("category", ""), ()))
    pool = [_thing_phrase(x) for x in inputs.get("absent", [])]
    rng.shuffle(pool)
    out = [x for x in pool if x not in taken][:want]
    taken |= set(out)
    if len(out) < want:
        out += _fill(rng, COMMON_OBJECTS, taken, want - len(out))
    return out


def _option_count(rng):
    """Mostly 2-12 options, about 10% of records with 13-25, as decision-v1 requires."""
    return rng.randint(13, 20) if rng.random() < 0.10 else rng.randint(4, 9)


# --------------------------------------------------------------------------- families
def make_scene_type(inputs, rng, unknown=False):
    """choice, 6-10 options.  `unknown=True` drops every scene plausible for this photo."""
    i, question = _pick(rng, SCENE_QUESTIONS)
    likely = CATEGORY_SCENES.get(inputs.get("category", ""), [])
    n = rng.randint(6, 10)
    if unknown:
        pool = [s for s in SCENES if s not in likely]
        rng.shuffle(pool)
        values = pool[:n]
    else:
        values = list(dict.fromkeys(likely))[:2]
        values += _fill(rng, SCENES, set(values), n - len(values))
    options = _options(values, rng, SCENE_CRITERIA)
    return [_field("choice", question, options=options)], f"scene_type/{i:02d}" + ("/unknown" if unknown else "")


def make_main_object(inputs, rng, unknown=False):
    """choice.  The candidate pool always mixes things that may be there with things verified absent."""
    i, question = _pick(rng, MAIN_OBJECT_QUESTIONS)
    n = _option_count(rng)
    present = [_thing_phrase(x) for x in inputs.get("present", [])]
    if unknown or not present:
        values = _candidate_objects(inputs, rng, n)
        unknown = True
    else:
        rng.shuffle(present)
        values = present[:1] + _candidate_objects(inputs, rng, n - 1, exclude=present[:1])
    criteria = {v: f"the photograph is mainly a picture of {v}" for v in values}
    return [_field("choice", question, options=_options(values, rng, criteria))], \
        f"main_object/{i:02d}" + ("/unknown" if unknown else "")


def make_object_present(inputs, rng, about_absent=False):
    """boolean.  About 40% of these ask about something verified (or believed) NOT to be there."""
    i, template = _pick(rng, OBJECT_PRESENT_QUESTIONS)
    present = [_thing_phrase(x) for x in inputs.get("present", [])]
    if about_absent or not present:
        thing = _candidate_objects(inputs, rng, 1, exclude=present)[0]
        about_absent = True
    else:
        thing = rng.choice(present)
    field = _field("boolean", template.format(thing=thing))
    if rng.random() < 0.30:
        field["yes_description"] = f"{thing} is clearly visible in the photograph"
        field["no_description"] = f"{thing} does not appear anywhere in the photograph"
    return [field], f"object_present/{i:02d}" + ("/absent" if about_absent else "")


def make_count_bucket(inputs, rng, about_absent=False):
    """ordinal, the none/1/2-3/4-6/many band."""
    i, template = _pick(rng, COUNT_QUESTIONS)
    present = [_thing_phrase(x) for x in inputs.get("present", [])]
    if about_absent or not present:
        thing = _candidate_objects(inputs, rng, 1, exclude=present)[0]
        about_absent = True
    else:
        thing = rng.choice(present)
    field = _field("ordinal", template.format(thing=_plural(thing)), levels=_levels(COUNT_LEVELS, rng))
    return [field], f"count_bucket/{i:02d}" + ("/absent" if about_absent else "")


def make_photo_quality(inputs, rng, unknown=False):
    """ordinal, 4 or 5 levels."""
    del inputs
    i, question = _pick(rng, QUALITY_QUESTIONS)
    levels = QUALITY_LEVELS if rng.random() < 0.7 else QUALITY_LEVELS_SHORT
    return [_field("ordinal", question, levels=_levels(levels, rng))], f"photo_quality/{i:02d}"


def make_colour_of_x(inputs, rng, unknown=False):
    """choice over colours, about something in the photo -- or, when `unknown`, about something that
    a human verified is NOT in the photo, so no colour can honestly be given."""
    i, template = _pick(rng, COLOUR_QUESTIONS)
    present = [_thing_phrase(x) for x in inputs.get("present", [])]
    if unknown or not present:
        thing = _candidate_objects(inputs, rng, 1, exclude=present)[0]
        unknown = True
    else:
        thing = rng.choice(present)
    n = rng.randint(4, 9) if rng.random() > 0.10 else rng.randint(13, 16)
    values = _fill(rng, COLOURS, set(), min(n, len(COLOURS)))
    options = _options(values, rng, COLOUR_CRITERIA)
    return [_field("choice", template.format(thing=thing), options=options)], \
        f"colour_of_x/{i:02d}" + ("/absent" if unknown else "")


def make_text_visible(inputs, rng, **_):
    """boolean."""
    del inputs
    i, question = _pick(rng, TEXT_VISIBLE_QUESTIONS)
    field = _field("boolean", question)
    if rng.random() < 0.30:
        field["yes_description"] = "at least one word, number or label in the frame can actually be read"
        field["no_description"] = "there is no text, or any text present is too small or blurred to read"
    return [field], f"text_visible/{i:02d}"


def make_material(inputs, rng, unknown=False):
    """choice over materials.  With `unknown=True` the question names something that is NOT in the
    photograph, so there is no honest material to give (a false-premise construction)."""
    n = rng.randint(4, 8)
    values = _fill(rng, MATERIALS, set(), n)
    if unknown:
        present = [_thing_phrase(x) for x in inputs.get("present", [])]
        thing = _candidate_objects(inputs, rng, 1, exclude=present)[0]
        i, template = _pick(rng, MATERIAL_OF_X_QUESTIONS)
        question, suffix = template.format(thing=thing), "/absent"
    else:
        i, question = _pick(rng, MATERIAL_QUESTIONS)
        suffix = ""
    return [_field("choice", question, options=_options(values, rng, MATERIAL_CRITERIA))], f"material/{i:02d}{suffix}"


def make_indoor_outdoor(inputs, rng, **_):
    """choice, 2-4 options."""
    del inputs
    i, question = _pick(rng, INDOOR_QUESTIONS)
    pairs = list(INDOOR_OPTIONS) if rng.random() < 0.7 else list(INDOOR_OPTIONS[:3])
    criteria = {value: text for value, text in pairs}
    return [_field("choice", question, options=_options([v for v, _ in pairs], rng, criteria))], f"indoor_outdoor/{i:02d}"


def make_two_image_same_subject(inputs, rng, **_):
    """boolean over a pair; image 1 is the reference, image 2 the new photo."""
    i, question = _pick(rng, SAME_SUBJECT_QUESTIONS)
    field = _field("boolean", question)
    if rng.random() < 0.30:
        field["yes_description"] = "both photographs show the same kind of subject"
        field["no_description"] = "the two photographs show different kinds of subject"
    return [field], f"two_image_same_subject/{i:02d}"


def make_two_image_which(inputs, rng, unknown=False):
    """choice: image 1 / image 2 / both images / neither image."""
    i, template = _pick(rng, WHICH_IMAGE_QUESTIONS)
    present = [_thing_phrase(x) for x in inputs.get("present", [])]
    second = inputs.get("second") or {}
    present2 = [_thing_phrase(x) for x in second.get("present", [])]
    if unknown or not (present or present2):
        thing = _candidate_objects(inputs, rng, 1, exclude=present + present2)[0]
        unknown = True
    else:
        thing = rng.choice(present or present2)
    criteria = {value: text for value, text in WHICH_OPTIONS}
    options = _options([v for v, _ in WHICH_OPTIONS], rng, criteria)
    return [_field("choice", template.format(thing=thing), options=options)], \
        f"two_image_which/{i:02d}" + ("/absent" if unknown else "")


MAKERS = {
    "scene_type": make_scene_type,
    "main_object": make_main_object,
    "object_present": make_object_present,
    "count_bucket": make_count_bucket,
    "photo_quality": make_photo_quality,
    "colour_of_x": make_colour_of_x,
    "text_visible": make_text_visible,
    "material": make_material,
    "indoor_outdoor": make_indoor_outdoor,
    "two_image_same_subject": make_two_image_same_subject,
    "two_image_which": make_two_image_which,
}

# Which families can be built so that the honest answer is `unknown` without inventing anything.
# Only these four constructions make the honest answer `unknown`:
#   scene_type/unknown   every scene plausible for the photo is removed from the options (not_listed)
#   main_object/unknown  no option names what the photo actually shows            (not_listed)
#   colour_of_x/absent   the question asks for the colour of something not there  (false_premise)
#   material/absent      the question asks what something not there is made of    (false_premise)
# `object_present/absent`, `count_bucket/absent` and `two_image_which` about a missing thing are
# NOT unknown constructions -- "no", "none at all" and "neither image" are honest, listed answers.
UNKNOWN_CAPABLE = {
    "scene_type": "unknown", "main_object": "unknown", "colour_of_x": "unknown", "material": "unknown",
}


def is_unknown_construction(family: str, template_id: str) -> bool:
    """True when the construction makes the honest answer `unknown`.

    Read from the template id rather than from the caller's intent, because a family can fall into
    an unknown construction by itself: `main_object` with no usable present label cannot list the
    right answer, whatever the caller asked for.
    """
    if template_id.endswith("/unknown"):
        return True
    return family in {"colour_of_x", "material"} and template_id.endswith("/absent")


def make(family, inputs, rng, unknown=False):
    """The single entry point the converter uses.  -> (fields, template_id)."""
    maker = MAKERS[family]
    keyword = UNKNOWN_CAPABLE.get(family)
    if unknown and keyword:
        return maker(inputs, rng, **{keyword: True})
    return maker(inputs, rng)


# --------------------------------------------------------------------------- state
STATE_KINDS = ("empty", "aligned", "irrelevant", "contradictory")

OTHER_SCENES = [
    ("a warehouse aisle", "pallet racking", "forklift"),
    ("a dentist's surgery", "a treatment chair", "an overhead lamp"),
    ("a ski slope", "snow", "a chairlift"),
    ("a fishing harbour", "nets", "a trawler"),
    ("a recording studio", "a mixing desk", "studio monitors"),
    ("a wheat field at harvest", "a combine harvester", "straw bales"),
    ("an underground car park", "concrete pillars", "parking bays"),
    ("a bakery counter", "trays of bread", "a price board"),
    ("a server room", "racks", "patch cables"),
    ("a greenhouse", "seedling trays", "irrigation pipes"),
]

IRRELEVANT_STATES = [
    {"session": {"locale": "en-GB", "channel": "mobile", "experiment": "B"}},
    {"queue": {"name": "overnight-batch", "depth": 42, "retries": 1}},
    {"account": {"tier": "standard", "opened": "2024-11-02", "region": "eu-west-1"}},
    {"shipment": {"carrier": "DPD", "weight_kg": 3.4, "status": "in transit"}},
    {"ticket": {"priority": "P3", "assigned": False, "labels": ["triage", "backlog"]}},
    {"build": {"commit": "9f2c1ab", "duration_s": 213, "warnings": 6}},
    {"weather_api": {"station": "EGLL", "temperature_c": 11.2, "wind_kt": 14}},
    {"cart": {"items": 3, "currency": "EUR", "promo_applied": False}},
]


def make_state(inputs, rng, kind=None, as_string=None):
    """-> (state, kind).  `aligned` describes the same photo, `contradictory` describes a different
    scene entirely, `irrelevant` is unrelated machine state, `empty` is `{}`.

    ~40% of non-empty states are rendered as a free-text string, as the spec asks.
    """
    if kind is None:
        kind = rng.choices(STATE_KINDS, weights=(25, 30, 25, 20))[0]
    if kind == "empty":
        return {}, "empty"

    category = str(inputs.get("category", "") or "item").replace("_", " ")
    title = str(inputs.get("title", "") or "").strip()[:120]

    if kind == "aligned":
        payload = {
            "record": {
                "reference": str(inputs.get("id", ""))[:64],
                "filed_under": category,
                "caption": title or f"photograph of {category}",
                "submitted_by": "field agent",
            }
        }
        text = (f"The record this photo is attached to is filed under \"{category}\""
                + (f" with the caption \"{title}\"." if title else "."))
    elif kind == "irrelevant":
        payload = dict(rng.choice(IRRELEVANT_STATES))
        first = next(iter(payload))
        text = f"Unrelated system state: {first} = " + ", ".join(f"{k}: {v}" for k, v in payload[first].items()) + "."
    else:  # contradictory: the state describes a DIFFERENT scene than the photograph
        scene, detail_a, detail_b = rng.choice(OTHER_SCENES)
        payload = {
            "record": {
                "reference": str(inputs.get("id", ""))[:64],
                "claimed_scene": scene,
                "claimed_contents": [detail_a, detail_b],
                "note": "the attached photograph is supposed to match this description",
            }
        }
        text = (f"The paperwork says this photograph shows {scene}, with {detail_a} and {detail_b} "
                f"in the frame.")
    if as_string is None:
        as_string = rng.random() < 0.40 / 0.75  # ~40% of ALL states, given `empty` is always an object
    return (text if as_string else payload), kind
