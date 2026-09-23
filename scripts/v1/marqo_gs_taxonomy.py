"""Product-category lexicon for the Marqo-GS-10M converter.

Marqo-GS ships no category column — the research doc's "2.4k leaf categories" are in
fact the *search queries*, long-tail natural-language phrases ("Breathable full slips
for summer", "Fashion Forward Trendsetting Wrap Designs") that no single photo
determines. The usable category signal is the product **title**, which states what the
thing is: "Ugg Women's Shearling Earmuffs - Black".

So: match curated phrases against the title, longest phrase first, and accept the
listing only when exactly one canonical category survives. `phrase -> (label, group)`.
Groups give sibling (hard) distractors.
"""

LEXICON = {}


def _add(group, mapping):
    for phrase, label in mapping.items():
        LEXICON[phrase] = (label, group)


_add("footwear", {
    "running shoes": "running shoes", "running shoe": "running shoes",
    "sneakers": "sneakers", "sneaker": "sneakers", "trainers": "sneakers",
    "boots": "boots", "boot": "boots", "rain boots": "rain boots",
    "snow boots": "snow boots", "ankle boots": "ankle boots",
    "cowboy boots": "cowboy boots", "hiking boots": "hiking boots",
    "sandals": "sandals", "sandal": "sandals", "flip flops": "flip flops",
    "slippers": "slippers", "slipper": "slippers", "clogs": "clogs",
    "loafers": "loafers", "loafer": "loafers", "oxfords": "oxford shoes",
    "heels": "high heels", "high heels": "high heels", "pumps": "pumps",
    "wedges": "wedge shoes", "flats": "flat shoes", "ballet flats": "flat shoes",
    "espadrilles": "espadrilles", "mules": "mules", "moccasins": "moccasins",
    "cleats": "cleats", "water shoes": "water shoes", "insoles": "shoe insoles",
    "shoelaces": "shoelaces",
})
_add("apparel_top", {
    "t-shirt": "t-shirt", "tee shirt": "t-shirt", "tshirt": "t-shirt",
    "polo shirt": "polo shirt", "dress shirt": "dress shirt",
    "blouse": "blouse", "tank top": "tank top", "camisole": "camisole",
    "sweater": "sweater", "pullover": "pullover", "cardigan": "cardigan",
    "hoodie": "hoodie", "sweatshirt": "sweatshirt", "crop top": "crop top",
    "tunic": "tunic", "bodysuit": "bodysuit", "henley": "henley shirt",
    "flannel shirt": "flannel shirt", "jersey top": "sports jersey",
})
_add("apparel_outer", {
    "jacket": "jacket", "coat": "coat", "parka": "parka", "blazer": "blazer",
    "trench coat": "trench coat", "puffer jacket": "puffer jacket",
    "down jacket": "down jacket", "raincoat": "raincoat",
    "windbreaker": "windbreaker", "vest": "vest", "poncho": "poncho",
    "shawl": "shawl", "bomber jacket": "bomber jacket",
    "denim jacket": "denim jacket", "fleece jacket": "fleece jacket",
})
_add("apparel_bottom", {
    "jeans": "jeans", "pants": "trousers", "trousers": "trousers",
    "chinos": "chinos", "shorts": "shorts", "leggings": "leggings",
    "joggers": "joggers", "sweatpants": "sweatpants", "skirt": "skirt",
    "overalls": "overalls", "cargo pants": "cargo trousers",
    "yoga pants": "yoga pants", "capris": "capri pants",
})
_add("apparel_whole", {
    "dress": "dress", "gown": "gown", "jumpsuit": "jumpsuit",
    "romper": "romper", "suit": "suit", "kaftan": "kaftan",
    "sundress": "sundress", "maxi dress": "maxi dress",
    "wedding dress": "wedding dress",
})
_add("underwear_sleep", {
    "bra": "bra", "bralette": "bralette", "panties": "panties",
    "briefs": "briefs", "boxers": "boxer shorts", "underwear": "underwear",
    "socks": "socks", "tights": "tights", "stockings": "stockings",
    "pantyhose": "pantyhose", "pajamas": "pyjamas", "pajama set": "pyjamas",
    "pyjamas": "pyjamas", "nightgown": "nightgown", "robe": "robe",
    "bathrobe": "bathrobe", "shapewear": "shapewear",
    "swimsuit": "swimsuit", "bikini": "bikini", "swim trunks": "swim trunks",
    "one piece swimsuit": "swimsuit",
})
_add("worn_accessory", {
    "hat": "hat", "cap": "cap", "beanie": "beanie", "visor": "visor",
    "fedora": "fedora", "sun hat": "sun hat", "earmuffs": "earmuffs",
    "earmuff": "earmuffs", "scarf": "scarf", "gloves": "gloves",
    "mittens": "mittens", "belt": "belt", "necktie": "necktie",
    "bow tie": "bow tie", "suspenders": "suspenders", "headband": "headband",
    "sunglasses": "sunglasses", "eyeglasses": "eyeglasses",
    "reading glasses": "reading glasses", "umbrella": "umbrella",
    "bandana": "bandana", "hair clip": "hair clip", "wig": "wig",
})
_add("jewellery", {
    "necklace": "necklace", "pendant": "pendant", "bracelet": "bracelet",
    "bangle": "bangle", "earrings": "earrings", "earring": "earrings",
    "studs": "stud earrings", "hoop earrings": "hoop earrings",
    "ring": "ring", "engagement ring": "engagement ring",
    "wedding band": "wedding band", "anklet": "anklet", "brooch": "brooch",
    "cufflinks": "cufflinks", "locket": "locket",
    "watch": "watch", "smartwatch": "smartwatch", "nose stud": "nose stud",
    "jewelry box": "jewellery box",
})
_add("bags", {
    "handbag": "handbag", "purse": "purse", "tote bag": "tote bag",
    "backpack": "backpack", "crossbody bag": "crossbody bag",
    "shoulder bag": "shoulder bag", "clutch": "clutch bag",
    "wallet": "wallet", "duffel bag": "duffel bag", "suitcase": "suitcase",
    "luggage": "luggage", "briefcase": "briefcase", "fanny pack": "fanny pack",
    "diaper bag": "diaper bag", "lunch bag": "lunch bag",
    "laptop bag": "laptop bag", "makeup bag": "makeup bag",
    "gym bag": "gym bag",
})
_add("furniture", {
    "chair": "chair", "armchair": "armchair", "recliner": "recliner",
    "sofa": "sofa", "couch": "sofa", "loveseat": "loveseat",
    "stool": "stool", "bar stool": "bar stool", "bench": "bench",
    "ottoman": "ottoman", "table": "table", "coffee table": "coffee table",
    "dining table": "dining table", "side table": "side table",
    "desk": "desk", "dresser": "dresser", "nightstand": "nightstand",
    "bookcase": "bookcase", "cabinet": "cabinet", "wardrobe": "wardrobe",
    "bed frame": "bed frame", "headboard": "headboard", "mattress": "mattress",
    "futon": "futon", "crib": "crib", "shelf": "shelf",
    "shelves": "shelf", "shelving unit": "shelf", "console table": "console table",
})
_add("home_textile", {
    "rug": "rug", "area rug": "rug", "carpet": "carpet", "doormat": "doormat",
    "curtain": "curtain", "curtains": "curtain", "blinds": "blinds",
    "pillow": "pillow", "cushion": "cushion", "throw pillow": "throw pillow",
    "blanket": "blanket", "throw blanket": "throw blanket",
    "comforter": "comforter", "duvet": "duvet", "quilt": "quilt",
    "sheet set": "bed sheet set", "bed sheet": "bed sheet set",
    "mattress protector": "mattress protector", "towel": "towel",
    "bath mat": "bath mat", "shower curtain": "shower curtain",
    "tablecloth": "tablecloth", "placemat": "placemat", "napkin": "napkin",
    "apron": "apron",
})
_add("home_decor", {
    "lamp": "lamp", "table lamp": "table lamp", "floor lamp": "floor lamp",
    "chandelier": "chandelier", "pendant light": "pendant light",
    "string lights": "string lights", "candle": "candle",
    "candle holder": "candle holder", "vase": "vase", "mirror": "mirror",
    "picture frame": "picture frame", "photo frame": "picture frame",
    "wall art": "wall art", "canvas print": "canvas print",
    "clock": "clock", "wall clock": "wall clock", "figurine": "figurine",
    "planter": "planter", "flower pot": "flower pot", "wreath": "wreath",
    "wind chime": "wind chime", "doorbell": "doorbell",
})
_add("storage", {
    "storage box": "storage box", "storage bin": "storage bin",
    "storage basket": "storage basket", "basket": "basket",
    "organizer": "organiser", "drawer organizer": "drawer organiser",
    "shoe rack": "shoe rack", "coat rack": "coat rack",
    "spice rack": "spice rack", "wine rack": "wine rack",
    "hanger": "clothes hanger", "hangers": "clothes hanger",
    "laundry basket": "laundry basket", "laundry hamper": "laundry hamper",
    "trash can": "trash can", "toolbox": "tool box",
})
_add("kitchen", {
    "frying pan": "frying pan", "skillet": "skillet", "saucepan": "saucepan",
    "stock pot": "stock pot", "dutch oven": "dutch oven",
    "baking sheet": "baking sheet", "cake pan": "cake pan",
    "muffin pan": "muffin pan", "mixing bowl": "mixing bowl",
    "colander": "colander", "cutting board": "cutting board",
    "knife": "kitchen knife", "knife set": "knife set",
    "spatula": "spatula", "whisk": "whisk", "ladle": "ladle",
    "measuring cup": "measuring cup", "rolling pin": "rolling pin",
    "mug": "mug", "tumbler": "tumbler", "water bottle": "water bottle",
    "wine glass": "wine glass", "drinking glass": "drinking glass",
    "plate set": "dinner plate set", "dinnerware set": "dinnerware set",
    "bowl set": "bowl set", "cutlery set": "cutlery set",
    "teapot": "teapot", "kettle": "kettle", "french press": "french press",
    "coffee maker": "coffee maker", "espresso machine": "espresso machine",
    "blender": "blender", "toaster": "toaster", "air fryer": "air fryer",
    "slow cooker": "slow cooker", "pressure cooker": "pressure cooker",
    "food processor": "food processor", "stand mixer": "stand mixer",
    "waffle maker": "waffle maker", "rice cooker": "rice cooker",
    "juicer": "juicer", "microwave": "microwave oven",
    "lunch box": "lunch box", "thermos": "vacuum flask",
})
_add("electronics", {
    "headphones": "headphones", "earbuds": "earbuds", "headset": "headset",
    "speaker": "speaker", "bluetooth speaker": "bluetooth speaker",
    "soundbar": "soundbar", "phone case": "phone case",
    "screen protector": "screen protector", "power bank": "power bank",
    "charger": "charger", "charging cable": "charging cable",
    "laptop": "laptop", "tablet": "tablet", "monitor": "computer monitor",
    "keyboard": "computer keyboard", "mouse pad": "mouse pad",
    "webcam": "webcam", "router": "wifi router", "printer": "printer",
    "camera": "camera", "tripod": "tripod", "drone": "drone",
    "flash drive": "usb flash drive", "hard drive": "hard drive",
    "remote control": "remote control", "projector": "projector",
    "television": "television", "game controller": "game controller",
    "fitness tracker": "fitness tracker",
})
_add("beauty", {
    "lipstick": "lipstick", "mascara": "mascara", "foundation": "foundation",
    "eyeshadow palette": "eyeshadow palette", "nail polish": "nail polish",
    "perfume": "perfume", "shampoo": "shampoo", "conditioner": "conditioner",
    "face cream": "face cream", "sunscreen": "sunscreen",
    "face mask sheet": "face mask sheet", "makeup brush": "makeup brush",
    "hair dryer": "hair dryer", "hair straightener": "hair straightener",
    "curling iron": "curling iron", "razor": "razor",
    "electric shaver": "electric shaver", "toothbrush": "toothbrush",
    "hair brush": "hair brush", "comb": "comb", "tweezers": "tweezers",
})
_add("sport_outdoor", {
    "yoga mat": "yoga mat", "dumbbell": "dumbbell", "kettlebell": "kettlebell",
    "resistance band": "resistance band", "jump rope": "jump rope",
    "foam roller": "foam roller", "treadmill": "treadmill",
    "exercise bike": "exercise bike", "bicycle": "bicycle",
    "helmet": "helmet", "tent": "tent", "sleeping bag": "sleeping bag",
    "backpacking pack": "hiking backpack", "cooler": "cooler box",
    "fishing rod": "fishing rod", "golf club": "golf club",
    "basketball": "basketball", "soccer ball": "football",
    "skateboard": "skateboard", "scooter": "scooter",
    "swim goggles": "swimming goggles", "life jacket": "life jacket",
})
_add("baby_kids", {
    "stroller": "stroller", "car seat": "child car seat",
    "high chair": "high chair", "baby carrier": "baby carrier",
    "pacifier": "pacifier", "baby bottle": "baby bottle", "bib": "bib",
    "onesie": "onesie", "swaddle": "swaddle", "diaper": "diaper",
    "playpen": "playpen", "baby monitor": "baby monitor",
    "teething toy": "teething toy", "plush toy": "plush toy",
    "building blocks": "building blocks", "puzzle": "jigsaw puzzle",
    "board game": "board game", "doll": "doll",
})
_add("pet", {
    "dog bed": "dog bed", "pet bed": "pet bed", "dog collar": "dog collar",
    "dog leash": "dog leash", "pet carrier": "pet carrier",
    "litter box": "litter box", "scratching post": "scratching post",
    "pet bowl": "pet bowl", "dog toy": "dog toy", "cat tree": "cat tree",
    "aquarium": "aquarium",
})
_add("tools_hardware", {
    "drill": "power drill", "screwdriver": "screwdriver", "wrench": "wrench",
    "hammer": "hammer", "pliers": "pliers", "tape measure": "tape measure",
    "sander": "sander",
    "ladder": "ladder", "flashlight": "flashlight", "extension cord": "extension cord",
    "door handle": "door handle", "door knob": "door knob", "hinge": "hinge",
    "padlock": "padlock", "faucet": "faucet", "showerhead": "shower head",
    "toilet seat": "toilet seat", "garden hose": "garden hose",
    "watering can": "watering can", "lawn mower": "lawn mower",
    "vacuum cleaner": "vacuum cleaner", "air purifier": "air purifier",
    "humidifier": "humidifier", "space heater": "space heater",
    "electric fan": "electric fan",
})
_add("stationery", {
    "notebook": "notebook", "journal": "journal", "planner": "planner",
    "pen": "pen", "pencil": "pencil", "marker": "marker pen",
    "sticker": "sticker", "greeting card": "greeting card",
    "gift bag": "gift bag", "wrapping paper": "wrapping paper",
    "binder": "ring binder", "backpack for school": "school backpack",
    "calendar": "calendar", "bookmark": "bookmark",
})

# labels a product photo does not reliably separate: never each other's distractor
CONFUSABLE = [
    {"sneakers", "running shoes", "cleats"},
    {"boots", "ankle boots", "snow boots", "rain boots", "hiking boots", "cowboy boots"},
    {"high heels", "pumps", "wedge shoes"},
    {"flat shoes", "loafers", "moccasins", "espadrilles", "mules"},
    {"slippers", "clogs"},
    {"sandals", "flip flops"},
    {"sweater", "pullover", "sweatshirt"},
    {"hoodie", "sweatshirt"},
    {"jacket", "coat", "parka", "windbreaker", "raincoat", "puffer jacket", "down jacket",
     "bomber jacket", "fleece jacket", "blazer", "trench coat"},
    {"trousers", "chinos", "joggers", "sweatpants", "cargo trousers", "yoga pants", "leggings"},
    {"dress", "gown", "sundress", "maxi dress", "wedding dress"},
    {"t-shirt", "polo shirt", "dress shirt", "henley shirt", "flannel shirt", "blouse"},
    {"tank top", "camisole", "crop top"},
    {"pyjamas", "nightgown", "robe", "bathrobe"},
    {"panties", "briefs", "boxer shorts", "underwear"},
    {"socks", "tights", "stockings", "pantyhose"},
    {"hat", "cap", "beanie", "visor", "fedora", "sun hat"},
    {"necklace", "pendant", "locket"},
    {"bracelet", "bangle", "anklet"},
    {"earrings", "stud earrings", "hoop earrings"},
    {"ring", "engagement ring", "wedding band"},
    {"watch", "smartwatch", "fitness tracker"},
    {"handbag", "purse", "tote bag", "shoulder bag", "crossbody bag", "clutch bag"},
    {"backpack", "hiking backpack", "school backpack", "gym bag", "duffel bag"},
    {"suitcase", "luggage", "briefcase"},
    {"chair", "armchair", "recliner"},
    {"sofa", "loveseat", "futon"},
    {"stool", "bar stool", "ottoman", "bench"},
    {"table", "coffee table", "dining table", "side table", "console table", "desk"},
    {"dresser", "nightstand", "cabinet", "wardrobe", "bookcase", "shelf"},
    {"rug", "carpet", "doormat", "bath mat"},
    {"pillow", "cushion", "throw pillow"},
    {"blanket", "throw blanket", "comforter", "duvet", "quilt"},
    {"lamp", "table lamp", "floor lamp"},
    {"chandelier", "pendant light"},
    {"clock", "wall clock"},
    {"picture frame", "wall art", "canvas print", "mirror"},
    {"vase", "planter", "flower pot"},
    {"storage box", "storage bin", "storage basket", "basket", "laundry basket",
     "laundry hamper"},
    {"frying pan", "skillet", "saucepan", "stock pot", "dutch oven"},
    {"baking sheet", "cake pan", "muffin pan"},
    {"mug", "tumbler", "vacuum flask", "water bottle", "drinking glass", "wine glass"},
    {"coffee maker", "espresso machine", "french press", "kettle", "teapot"},
    {"blender", "food processor", "juicer", "stand mixer"},
    {"air fryer", "slow cooker", "pressure cooker", "rice cooker", "microwave oven"},
    {"headphones", "earbuds", "headset"},
    {"speaker", "bluetooth speaker", "soundbar"},
    {"laptop", "tablet", "computer monitor", "television"},
    {"charger", "charging cable", "power bank"},
    {"shampoo", "conditioner"},
    {"face cream", "sunscreen", "foundation"},
    {"hair dryer", "hair straightener", "curling iron"},
    {"razor", "electric shaver"},
    {"hair brush", "comb", "makeup brush"},
    {"dumbbell", "kettlebell"},
    {"bicycle", "exercise bike", "scooter"},
    {"pet bed", "dog bed"},
    {"notebook", "journal", "planner", "calendar"},
    {"pen", "pencil", "marker pen"},
    {"screwdriver", "wrench", "pliers", "hammer"},
    {"electric fan", "space heater", "air purifier", "humidifier"},
]
