"""Attribute typing for convert_vg_attributes.py.

Visual Genome attributes are free-text, so a question can only be well posed if the answer
comes from a set whose members are *mutually exclusive*: exactly one of them can be true of
one object at one time.  Each entry below is such a set (a "dimension"); distractors for a
gold attribute are drawn only from the dimension the gold belongs to, and never from the same
synonym group as the gold.
"""

# family -> list of (dimension name, [synonym group, ...]); one group = one answer concept
DIMENSIONS: dict[str, list[tuple[str, list[list[str]]]]] = {
    "color": [("color", [
        ["white"], ["black"], ["blue"], ["green"], ["red"], ["brown"], ["yellow"],
        ["gray", "grey"], ["silver"], ["orange"], ["pink"], ["purple"], ["tan"],
        ["gold", "golden"], ["beige"], ["blonde", "blond"], ["maroon"], ["teal"],
        ["turquoise"], ["navy", "navy blue"], ["olive"], ["burgundy"], ["cream"],
        ["khaki"], ["ivory"], ["lavender"], ["peach"], ["bronze"], ["copper"],
        ["light brown"], ["dark brown"], ["light blue"], ["dark blue"], ["light green"],
        ["dark green"], ["dark red"], ["light gray", "light grey"], ["dark gray", "dark grey"],
        ["black and white"], ["red and white"], ["blue and white"], ["green and white"],
        ["multicolored", "multi-colored", "multi colored", "colorful"],
    ])],
    "material": [("material", [
        ["wooden", "wood", "hardwood", "made of wood"],
        ["metal", "metallic", "steel", "stainless steel", "iron", "chrome", "aluminum", "made of metal"],
        ["glass", "made of glass"],
        ["plastic", "made of plastic"],
        ["brick", "bricks", "made of bricks", "made of brick"],
        ["concrete", "cement"],
        ["stone", "rock", "granite"],
        ["marble"],
        ["leather"],
        ["paper"],
        ["cardboard"],
        ["ceramic", "porcelain"],
        ["rubber"],
        ["wicker", "rattan"],
        ["tile", "tiled"],
        ["cloth", "fabric", "cotton"],
        ["denim"],
        ["wool", "woolen"],
        ["straw"],
        ["bamboo"],
        ["clay"],
        ["canvas"],
        ["mesh", "wire"],
        ["carpeted", "carpet"],
    ])],
    "shape": [("shape", [
        ["round", "circular", "rounded", "spherical"],
        ["square", "rectangular", "rectangle"],
        ["oval"],
        ["triangular"],
        ["curved"],
        ["straight"],
        ["flat"],
        ["pointy", "pointed", "sharp"],
        ["arched"],
        ["cylindrical"],
        ["bent", "crooked"],
        ["octagonal"],
        ["heart shaped"],
        ["star shaped"],
        ["diamond shaped"],
        ["hexagonal"],
        ["wavy"],
        ["twisted"],
        ["slanted", "tilted"],
    ])],
    "state": [
        ("openness", [["open", "opened"], ["closed", "shut"]]),
        ("cleanliness", [["clean"], ["dirty", "filthy"], ["dusty"], ["muddy"]]),
        ("wetness", [["wet", "damp"], ["dry"]]),
        ("power", [["on", "lit", "illuminated", "turned on", "lit up"],
                   ["off", "unlit", "turned off"]]),
        ("fullness", [["empty"], ["full"], ["half full"]]),
        ("cooked", [["cooked"], ["raw", "uncooked"], ["burnt", "burned"], ["sliced"], ["whole"]]),
        ("ripeness", [["ripe"], ["unripe", "green"], ["rotten", "spoiled"]]),
        ("temperature_state", [["frozen"], ["melted"], ["melting"]]),
        ("age", [["young"], ["old", "elderly"], ["baby"], ["adult"]]),
        ("posture", [["standing"], ["sitting", "seated", "sitting down"], ["walking"],
                     ["running"], ["laying", "lying", "laying down", "lying down"],
                     ["jumping"], ["kneeling"], ["crouching", "squatting"], ["leaning"],
                     ["bending", "bent over"]]),
        ("vehicle_motion", [["parked"], ["moving", "driving", "in motion"], ["stopped"]]),
        ("air_state", [["flying", "airborne", "in the air"], ["perched"], ["landed"],
                       ["floating"], ["swimming"]]),
        ("doing", [["grazing"], ["sleeping"], ["resting"], ["eating"], ["drinking"],
                   ["playing"], ["riding"], ["surfing"], ["skiing"], ["skateboarding"],
                   ["snowboarding"], ["climbing"], ["reading"], ["cooking"], ["talking"],
                   ["smiling"], ["waving"], ["swinging"], ["skating"]]),
    ],
}

TEMPLATES: dict[str, list[str]] = {
    "color": [
        "What color is the {obj}?",
        "What is the color of the {obj}?",
        "The {obj} is what color?",
        "Which color best describes the {obj} in this photo?",
        "Looking at the {obj}, what colour is it?",
        "In this image, what color is the {obj}?",
    ],
    "material": [
        "What is the {obj} made of?",
        "What material is the {obj}?",
        "Which material is the {obj} made from?",
        "The {obj} appears to be made of what?",
        "What does the {obj} seem to be made out of?",
        "Judging by the photo, what is the {obj} made from?",
    ],
    "shape": [
        "What shape is the {obj}?",
        "What is the shape of the {obj}?",
        "Which shape best describes the {obj}?",
        "The {obj} has what shape?",
        "How would you describe the shape of the {obj}?",
        "In this picture, what shape is the {obj}?",
    ],
    "state": [
        "Which of these best describes the {obj} in this image?",
        "What is the state of the {obj}?",
        "In this photo, the {obj} is which of the following?",
        "How would you describe the {obj} here?",
        "Which option describes the {obj} shown in the picture?",
        "The {obj} in this image is what?",
    ],
}


def build_index():
    """attribute string -> (family, dimension, canonical group value)."""
    index: dict[str, tuple[str, str, str]] = {}
    groups: dict[tuple[str, str], list[str]] = {}
    for family, dims in DIMENSIONS.items():
        for dim, gs in dims:
            groups[(family, dim)] = [g[0] for g in gs]
            for g in gs:
                for word in g:
                    index.setdefault(word, (family, dim, g[0]))
    return index, groups
