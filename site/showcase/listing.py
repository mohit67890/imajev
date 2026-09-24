import json, requests

URL = "http://127.0.0.1:8765/v1/systemone"

listing = {
    "title": "Men's suede boat shoes",
    "color": "red",
    "product_type": "shoe",
}

questions = {
    "contradicted_field": {
        "type": "choice",
        "instructions":
            "Which field of `listing` does this photo contradict?",
        "criteria": {
            "listing.color": None,
            "listing.product_type": None,
            "none of these": "the photo agrees with every field",
        },
    },
    "color_matches": {
        "type": "noul",
        "instructions":
            "The product in the photo matches `listing.color`.",
    },
    "type_matches": {
        "type": "noul",
        "instructions": "The photo shows the kind of product "
                        "given in `listing.product_type`.",
    },
}

request = {"state": {"listing": listing}, "questions": questions}
with open("listing.jpg", "rb") as photo:
    r = requests.post(URL, files={"image": photo},
                      data={"request": json.dumps(request)})
print(json.dumps(r.json(), indent=2))
