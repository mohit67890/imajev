"""Write the decision-v2 licence evidence receipts.

Each receipt names one evidence file, its sha256, the SPDX id it establishes, the primary-source
URL it was fetched from, and the exact sentence in it that grants commercial use.  `scope` says
what the grant covers.  `commercial_use_reviewed` is set to true only where the quoted sentence is
actually in the stored file -- this script re-reads the file and fails if the quote is missing.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L = ROOT / "data" / "decision-v2" / "licenses"
REVIEWER = "Claude v2 collection agent"
REVIEWED_AT = "2026-09-23"

CC_BY_GRANT = ("Share — copy and redistribute the material in any medium or format for any purpose, "
               "even commercially.")
CC0_GRANT = ("You can copy, modify, distribute and perform the work, even for commercial purposes, "
             "all without asking permission.")

RECEIPTS = [
    # (evidence path, spdx, source_url, quoted grant, scope, notes)
    ("commons_photos/CC0-1.0.deed.html", "CC0-1.0",
     "https://creativecommons.org/publicdomain/zero/1.0/", CC0_GRANT,
     "Wikimedia Commons files whose imageinfo extmetadata LicenseShortName is 'CC0' and whose machine-readable License tag is 'cc0'",
     "Per-file licence read from the Commons API for every file admitted; see commons-api-license-samples.json for verbatim API responses."),
    ("commons_photos/CC-BY-3.0.deed.html", "CC-BY-3.0",
     "https://creativecommons.org/licenses/by/3.0/", CC_BY_GRANT,
     "Wikimedia Commons files whose LicenseShortName is 'CC BY 3.0' and whose License tag is 'cc-by-3.0'",
     "Attribution is required: author, licence and file page are kept per file in data/decision-v2/commons_photos/attribution.jsonl."),
    ("commons_photos/CC-BY-4.0.deed.html", "CC-BY-4.0",
     "https://creativecommons.org/licenses/by/4.0/", CC_BY_GRANT,
     "Wikimedia Commons files whose LicenseShortName is 'CC BY 4.0' and whose License tag is 'cc-by-4.0'",
     "Attribution is required: author, licence and file page are kept per file in data/decision-v2/commons_photos/attribution.jsonl."),
    ("openimages_v2/CC-BY-2.0.deed.html", "CC-BY-2.0",
     "https://creativecommons.org/licenses/by/2.0/", CC_BY_GRANT,
     "the Open Images V7 photographs themselves (Flickr originals), which the dataset lists as CC BY 2.0",
     "CC-BY-2.0 is NOT in scripts/v1_text/common.py:COMMERCIAL_ALLOWLIST; decision-v2 widens the list to include it "
     "(scripts/v2/common.py:V2_COMMERCIAL_ALLOWLIST) because 2.0 is attribution-only with no NC, ND or SA term. "
     "Google's own note disclaims any warranty about each individual photo's licence status -- see "
     "openimages-factsfigures_v7.html and the source README."),
    ("openimages_v2/CC-BY-4.0.deed.html", "CC-BY-4.0",
     "https://creativecommons.org/licenses/by/4.0/", CC_BY_GRANT,
     "the Open Images V7 annotations (image-level labels), which are used only to select diverse photos and to build present/absent candidates, never as targets",
     "Quoted grant for the annotations: 'The annotations are licensed by Google LLC under CC BY 4.0 license.' "
     "(openimages-factsfigures_v7.html)."),
]

# Files kept as supporting evidence but which do not by themselves establish an SPDX id.
PROVENANCE = [
    ("commons_photos/CC0-1.0.legalcode.txt", "https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt",
     "Full CC0 1.0 legal code, the operative text behind the deed."),
    ("commons_photos/CC-BY-3.0.legalcode.html", "https://creativecommons.org/licenses/by/3.0/legalcode.en",
     "Full CC BY 3.0 Unported legal code. It contains no NonCommercial, NoDerivatives or ShareAlike term."),
    ("commons_photos/CC-BY-4.0.legalcode.txt", "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
     "Full CC BY 4.0 International legal code. It contains no NonCommercial, NoDerivatives or ShareAlike term."),
    ("commons_photos/commons-licensing-policy.html", "https://commons.wikimedia.org/wiki/Commons:Licensing",
     "Commons' own licensing policy. Quoted verbatim: 'Wikimedia Commons only accepts free content, that is, images "
     "and other media files that are not subject to copyright restrictions which would prevent them being used by "
     "anyone, anytime, for any purpose.' and 'Media licensed exclusively under non-commercial only licenses (like "
     "CC BY-NC-SA) are not accepted either.' This is background: the admission decision for every single file in "
     "this source was made from that file's own API licence fields, not from this page.",
     "Wikimedia Commons only accepts free content, that is, images and other media files that are not subject to "
     "copyright restrictions which would prevent them being used by anyone, anytime, for any purpose."),
    ("openimages_v2/CC-BY-2.0.legalcode.html", "https://creativecommons.org/licenses/by/2.0/legalcode.en",
     "Full CC BY 2.0 Generic legal code. It contains no NonCommercial, NoDerivatives or ShareAlike term."),
    ("openimages_v2/CC-BY-4.0.legalcode.txt", "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
     "Full CC BY 4.0 International legal code."),
    ("openimages_v2/openimages-factsfigures_v7.html", "https://storage.googleapis.com/openimages/web/factsfigures_v7.html",
     "Open Images V7 licence statement, quoted verbatim: 'The annotations are licensed by Google LLC under CC BY 4.0 "
     "license. The images are listed as having a CC BY 2.0 license. Note: while we tried to identify images that are "
     "licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the "
     "license status of each image and you should verify the license for each image yourself.'",
     "The annotations are licensed by Google LLC under CC BY 4.0 license. The images are listed as having a CC BY "
     "2.0 license."),
    ("openimages_v2/openimages-download_v7.html", "https://storage.googleapis.com/openimages/web/download_v7.html",
     "Open Images V7 download page: the source of the CSV URLs used, and of the CVDF image mirror."),
]


def flatten(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix in (".html", ".htm"):
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", html.unescape(text))
    # wiki and deed markup leaves a space in front of punctuation once the tags are stripped;
    # drop it so a verbatim quote can be matched against the stored file
    return re.sub(r"\s+([,.;:])", r"\1", text)


def main() -> int:
    problems = []
    for rel, spdx, url, quote, scope, notes in RECEIPTS:
        path = L / rel
        if not path.is_file():
            problems.append(f"missing evidence {rel}")
            continue
        body = flatten(path)
        # compare with normalised dashes so a typographic em/en dash does not fake a failure
        norm = lambda s: re.sub(r"[‐-―−]", "-", s)
        verified = norm(quote) in norm(body)
        if not verified:
            problems.append(f"quoted grant not found in {rel}")
        receipt = {
            "spdx": spdx,
            "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "commercial_use_reviewed": bool(verified),
            "source_url": url,
            "reviewed_at": REVIEWED_AT,
            "reviewed_by": REVIEWER,
            "scope": scope,
            "quoted_grant": quote,
            "notes": notes,
        }
        path.with_name(path.name + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"receipt {rel} spdx={spdx} commercial_use_reviewed={verified}")
    for entry in PROVENANCE:
        rel, url, note = entry[0], entry[1], entry[2]
        quote = entry[3] if len(entry) > 3 else None
        path = L / rel
        if not path.is_file():
            problems.append(f"missing provenance {rel}")
            continue
        quote_found = None
        if quote is not None:
            quote_found = quote in flatten(path)
            if not quote_found:
                problems.append(f"quote not found in {rel}")
        payload = {
            "source_url": url,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "fetched_at": REVIEWED_AT,
            "fetched_by": REVIEWER,
            "note": note,
            "kind": "supporting evidence, not a licence receipt",
        }
        if quote is not None:
            payload["quote"] = quote
            payload["quote_found_in_stored_file"] = quote_found
        path.with_name(path.name + ".provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"provenance {rel}" + ("" if quote is None else f" quote_found={quote_found}"))
    if problems:
        print("PROBLEMS:", *problems, sep="\n  ")
        return 1
    print("all receipts written and their commercial-use grants verified against the stored files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
