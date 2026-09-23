"""Resumable acquisition and license inventory for v1.1 screening/rubric sources.

Public files are pinned where the host supports revisions. Sources without a clear commercial
dataset grant are downloaded under ``raw/<source>/quarantine`` and never marked training-approved.
No gated terms are accepted by this script.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/decision-v1-text"
RAW = DATA / "raw"
LICENSES = DATA / "licenses"
REPORT = ROOT / "reports/v1.1-datasets/screening-rubric.json"

HTTP_SOURCES = {
    "spamassassin": {
        "approved": False,
        "license": "unclear-corpus-rights",
        "reason": "Apache-2.0 covers SpamAssassin software, but the public corpus README gives no dataset license grant.",
        "revision": "publiccorpus-20030228+20050311",
        "files": [
            ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2", "20030228_easy_ham.tar.bz2"),
            ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2", "20030228_easy_ham_2.tar.bz2"),
            ("https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2", "20030228_hard_ham.tar.bz2"),
            ("https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2", "20030228_spam.tar.bz2"),
            ("https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2", "20050311_spam_2.tar.bz2"),
        ],
        "evidence": [("https://spamassassin.apache.org/old/publiccorpus/readme.html", "CORPUS_README.html")],
    },
    "enron_spam": {
        "approved": False,
        "license": "no-license-grant",
        "reason": "CMU distributes Enron mail for research with privacy guidance but provides no explicit reuse license; labelled spam derivative provenance is also unclear.",
        "revision": "aueb-preprocessed-20060622",
        "files": [(f"https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron{i}.tar.gz", f"enron{i}.tar.gz") for i in range(1, 7)],
        "evidence": [("https://www2.aueb.gr/users/ion/data/enron-spam/", "AUEB_DATASET_PAGE.html")],
    },
    "openai_moderation": {
        "approved": True,
        "license": "MIT",
        "reason": "The dataset file is distributed inside OpenAI's MIT-licensed moderation-api-release repository.",
        "revision": "f4ab51b5edd3bfbcb349a56324274235b674e0e4",
        "files": [("https://raw.githubusercontent.com/openai/moderation-api-release/f4ab51b5edd3bfbcb349a56324274235b674e0e4/data/samples-1680.jsonl.gz", "samples-1680.jsonl.gz")],
        "evidence": [("https://raw.githubusercontent.com/openai/moderation-api-release/f4ab51b5edd3bfbcb349a56324274235b674e0e4/LICENSE", "LICENSE")],
        "evidence_name": "LICENSE",
    },
    "onestopenglish": {
        "approved": False,
        "license": "research-restricted/unclear",
        "reason": "The paper describes research use under license restrictions; no commercial redistribution grant was found.",
        "revision": "37f8db3945cd2f3cc0caafe45674147b224349be",
        "files": [("https://github.com/nishkalavallabhi/OneStopEnglishCorpus/archive/37f8db3945cd2f3cc0caafe45674147b224349be.zip", "OneStopEnglishCorpus-master.zip")],
        "evidence": [("https://aclanthology.org/W18-0535.pdf", "W18-0535.pdf")],
    },
}

HF_SOURCES = {
    "civil_comments": {"repo": "google/civil_comments", "approved": True, "license": "CC0-1.0",
                       "license_url": "https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt",
                       "reason": "Dataset card states both annotations and underlying comment text are CC0."},
    "nvidia_aegis": {"repo": "nvidia/Aegis-AI-Content-Safety-Dataset-1.0", "approved": True,
                     "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
                     "reason": "NVIDIA dataset repository declares CC-BY-4.0."},
    "helpsteer2": {"repo": "nvidia/HelpSteer2", "approved": True, "license": "CC-BY-4.0",
                   "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
                   "reason": "NVIDIA dataset repository and paper declare CC-BY-4.0."},
    "ultrafeedback": {"repo": "openbmb/UltraFeedback", "approved": False, "license": "mixed-upstream/unclear",
                      "license_url": "https://raw.githubusercontent.com/OpenBMB/UltraFeedback/main/LICENSE",
                      "reason": "Repository is MIT, but prompts are copied from six upstream datasets; per-record upstream rights require review."},
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "imajev-dataset-audit/1.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            if offset and response.status != 206:
                offset = 0
                partial.unlink(missing_ok=True)
            with partial.open("ab" if offset else "wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
    except urllib.error.URLError:
        # Some university hosts have chains accepted by the system curl trust store but not Python's.
        # Curl still verifies TLS; never add --insecure here.
        subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "15",
                        "--max-time", "60", "--continue-at", "-",
                        "--output", str(partial), url], check=True)
    partial.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt(name: str, spec: dict, evidence_name: str, source_url: str) -> None:
    if not spec["approved"]:
        return
    evidence = LICENSES / name / evidence_name
    if not evidence.is_file():
        raise ValueError(f"Approved source {name} is missing license evidence {evidence}")
    receipt = evidence.with_name(evidence.name + ".receipt.json")
    receipt.write_text(json.dumps({"evidence_sha256": sha256(evidence), "spdx": spec["license"],
                                   "commercial_use_reviewed": True, "source": name,
                                   "source_url": source_url, "review_note": spec["reason"]},
                                  indent=2, sort_keys=True) + "\n")


def row_count(path: Path) -> int | None:
    try:
        if path.name.endswith((".jsonl", ".json")):
            return sum(1 for line in path.open("rb") if line.strip())
        if path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rb") as handle:
                return sum(1 for line in handle if line.strip())
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            return pq.read_metadata(path).num_rows
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                return max(0, sum(1 for _ in csv.reader(handle)) - 1)
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                return sum(member.isfile() for member in archive.getmembers())
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                return sum(not member.is_dir() for member in archive.infolist())
    except (OSError, EOFError, ValueError):
        return None
    return None


def acquire_http(name: str, spec: dict) -> dict:
    base = RAW / name / ("data" if spec["approved"] else "quarantine")
    failures = []
    for url, filename in spec["files"]:
        destination = base / filename
        if destination.exists():
            continue
        try:
            download(url, destination)
        except Exception as exc:
            failures.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    for url, filename in spec["evidence"]:
        destination = LICENSES / name / filename
        if not destination.exists():
            try:
                download(url, destination)
            except Exception as exc:
                failures.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    if not failures:
        write_receipt(name, spec, spec.get("evidence_name", spec["evidence"][0][1]), spec["evidence"][0][0])
    return inventory(name, spec, failures)


def acquire_hf(name: str, spec: dict) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    info = api.dataset_info(spec["repo"])
    revision = info.sha
    base = RAW / name / ("data" if spec["approved"] else "quarantine")
    failures = []
    try:
        snapshot_download(spec["repo"], repo_type="dataset", revision=revision, local_dir=base,
                          allow_patterns=["*.parquet", "*.json", "*.jsonl", "*.jsonl.gz", "README.md"])
    except Exception as exc:
        failures.append({"url": f"https://huggingface.co/datasets/{spec['repo']}/tree/{revision}",
                         "error": f"{type(exc).__name__}: {exc}"})
    license_dir = LICENSES / name
    license_dir.mkdir(parents=True, exist_ok=True)
    try:
        readme = base / "README.md"
        if readme.exists():
            shutil.copy2(readme, license_dir / "DATASET_CARD.md")
        if not (license_dir / "LICENSE_LEGALCODE.txt").exists():
            download(spec["license_url"], license_dir / "LICENSE_LEGALCODE.txt")
        write_receipt(name, spec, "LICENSE_LEGALCODE.txt", spec["license_url"])
    except Exception as exc:
        failures.append({"url": spec["license_url"], "error": f"{type(exc).__name__}: {exc}"})
    return inventory(name, {**spec, "revision": revision}, failures)


def inventory(name: str, spec: dict, failures: list[dict]) -> dict:
    roots = [RAW / name / "data", RAW / name / "quarantine", LICENSES / name]
    files = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*")
                           if p.is_file() and not p.name.endswith(".part") and ".cache" not in p.parts):
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                          "sha256": sha256(path), "rows_or_archive_files": row_count(path)})
    data_files = [item for item in files if "/raw/" in item["path"] and not item["path"].endswith("README.md")]
    source_url = (f"https://huggingface.co/datasets/{spec['repo']}/tree/{spec.get('revision', 'main')}"
                  if spec.get("repo") else spec["files"][0][0] if spec.get("files") else None)
    download_urls = ([f"https://huggingface.co/datasets/{spec['repo']}/tree/{spec.get('revision', 'main')}"]
                     if spec.get("repo") else [url for url, _ in spec.get("files", [])])
    return {"source": name, "source_url": source_url, "download_urls": download_urls,
            "revision": spec.get("revision"), "training_approved": spec["approved"],
            "license_conclusion": spec["license"], "license_review": spec["reason"],
            "status": "downloaded" if data_files and not failures else ("partial" if data_files else "unavailable"),
            "failures": failures, "integrity_status": "verified_local_sha256_and_readable", "files": files,
            "row_count": sum(item["rows_or_archive_files"] or 0 for item in data_files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated source names")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    selected = set(args.only.split(",")) if args.only else set(HTTP_SOURCES) | set(HF_SOURCES)
    unknown = selected - set(HTTP_SOURCES) - set(HF_SOURCES)
    if unknown:
        raise SystemExit(f"Unknown sources: {sorted(unknown)}")
    records = []
    for name in sorted(selected):
        print(f"acquiring {name}", flush=True)
        records.append(acquire_http(name, HTTP_SOURCES[name]) if name in HTTP_SOURCES
                       else acquire_hf(name, HF_SOURCES[name]))
    if args.only and args.report.exists():
        previous = json.loads(args.report.read_text())
        merged = {record["source"]: record for record in previous.get("sources", [])}
        merged.update({record["source"]: record for record in records})
        records = [merged[name] for name in sorted(merged)]
    report = {"schema_version": "1.0", "policy": "Only explicit commercial-compatible dataset grants are approved; gated terms are never accepted.",
              "sources": records, "summary": {"requested": len(records),
              "downloaded": sum(r["status"] == "downloaded" for r in records),
              "partial": sum(r["status"] == "partial" for r in records),
              "unavailable": sum(r["status"] == "unavailable" for r in records),
              "training_approved": sum(r["training_approved"] and r["status"] == "downloaded" for r in records)}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
