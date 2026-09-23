"""Resumable acquisition of the v1.1 routing/sentiment held-out sources.

Downloads only public, ungated primary-project files. A downloaded source is not training-approved:
the generated inventory distinguishes an applicable primary license file from missing/unclear terms.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,urllib.request,tarfile,zipfile
from pathlib import Path

ROOT=Path("data/decision-v1-text");RAW=ROOT/"raw";LICENSES=ROOT/"licenses"
SOURCES={
 "typed-decisions":{"revision":"ea9306458d6e9563628369a3d1e72e362fb381d2","files":{
  "README.md":"https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/ea9306458d6e9563628369a3d1e72e362fb381d2/README.md",
  "train.parquet":"https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/ea9306458d6e9563628369a3d1e72e362fb381d2/all/train-00000-of-00001.parquet",
  "test.parquet":"https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/ea9306458d6e9563628369a3d1e72e362fb381d2/all/test-00000-of-00001.parquet"},"license":None},
 "banking77":{"revision":"57ec275d8078af65b7731c2a98be812d844a6d6b","base":"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/","files":{"train.csv":"banking_data/train.csv","test.csv":"banking_data/test.csv","categories.json":"banking_data/categories.json","LICENSE":"LICENSE"},"license":"LICENSE"},
 "clinc150":{"revision":"828f8093932c8fe6ca7936c3d2e52903b1c523de","base":"https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/","files":{"data_full.json":"data/data_full.json","LICENSE":"LICENSE"},"license":"LICENSE"},
 "massive":{"revision":"f966f21846043aabef9b0f974fa7970027f43738","base":"https://raw.githubusercontent.com/alexa/massive/f966f21846043aabef9b0f974fa7970027f43738/","files":{"amazon-massive-dataset-1.1.tar.gz":"https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz","LICENSE.txt":"LICENSE.txt","NOTICE.md":"NOTICE.md","THIRD-PARTY.md":"THIRD-PARTY.md"},"license":"LICENSE.txt"},
 "snips":{"revision":"b86ac7f1577868c42158d0dec77db50956046696","base":"https://raw.githubusercontent.com/sonos/nlu-benchmark/b86ac7f1577868c42158d0dec77db50956046696/","files":dict({f"{intent}/{split}_{intent}{'_full' if split=='train' else ''}.json":f"2017-06-custom-intent-engines/{intent}/{split}_{intent}{'_full' if split=='train' else ''}.json" for intent in ("AddToPlaylist","BookRestaurant","GetWeather","PlayMusic","RateBook","SearchCreativeWork","SearchScreeningEvent") for split in ("train","validate")},**{"README.md":"2017-06-custom-intent-engines/README.md","LICENSE":"LICENSE"}),"license":"LICENSE"},
 "goemotions":{"revision":"758b894eb02dc2a7097068031089a2803be147c6","base":"https://raw.githubusercontent.com/google-research/google-research/758b894eb02dc2a7097068031089a2803be147c6/","files":{"LICENSE":"LICENSE","emotions.txt":"goemotions/data/emotions.txt","train.tsv":"goemotions/data/train.tsv","dev.tsv":"goemotions/data/dev.tsv","test.tsv":"goemotions/data/test.tsv"},"license":"LICENSE"},
 "sst5":{"revision":"original-site","files":{"stanfordSentimentTreebank.zip":"https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"},"license":None},
 "mmlu":{"revision":"4450500f923c49f1fb1dd3d99108a0bd9717b660","base":"https://raw.githubusercontent.com/hendrycks/test/4450500f923c49f1fb1dd3d99108a0bd9717b660/","files":{"data.tar":"https://people.eecs.berkeley.edu/~hendrycks/data.tar","README.md":"README.md","LICENSE":"LICENSE"},"license":"LICENSE"},
}
LICENSE_REVIEW={
 "banking77":("CC-BY-4.0",True,"Repository LICENSE is the CC Attribution 4.0 legal code."),
 "clinc150":("CC-BY-3.0",True,"Repository LICENSE is the CC Attribution 3.0 Unported legal code."),
 "massive":("CC-BY-4.0",True,"NOTICE explicitly licenses the MASSIVE dataset under CC BY 4.0; LICENSE.txt points dataset terms to NOTICE."),
 "snips":("CC0-1.0",True,"Repository LICENSE is the CC0 1.0 Universal legal code."),
 "goemotions":("Apache-2.0",True,"The GoEmotions files are in google-research under its repository Apache-2.0 LICENSE."),
 "mmlu":("MIT",True,"The MMLU repository LICENSE is MIT and accompanies the repository's linked data archive."),
 "typed-decisions":(None,False,"Primary dataset repository contains no license file; its metadata tag is insufficient."),
 "sst5":(None,False,"Original archive contains a README but no license grant."),
}
EVALUATION_ONLY={"sst5","mmlu"}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def download(url,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.stat().st_size:return "cached"
 tmp=path.with_suffix(path.suffix+".part")
 req=urllib.request.Request(url,headers={"User-Agent":"imajev-dataset-audit/1.0"})
 with urllib.request.urlopen(req,timeout=120) as r,tmp.open("wb") as f:
  while block:=r.read(1<<20):f.write(block)
 tmp.replace(path);return "downloaded"
def rows(path):
 try:
  if path.suffix==".csv":
   with path.open(errors="replace",newline="") as f:n=sum(1 for _ in csv.reader(f))
   return n if "mmlu/extracted" in path.as_posix() else max(0,n-1)
  if path.suffix==".tsv":return sum(1 for x in path.open(errors="replace") if x.strip())
  if path.suffix==".txt":return sum(1 for x in path.open(errors="replace") if x.strip())-(1 if path.name in {"datasetSentences.txt","datasetSplit.txt","sentiment_labels.txt"} else 0)
  if path.suffix==".jsonl":return sum(1 for x in path.open(errors="replace") if x.strip())
  if path.suffix==".json":
   x=json.loads(path.read_text());return len(x) if isinstance(x,list) else sum(len(v) for v in x.values() if isinstance(v,list))
  if path.suffix==".parquet":
   import pyarrow.parquet as pq;return pq.ParquetFile(path).metadata.num_rows
 except Exception:return None
 return None
def extracted(source):
 out=[];root=RAW/source
 if source=="massive":
  archive=root/"amazon-massive-dataset-1.1.tar.gz";wanted={f"1.1/data/{x}.jsonl" for x in ("en-US","de-DE","es-ES","fr-FR","hi-IN","ja-JP")}
  with tarfile.open(archive) as tf:
   for member in tf.getmembers():
    if member.name in wanted:
     dest=root/"selected"/Path(member.name).name;dest.parent.mkdir(parents=True,exist_ok=True)
     if not dest.exists():dest.write_bytes(tf.extractfile(member).read())
     out.append(dest)
 elif source=="mmlu":
  dest=root/"extracted"
  if not dest.exists():
   dest.mkdir(parents=True);tarfile.open(root/"data.tar").extractall(dest,filter="data")
  out += sorted(dest.rglob("*.csv"))
 elif source=="sst5":
  dest=root/"extracted"
  if not dest.exists():dest.mkdir(parents=True);zipfile.ZipFile(root/"stanfordSentimentTreebank.zip").extractall(dest)
  out += sorted((dest/"stanfordSentimentTreebank").glob("*.txt"))
 return [{"path":str(p),"url":"extracted from acquired archive","bytes":p.stat().st_size,"sha256":digest(p),"rows":rows(p),"status":"extracted"} for p in out]
def acquire(names=None):
 inventory=[]
 for name,spec in SOURCES.items():
  if names and name not in names:continue
  record={"source":name,"revision":spec["revision"],"files":[],"errors":[]}
  for local,remote in spec["files"].items():
   url=remote if remote.startswith("http") else spec["base"]+remote;path=RAW/name/local
   try:
    status=download(url,path);entry={"path":str(path),"url":url,"bytes":path.stat().st_size,"sha256":digest(path),"rows":rows(path),"status":status};record["files"].append(entry)
    if local==spec.get("license"):
     dest=LICENSES/name/Path(local).name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());record["license"]={"path":str(dest),"source_url":url,"sha256":digest(dest),"status":"primary_file_downloaded_review_required"}
    elif Path(local).name.lower() in {"notice.md","third-party.md"}:
     dest=LICENSES/name/Path(local).name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
   except Exception as exc:record["errors"].append({"file":local,"url":url,"error":str(exc)})
  if "license" not in record:record["license"]={"status":"quarantined_license_file_download_failed"}
  if not spec.get("license"):record["license"]={"status":"quarantined_no_primary_license_file_identified"}
  try:record["files"]+=extracted(name)
  except Exception as exc:record["errors"].append({"file":"archive extraction","error":str(exc)})
  spdx,approved,note=LICENSE_REVIEW[name];record["license"].update({"spdx":spdx,"review_note":note,"status":"verified_primary_license" if approved else record["license"]["status"]})
  if name=="massive" and approved:
   evidence=LICENSES/name/"NOTICE.md";record["license"].update({"path":str(evidence),"sha256":digest(evidence),"source_url":spec["base"]+"NOTICE.md"})
  if approved:
   evidence=Path(record["license"]["path"]);receipt=evidence.with_name(evidence.name+".receipt.json")
   receipt.write_text(json.dumps({"evidence_sha256":digest(evidence),"spdx":spdx,"commercial_use_reviewed":True,
     "source":name,"source_url":record["license"]["source_url"],"review_note":note},indent=2)+"\n")
   record["license"]["receipt"]=str(receipt)
  record["acquisition_status"]="complete" if not record["errors"] else "partial"
  record["training_approved"]=approved and name not in EVALUATION_ONLY
  if name in EVALUATION_ONLY:record["training_blocker"]="held-out evaluation family; never admit to training"
  inventory.append(record)
 return inventory
def main():
 p=argparse.ArgumentParser();p.add_argument("sources",nargs="*",choices=sorted(SOURCES));p.add_argument("--report",type=Path,default=Path("reports/v1.1-datasets/routing.json"));a=p.parse_args()
 result=acquire(set(a.sources) if a.sources else None);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps({"schema_version":1,"sources":result},indent=2)+"\n")
 print(json.dumps({x["source"]:{"files":len(x["files"]),"errors":len(x["errors"]),"license":x["license"]["status"]} for x in result},indent=2))
if __name__=="__main__":main()
