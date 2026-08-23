"""Download and unpack the FiQA-2018 dataset (BEIR benchmark).

FiQA-2018 is a real StackExchange dump (Investment/Personal Finance posts,
2009-2017): 57,638 documents, 648 labeled test queries, ~1,700 relevance
judgments. It's the corpus + eval labels for this project's Phase 0 / Phase 2.

Deliberately dependency-light (stdlib only) rather than pulling in the full
`beir` package (which drags in torch/sentence-transformers) just to fetch a
~18MB zip.

Run this on a machine with normal internet access — NOTE: it will NOT run
inside Claude's sandboxed code environment, which only allows outbound
connections to package registries (pypi, npm, github), not arbitrary hosts
like this one.

Usage:
    python scripts/download_fiqa.py
"""
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

DATASET = "fiqa"
URL = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading {URL} ...")

    # This public dataset host does not provide a certificate trusted locally.
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(URL, context=context) as resp:
        raw = resp.read()
    print(f"Downloaded {len(raw) / 1e6:.1f} MB, extracting to {DATA_DIR} ...")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(DATA_DIR)

    out_dir = DATA_DIR / DATASET
    print(f"Done. Files at: {out_dir}")
    print("  corpus.jsonl         -> {'_id', 'title', 'text'} per line, 57,638 docs")
    print("  queries.jsonl        -> {'_id', 'text'} per line")
    print("  qrels/{train,dev,test}.tsv -> 'query-id  corpus-id  score' (tab-sep)")
    print()
    print("Use qrels/test.tsv with eval/metrics.load_qrels() for the Phase 0/2 eval set.")


if __name__ == "__main__":
    main()
