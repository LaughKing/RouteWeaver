#!/usr/bin/env python
"""Fetch the prepared datasets from the Hub into $ROUTEWEAVER_DATASETS.

    python data/hf_download.py

    datasets/
      parquet/    routeweaver_train.parquet      the 3,200-question training set
                  routeweaver_cold_pool.parquet  the cold start's own pool
                  val_stub.parquet               the 32-row stub verl requires
      eval/       eval_600_free, eval_700_free, eval_triviaqa120_free
                  ood_{aime149,gpqa_diamond198,bbh1080}_free
      cost_calib/ cost_calib_128.parquet         the C_ref calibration set

Set ROUTEWEAVER_DATASET_REPO to pull from a different dataset repository.
"""
import os
import sys
from pathlib import Path

REPO_ID = os.environ.get("ROUTEWEAVER_DATASET_REPO", "e2rea1/RouteWeaver-data")
HERE = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("ROUTEWEAVER_DATASETS", HERE.parent / "datasets"))


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                             local_dir=str(DATASET_ROOT))
    got = sorted(p.relative_to(path) for p in Path(path).rglob("*.parquet"))
    print(f"{len(got)} parquets in {path}")
    for p in got:
        print(f"  {p}")
    if not got:
        sys.exit(f"nothing downloaded from {REPO_ID}")


if __name__ == "__main__":
    main()
