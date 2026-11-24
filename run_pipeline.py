"""
Master Reproducible Pipeline Orchestrator — BFT65B Final Individual Report.
"A Big-Data Pipeline for Bank Failure Early Warning"

Author      : Pham Bach Tung
Student ID  : 11231023
Course      : BFT65B – Big Data Analytics in Finance (National Economics University)
Academic Integrity & Attribution:
  - Course Starter Script: `fdic_download_starter.py` (BFT65B Instructor, NEU, 2026)
  - Primary Data Source  : FDIC BankFind Suite REST API (https://api.fdic.gov/banks)
  - Open-Source Engines  : Polars (Vink et al., 2024), PyArrow, PySpark (Zaharia et al., 2016),
                           pandas (McKinney, 2010), scikit-learn (Pedregosa et al., 2011), LightGBM (Ke et al., 2017)
  - AI Tool Disclosure   : Google DeepMind Antigravity AI Assistant assisted in modular script structuring
                           and formatting; all empirical runs and verifications were executed on the 1.68M-row FDIC panel.

Runs all stages end-to-end with fixed random seeds (SEED = 42):
  Step 1: Data Acquisition & Completeness Verification (`src/part1_download.py`)
  Step 2: Storage Layout Design & Compression/Partitioning Benchmarks (`src/part1_storage_benchmark.py`)
  Step 3: Automated Data Quality Audit & Point-in-Time Panel Construction (`src/part2_quality_panel.py`)
  Step 4: Scalable Processing Benchmarks (Polars vs Pandas vs PySpark) (`src/part3_scalable_processing.py`)
  Step 5: Leak-Free CAMELS Feature Engineering & Training-Only Pipeline (`src/part4_feature_pipeline.py`)
  Step 6: Purged Temporal Validation, Data-Driven Cost Threshold, Single Test Evaluation & Spark MLlib (`src/part5_modelling.py`)

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-download   # if data_raw/fdic is already downloaded
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(title: str, cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f">>> {title}")
    print(f"    Command: {' '.join(cmd)}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Step '{title}' exited with code {res.returncode} after {elapsed:.1f}s")
        sys.exit(res.returncode)
    print(f"[OK] Completed '{title}' in {elapsed:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true", help="Skip API download check if already complete")
    args = ap.parse_args()

    py = sys.executable
    t_all = time.time()

    if not args.skip_download:
        run_step("Part 1A: FDIC Data Acquisition & Completeness Check", [py, "src/part1_download.py"])
    run_step("Part 1B: Storage Layout, Schema, Compression & Partitioning Benchmarks", [py, "src/part1_storage_benchmark.py"])
    run_step("Part 2: Automated Data Quality Audit & Clean Point-in-Time Panel", [py, "src/part2_quality_panel.py"])
    run_step("Part 3: Scalable Processing (Polars vs Pandas vs PySpark & Scaling Curve)", [py, "src/part3_scalable_processing.py"])
    run_step("Part 4: Point-in-Time CAMELS Feature Pipeline & Leakage Audit", [py, "src/part4_feature_pipeline.py"])
    run_step("Part 5: Purged Temporal CV, Cost Threshold, Single Test Eval & Spark MLlib", [py, "src/part5_modelling.py"])

    print("\n" + "=" * 80)
    print(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {time.time() - t_all:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
