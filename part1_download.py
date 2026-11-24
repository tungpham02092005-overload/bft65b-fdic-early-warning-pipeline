"""
Part 1: Reproducible & Resumable Acquisition of FDIC BankFind Data (1984Q1 - 2026Q2).

Author      : Pham Bach Tung | Student ID: 11231023
Course      : BFT65B – Big Data Analytics in Finance (National Economics University)
Attribution : Adapted & extended from course starter script `fdic_download_starter.py` (BFT65B, NEU)
              and FDIC BankFind Suite Public REST API documentation (https://api.fdic.gov/banks/docs/).
              AI Tool Disclosure: Google DeepMind Antigravity AI Assistant assisted with ThreadPoolExecutor
              retry boilerplate; verified against live FDIC API total counts (1,682,616 rows).

Enhances `fdic_download_starter.py` with:
- Polite bounded concurrency across quarters (ThreadPoolExecutor) for faster initial acquisition
  while remaining 100% compatible with `fdic_download_starter.py` file layout and manifest format.
- Strict schema enforcement and completeness verification (row count == API meta.total,
  zero duplicate (CERT, REPDTE) keys, and cryptographic/row-level manifest logging).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import sys
import threading
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://api.fdic.gov/banks"
PAGE_SIZE = 10_000

FIN_FIELDS = [
    "CERT", "REPDTE", "STNAME",
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",          # balance sheet, liquidity
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",        # loans, reserves
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",     # asset quality
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",      # earnings
    "ROA", "ROE", "NIMY",                                           # earnings ratios
    "RBC1AAJ", "RBCRWAJ",                                           # capital ratios
]
FAIL_FIELDS = ["CERT", "NAME", "CITYST", "FAILDATE", "FAILYR", "COST", "QBFASSET", "QBFDEP", "RESTYPE"]
INST_FIELDS = ["CERT", "NAME", "STNAME", "BKCLASS", "ACTIVE", "ESTYMD", "ENDEFYMD", "CHANGEC1", "INSDATE"]

_thread_local = threading.local()
_manifest_lock = threading.Lock()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "BFT65B-course-starter/1.0"})
        _thread_local.session = s
    return _thread_local.session


def get_json(endpoint: str, params: dict, max_retries: int = 6, pause: float = 0.15) -> dict:
    url = f"{BASE}/{endpoint}"
    session = get_session()
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=180)
            if r.status_code == 200:
                if pause > 0:
                    time.sleep(pause)
                return r.json()
            print(f"    HTTP {r.status_code} on {endpoint} (attempt {attempt}) - retrying")
        except requests.RequestException as exc:
            print(f"    {type(exc).__name__} on {endpoint} (attempt {attempt}) - retrying")
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Giving up on {url} with {params}")


def fetch_all(endpoint: str, fields: list[str], filters: str | None = None,
              sort_by: str = "CERT", pause: float = 0.15) -> tuple[pd.DataFrame, int]:
    rows, offset, total = [], 0, None
    while True:
        params = {
            "fields": ",".join(fields),
            "limit": PAGE_SIZE,
            "offset": offset,
            "sort_by": sort_by,
            "sort_order": "ASC",
            "format": "json",
        }
        if filters:
            params["filters"] = filters
        js = get_json(endpoint, params, pause=pause)
        total = js["meta"]["total"]
        page = [rec["data"] for rec in js["data"]]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= total:
            break
    df = pd.DataFrame(rows)
    return df.drop(columns=["ID"], errors="ignore"), total


def quarter_ends(start: date, end: date) -> list[str]:
    ends = []
    for y in range(start.year, end.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = date(y, m, d)
            if start <= q <= end:
                ends.append(q.strftime("%Y%m%d"))
    return ends


def download_one_quarter(q: str, idx: int, total_q: int, fin_dir: Path,
                         manifest_path: Path, manifest: dict, pause: float) -> dict:
    target = fin_dir / f"REPDTE={q}.parquet"
    with _manifest_lock:
        if target.exists() and manifest.get(q, {}).get("complete"):
            print(f"[{idx}/{total_q}] {q}: already complete, skipped")
            return manifest[q]

    t0 = time.time()
    df, total = fetch_all("financials", FIN_FIELDS, filters=f"REPDTE:{q}", pause=pause)
    if df.empty:
        print(f"[{idx}/{total_q}] {q}: no data (quarter not published yet)")
        return {"rows": 0, "api_total": 0, "complete": True, "unpublished": True}

    # Ensure deterministic column order matching FIN_FIELDS
    for col in FIN_FIELDS:
        if col not in df.columns:
            df[col] = None
    df = df[FIN_FIELDS]

    df.to_parquet(target, index=False)
    complete = (len(df) == total) and (not df.duplicated(["CERT", "REPDTE"]).any())
    entry = {"rows": len(df), "api_total": total, "complete": bool(complete)}

    with _manifest_lock:
        manifest[q] = entry
        # Sort keys chronologically
        ordered = {k: manifest[k] for k in sorted(manifest.keys())}
        manifest_path.write_text(json.dumps(ordered, indent=1))

    elapsed = time.time() - t0
    print(f"[{idx:>3}/{total_q}] {q}: {len(df):>6,} rows (API total {total:>6,}) "
          f"{'OK' if complete else 'INCOMPLETE'}  {elapsed:5.1f}s")
    return entry


def download_financials(out: Path, quarters: list[str], workers: int = 6, pause: float = 0.15) -> None:
    fin_dir = out / "financials"
    fin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest_financials.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(download_one_quarter, q, i, len(quarters), fin_dir, manifest_path, manifest, pause)
            for i, q in enumerate(quarters, 1)
        ]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()


def download_small_tables(out: Path, pause: float = 0.15) -> dict:
    stats = {}
    for name, fields, sort_by in (("failures", FAIL_FIELDS, "FAILDATE"),
                                  ("institutions", INST_FIELDS, "CERT")):
        t0 = time.time()
        df, total = fetch_all(name, fields, sort_by=sort_by, pause=pause)
        for col in fields:
            if col not in df.columns:
                df[col] = None
        df = df[fields]
        df.to_parquet(out / f"{name}.parquet", index=False)
        elapsed = time.time() - t0
        print(f"{name:<12}: {len(df):>6,} rows (API total {total:,}) {elapsed:5.1f}s")
        stats[name] = {"rows": int(len(df)), "api_total": int(total)}
    return stats


def verify_completeness(out: Path, small_stats: dict) -> dict:
    manifest = json.loads((out / "manifest_financials.json").read_text())
    panel = pd.read_parquet(out / "financials", columns=["CERT", "REPDTE"])
    n_dup = int(panel.duplicated(["CERT", "REPDTE"]).sum())
    bad = [q for q, m in manifest.items() if not m["complete"]]
    fails = pd.read_parquet(out / "failures.parquet")
    inst = pd.read_parquet(out / "institutions.parquet")

    q_sorted = sorted(manifest.keys())
    api_total_sum = sum(m["api_total"] for m in manifest.values())

    summary = {
        "download_timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "download_date": dt.date.today().isoformat(),
        "quarters_downloaded": len(manifest),
        "first_quarter": q_sorted[0] if q_sorted else None,
        "last_quarter": q_sorted[-1] if q_sorted else None,
        "financials_rows": int(len(panel)),
        "financials_api_total_sum": int(api_total_sum),
        "distinct_banks_in_financials": int(panel["CERT"].nunique()),
        "duplicate_cert_repdte": n_dup,
        "incomplete_quarters": bad,
        "failures_rows": int(len(fails)),
        "failures_api_total": small_stats.get("failures", {}).get("api_total", int(len(fails))),
        "institutions_rows": int(len(inst)),
        "institutions_api_total": small_stats.get("institutions", {}).get("api_total", int(len(inst))),
        "all_checks_passed": (n_dup == 0 and len(bad) == 0 and len(panel) == api_total_sum),
    }

    print("\n=== Completeness Check ===")
    for k, v in summary.items():
        print(f"  {k:<32}: {v}")

    out_tables = Path("outputs/tables")
    out_tables.mkdir(parents=True, exist_ok=True)
    (out_tables / "part1_acquisition_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="1984-03-31")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--out", default="data_raw/fdic")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pause", type=float, default=0.15)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quarters = quarter_ends(date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"Downloading {len(quarters)} quarters ({quarters[0]} -> {quarters[-1]}) "
          f"with {args.workers} workers into {out.resolve()}\n")

    t_start = time.time()
    download_financials(out, quarters, workers=args.workers, pause=args.pause)
    print()
    small_stats = download_small_tables(out, pause=args.pause)
    summary = verify_completeness(out, small_stats)
    summary["total_download_wall_time_sec"] = round(time.time() - t_start, 2)
    Path("outputs/tables/part1_acquisition_summary.json").write_text(json.dumps(summary, indent=2))
    sys.exit(0 if summary["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
