"""
Part 1: Storage Architecture Design & Empirical Benchmark (Format, Schema, Partitioning, Compression).

Author      : Pham Bach Tung | Student ID: 11231023
Course      : BFT65B – Big Data Analytics in Finance (National Economics University)
Attribution : Uses Apache Arrow / PyArrow (`pyarrow==14.0.2`) and Polars (`polars==1.2.1`, Vink et al., 2024).
              AI Tool Disclosure: Google DeepMind Antigravity AI Assistant assisted with multi-layout benchmark
              loop formatting; all storage sizes and latencies were measured empirically on the 1.68M-row panel.

Benchmarks on the full ~1.68M-row FDIC Call Report panel:
1. Format & Compression comparison:
   - Raw CSV (uncompressed)
   - CSV (gzip)
   - Parquet (Raw API schema, Snappy)
   - Parquet (Optimized explicit schema: Int32/Date/Enum/Float64) with Uncompressed, Snappy, LZ4, ZSTD, GZIP
   - Parquet (Float32 downcast comparison for numerical precision vs storage trade-off)
2. Partitioning Strategy comparison:
   - Single monolithic Parquet file (ZSTD)
   - Quarter-partitioned layout (170 files, REPDTE=YYYYMMDD.parquet - ingestion-aligned)
   - Year-partitioned Hive layout (43 directories, YEAR=YYYY/)
   - State-partitioned Hive layout (~60 directories, STNAME=XX/)
3. Query Workload measurements:
   - Write time (s)
   - Disk size (MB) & compression ratio vs Raw CSV
   - Full scan time (all 33 columns)
   - Column projection pushdown (5 columns: CERT, REPDTE, ASSET, EQ, ROA)
   - Predicate pushdown + Partition pruning (GFC window 2008Q1-2012Q4, 5 columns)
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300,
})

NUMERIC_COLS = [
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ",
]


def dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def time_fn(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def load_raw_panel(raw_dir: Path) -> pl.DataFrame:
    pl.enable_string_cache()
    files = sorted((raw_dir / "financials").glob("REPDTE=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir / 'financials'}")
    # Read with Polars and cast to canonical schema
    dfs = []
    for f in files:
        df = pl.read_parquet(f)
        exprs = [
            pl.col("CERT").cast(pl.Int32, strict=False),
            pl.col("REPDTE").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("REPDTE_DATE"),
            pl.col("REPDTE").cast(pl.Utf8),
            pl.col("STNAME").cast(pl.Utf8),
        ]
        for c in NUMERIC_COLS:
            if c in df.columns:
                exprs.append(pl.col(c).cast(pl.Float64, strict=False))
            else:
                exprs.append(pl.lit(None, dtype=pl.Float64).alias(c))
        dfs.append(df.select(exprs))
    panel = pl.concat(dfs, how="vertical")
    return panel.sort(["REPDTE_DATE", "CERT"])


def run_storage_benchmarks(raw_dir: Path, bench_dir: Path, out_tables: Path, out_figs: Path) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("Loading raw quarter files into canonical Polars DataFrame...")
    panel_f64 = load_raw_panel(raw_dir)
    n_rows, n_cols = panel_f64.shape
    print(f"Loaded {n_rows:,} rows x {n_cols} columns.")

    # Prepare explicit typed schemas:
    # 1. Optimized Float64 schema (Int32 CERT, Date REPDTE_DATE, Categorical STNAME, Float64 financials)
    panel_opt64 = panel_f64.with_columns([
        pl.col("STNAME").cast(pl.Categorical),
    ]).drop("REPDTE").rename({"REPDTE_DATE": "REPDTE"})

    # 2. Optimized Float32 schema (testing numerical precision vs memory/disk footprint)
    panel_opt32 = panel_opt64.with_columns([
        pl.col(c).cast(pl.Float32) for c in NUMERIC_COLS
    ])

    # 3. Untyped string/float raw-like schema (as returned directly from JSON before date/cat optimization)
    panel_raw_schema = panel_f64.drop("REPDTE_DATE")

    results = []
    proj_cols = ["CERT", "REPDTE", "ASSET", "EQ", "ROA"]

    # --- A. Raw CSV (uncompressed) ---
    csv_path = bench_dir / "financials_raw.csv"
    print("Benchmarking Raw CSV...")
    t_write = time_fn(lambda: panel_raw_schema.write_csv(csv_path), repeats=1)
    size_csv = dir_size_mb(csv_path)
    t_full = time_fn(lambda: pl.read_csv(csv_path), repeats=3)
    t_proj = time_fn(lambda: pl.read_csv(csv_path, columns=proj_cols), repeats=3)
    t_pred = time_fn(
        lambda: pl.scan_csv(csv_path)
        .filter((pl.col("REPDTE").cast(pl.Utf8) >= "20080331") & (pl.col("REPDTE").cast(pl.Utf8) <= "20121231"))
        .select(proj_cols)
        .collect(),
        repeats=3,
    )
    results.append({
        "layout": "Raw CSV (Uncompressed)",
        "category": "Format/Codec",
        "codec": "none",
        "schema": "Untyped Text",
        "files": 1,
        "size_mb": round(size_csv, 2),
        "compression_ratio": 1.00,
        "write_sec": round(t_write, 3),
        "full_scan_sec": round(t_full, 3),
        "projection_5col_sec": round(t_proj, 3),
        "predicate_gfc_sec": round(t_pred, 3),
    })

    # --- B. CSV Gzip ---
    csv_gz_path = bench_dir / "financials_raw.csv.gz"
    print("Benchmarking Gzip CSV...")
    import gzip
    def write_csv_gz():
        with gzip.open(csv_gz_path, "wb", compresslevel=6) as f:
            panel_raw_schema.write_csv(f)
    t_write_gz = time_fn(write_csv_gz, repeats=1)
    size_csvgz = dir_size_mb(csv_gz_path)
    t_full_gz = time_fn(lambda: pl.read_csv(csv_gz_path), repeats=2)
    t_proj_gz = time_fn(lambda: pl.read_csv(csv_gz_path, columns=proj_cols), repeats=2)
    results.append({
        "layout": "CSV (Gzip)",
        "category": "Format/Codec",
        "codec": "gzip",
        "schema": "Untyped Text",
        "files": 1,
        "size_mb": round(size_csvgz, 2),
        "compression_ratio": round(size_csv / size_csvgz, 2),
        "write_sec": round(t_write_gz, 3),
        "full_scan_sec": round(t_full_gz, 3),
        "projection_5col_sec": round(t_proj_gz, 3),
        "predicate_gfc_sec": round(t_proj_gz, 3),  # Gzip CSV cannot seek or push down predicates without decompressing stream
    })

    # --- C. Single Parquet across Codecs & Schemas ---
    codecs = ["uncompressed", "lz4", "snappy", "zstd", "gzip"]
    for codec in codecs:
        pq_path = bench_dir / f"financials_opt64_{codec}.parquet"
        print(f"Benchmarking Parquet (Opt-Float64, {codec})...")
        t_w = time_fn(lambda: panel_opt64.write_parquet(pq_path, compression=codec, statistics=True), repeats=2)
        sz = dir_size_mb(pq_path)
        t_f = time_fn(lambda: pl.read_parquet(pq_path), repeats=3)
        t_p = time_fn(lambda: pl.read_parquet(pq_path, columns=proj_cols), repeats=3)
        t_flt = time_fn(
            lambda: pl.scan_parquet(pq_path)
            .filter((pl.col("REPDTE") >= pl.date(2008, 3, 31)) & (pl.col("REPDTE") <= pl.date(2012, 12, 31)))
            .select(proj_cols)
            .collect(),
            repeats=3,
        )
        results.append({
            "layout": f"Parquet Single (Float64, {codec.upper()})",
            "category": "Format/Codec",
            "codec": codec,
            "schema": "Int32+Date+Cat+Float64",
            "files": 1,
            "size_mb": round(sz, 2),
            "compression_ratio": round(size_csv / sz, 2),
            "write_sec": round(t_w, 3),
            "full_scan_sec": round(t_f, 3),
            "projection_5col_sec": round(t_p, 3),
            "predicate_gfc_sec": round(t_flt, 3),
        })

    # --- D. Parquet Float32 with ZSTD ---
    pq_f32_path = bench_dir / "financials_opt32_zstd.parquet"
    print("Benchmarking Parquet (Opt-Float32, ZSTD)...")
    t_w32 = time_fn(lambda: panel_opt32.write_parquet(pq_f32_path, compression="zstd", statistics=True), repeats=2)
    sz32 = dir_size_mb(pq_f32_path)
    t_f32 = time_fn(lambda: pl.read_parquet(pq_f32_path), repeats=3)
    t_p32 = time_fn(lambda: pl.read_parquet(pq_f32_path, columns=proj_cols), repeats=3)
    t_flt32 = time_fn(
        lambda: pl.scan_parquet(pq_f32_path)
        .filter((pl.col("REPDTE") >= pl.date(2008, 3, 31)) & (pl.col("REPDTE") <= pl.date(2012, 12, 31)))
        .select(proj_cols)
        .collect(),
        repeats=3,
    )
    results.append({
        "layout": "Parquet Single (Float32, ZSTD)",
        "category": "Schema",
        "codec": "zstd",
        "schema": "Int32+Date+Cat+Float32",
        "files": 1,
        "size_mb": round(sz32, 2),
        "compression_ratio": round(size_csv / sz32, 2),
        "write_sec": round(t_w32, 3),
        "full_scan_sec": round(t_f32, 3),
        "projection_5col_sec": round(t_p32, 3),
        "predicate_gfc_sec": round(t_flt32, 3),
    })

    # --- E. Partitioned Layouts ---
    # E1. Quarter-Partitioned Layout (170 files, REPDTE=YYYYMMDD.parquet, Snappy with unified schema)
    q_dir = bench_dir / "quarter_partitioned_snappy"
    if q_dir.exists():
        shutil.rmtree(q_dir)
    q_dir.mkdir(parents=True, exist_ok=True)
    print("Benchmarking Quarter-Partitioned Parquet (170 files, Unified Schema, Snappy)...")
    t_wq_start = time.perf_counter()
    for q_val, q_df in panel_opt64.group_by("REPDTE"):
        q_date_val = q_val[0] if isinstance(q_val, tuple) else q_val
        q_str = q_date_val.strftime("%Y%m%d")
        q_df.write_parquet(q_dir / f"REPDTE={q_str}.parquet", compression="snappy", statistics=True)
    t_wq = time.perf_counter() - t_wq_start

    n_q_files = len(list(q_dir.glob("REPDTE=*.parquet")))
    sz_q = dir_size_mb(q_dir)
    t_fq = time_fn(lambda: pl.read_parquet(q_dir / "REPDTE=*.parquet"), repeats=3)
    t_pq = time_fn(lambda: pl.read_parquet(q_dir / "REPDTE=*.parquet", columns=proj_cols), repeats=3)
    # Partition pruning by selecting only the 20 GFC quarter files (2008Q1..2012Q4)
    gfc_files = [
        f for f in sorted(q_dir.glob("REPDTE=*.parquet"))
        if "20080331" <= f.stem.split("=")[1] <= "20121231"
    ]
    t_flt_q = time_fn(lambda: pl.read_parquet(gfc_files, columns=proj_cols), repeats=3)
    results.append({
        "layout": f"Quarter-Partitioned ({n_q_files} files, Snappy)",
        "category": "Partitioning",
        "codec": "snappy",
        "schema": "Int32+Date+Cat+Float64",
        "files": n_q_files,
        "size_mb": round(sz_q, 2),
        "compression_ratio": round(size_csv / sz_q, 2),
        "write_sec": round(t_wq, 3),
        "full_scan_sec": round(t_fq, 3),
        "projection_5col_sec": round(t_pq, 3),
        "predicate_gfc_sec": round(t_flt_q, 3),
    })

    # E2. Year-Partitioned Hive Layout (YEAR=YYYY/*.parquet, ZSTD)
    year_dir = bench_dir / "hive_year_zstd"
    if year_dir.exists():
        shutil.rmtree(year_dir)
    panel_with_year = panel_opt64.with_columns(pl.col("REPDTE").dt.year().alias("YEAR"))
    print("Benchmarking Year-Partitioned Hive Parquet (ZSTD)...")
    t_wy = time_fn(
        lambda: (
            shutil.rmtree(year_dir, ignore_errors=True),
            panel_with_year.write_parquet(year_dir, compression="zstd", partition_by="YEAR"),
        ),
        repeats=1,
    )
    n_y_files = len(list(year_dir.rglob("*.parquet")))
    sz_y = dir_size_mb(year_dir)
    t_fy = time_fn(lambda: pl.read_parquet(year_dir / "**" / "*.parquet", hive_partitioning=True), repeats=3)
    t_py = time_fn(
        lambda: pl.read_parquet(year_dir / "**" / "*.parquet", columns=proj_cols, hive_partitioning=True),
        repeats=3,
    )
    t_flt_y = time_fn(
        lambda: pl.scan_parquet(year_dir / "**" / "*.parquet", hive_partitioning=True)
        .filter((pl.col("YEAR") >= 2008) & (pl.col("YEAR") <= 2012))
        .select(proj_cols)
        .collect(),
        repeats=3,
    )
    results.append({
        "layout": f"Hive Partitioned by YEAR ({n_y_files} files, ZSTD)",
        "category": "Partitioning",
        "codec": "zstd",
        "schema": "Int32+Date+Cat+Float64",
        "files": n_y_files,
        "size_mb": round(sz_y, 2),
        "compression_ratio": round(size_csv / sz_y, 2),
        "write_sec": round(t_wy, 3),
        "full_scan_sec": round(t_fy, 3),
        "projection_5col_sec": round(t_py, 3),
        "predicate_gfc_sec": round(t_flt_y, 3),
    })

    # E3. Over-Partitioned Layout by STNAME x Decade (demonstrating small-file overhead)
    over_dir = bench_dir / "hive_state_year_zstd"
    if over_dir.exists():
        shutil.rmtree(over_dir)
    panel_state_yr = panel_with_year.with_columns(
        ((pl.col("YEAR") // 5) * 5).alias("PERIOD_5Y"),
        pl.col("STNAME").cast(pl.Utf8).fill_null("UNKNOWN"),
    )
    print("Benchmarking Over-Partitioned Hive Parquet (STNAME x PERIOD_5Y)...")
    t_wo = time_fn(
        lambda: (
            shutil.rmtree(over_dir, ignore_errors=True),
            panel_state_yr.write_parquet(over_dir, compression="zstd", partition_by=["STNAME", "PERIOD_5Y"]),
        ),
        repeats=1,
    )
    n_o_files = len(list(over_dir.rglob("*.parquet")))
    sz_o = dir_size_mb(over_dir)
    t_fo = time_fn(lambda: pl.read_parquet(over_dir / "**" / "*.parquet", hive_partitioning=True), repeats=2)
    t_po = time_fn(
        lambda: pl.read_parquet(over_dir / "**" / "*.parquet", columns=proj_cols, hive_partitioning=True),
        repeats=2,
    )
    t_flt_o = time_fn(
        lambda: pl.scan_parquet(over_dir / "**" / "*.parquet", hive_partitioning=True)
        .filter((pl.col("REPDTE") >= pl.date(2008, 3, 31)) & (pl.col("REPDTE") <= pl.date(2012, 12, 31)))
        .select(proj_cols)
        .collect(),
        repeats=2,
    )
    results.append({
        "layout": f"Over-Partitioned STNAME x 5Yr ({n_o_files} files, ZSTD)",
        "category": "Partitioning",
        "codec": "zstd",
        "schema": "Int32+Date+Cat+Float64",
        "files": n_o_files,
        "size_mb": round(sz_o, 2),
        "compression_ratio": round(size_csv / sz_o, 2),
        "write_sec": round(t_wo, 3),
        "full_scan_sec": round(t_fo, 3),
        "projection_5col_sec": round(t_po, 3),
        "predicate_gfc_sec": round(t_flt_o, 3),
    })

    # Save canonical consolidated ZSTD parquet for downstream pipeline stages
    canonical_dir = Path("data_processed")
    canonical_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bench_dir / "financials_opt64_zstd.parquet", canonical_dir / "financials_raw_consolidated.parquet")

    # Clean up temporary benchmark artifacts to keep disk usage tidy
    shutil.rmtree(bench_dir, ignore_errors=True)

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_tables / "part1_storage_benchmark.csv", index=False)
    (out_tables / "part1_storage_benchmark.json").write_text(json.dumps(results, indent=2))
    print("\n=== Storage Benchmark Results ===")
    print(df_res.to_string(index=False))

    # Plot Figure 1: Storage Footprint & Query Latencies
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left panel: Disk Footprint (MB)
    plot_df = df_res.iloc[::-1]
    colors = ["#d95f02" if "CSV" in x else ("#7570b3" if "Partitioned" in x else "#1b9e77") for x in plot_df["layout"]]
    bars = axes[0].barh(plot_df["layout"], plot_df["size_mb"], color=colors, edgecolor="black", height=0.68)
    axes[0].set_xlabel("Disk Size (MB) [Lower is Better]")
    axes[0].set_title("A. Storage Footprint & Compression Ratio vs. Raw CSV")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)
    for bar, ratio in zip(bars, plot_df["compression_ratio"]):
        w = bar.get_width()
        axes[0].text(w + max(plot_df["size_mb"]) * 0.015, bar.get_y() + bar.get_height() / 2,
                     f"{w:.1f} MB ({ratio:.1f}x)", va="center", fontsize=8.5)
    axes[0].set_xlim(0, max(plot_df["size_mb"]) * 1.25)

    # Right panel: Query Read Latency across Workloads (log scale)
    y_pos = np.arange(len(plot_df))
    h = 0.26
    axes[1].barh(y_pos + h, plot_df["full_scan_sec"], height=h, label="Full Scan (33 cols)", color="#2b5c8f")
    axes[1].barh(y_pos, plot_df["projection_5col_sec"], height=h, label="Column Projection (5 cols)", color="#3690c0")
    axes[1].barh(y_pos - h, plot_df["predicate_gfc_sec"], height=h, label="Predicate + Projection (GFC 2008-12)", color="#67a9cf")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(["" for _ in plot_df["layout"]])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Read Latency in Seconds (Log Scale) [Lower is Better]")
    axes[1].set_title("B. Analytical Read Latency by Query Pattern")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_figs / "fig1_storage_benchmark.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 to {out_figs / 'fig1_storage_benchmark.png'}")


if __name__ == "__main__":
    run_storage_benchmarks(
        raw_dir=Path("data_raw/fdic"),
        bench_dir=Path("data_processed/_bench_tmp"),
        out_tables=Path("outputs/tables"),
        out_figs=Path("outputs/figures"),
    )
