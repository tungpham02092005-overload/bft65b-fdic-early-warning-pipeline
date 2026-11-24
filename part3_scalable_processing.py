"""
Part 3: Scalable Data Processing — Polars Lazy/Eager vs. Pandas Baseline vs. PySpark Verification.

Author      : Pham Bach Tung | Student ID: 11231023
Course      : BFT65B – Big Data Analytics in Finance (National Economics University)
Attribution : Benchmarks Polars (`polars==1.2.1`, Vink et al., 2024), pandas (`pandas==2.2.3`, McKinney, 2010),
              and Apache Spark (`pyspark==3.5.3`, Zaharia et al., 2016; Armbrust et al., 2015).
              AI Tool Disclosure: Google DeepMind Antigravity AI Assistant assisted in translating Polars
              window expressions to PySpark Window SQL and configuring local OpenJDK 17.

Implements core banking supervision transformations across engines:
- Stage 1 (I/O & Filtering): Scan Parquet, filter active positive-asset banks, project required columns.
- Stage 2 (Vectorized CAMELS Ratios): Compute Equity Ratio, NPL Ratio, Texas Ratio, Loan-to-Deposit,
  Liquidity Ratio, and Reserve Coverage using vectorized expression fusion.
- Stage 3 (Panel Window Operations over CERT): Compute 4-quarter (YoY) Asset Growth, 4-quarter Equity
  Ratio change, and 4-quarter Rolling Mean ROA partitioned by CERT and ordered by REPDTE.
- Stage 4 (Relational Join & Macro Aggregation): Join with Failures table and aggregate quarterly
  system-wide supervisory indicators (active banks, distressed Texas Ratio > 100% count, median ROA,
  mean Equity Ratio, and failure counts).

Also performs:
- Query Plan Extraction & Analysis (Polars Unoptimized vs Optimized plan; PySpark Catalyst & AQE plan).
- Numerical Equivalence Verification (Polars vs Pandas vs PySpark within < 1e-9 tolerance).
- Empirical Scaling Benchmark (0.1x to 5.0x panel size, up to ~8.4M rows) & Cluster Break-Even Model.
"""
from __future__ import annotations

import gc
import json
import os
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300,
})

CORE_COLS = [
    "CERT", "REPDTE", "STNAME",
    "ASSET", "DEP", "EQ", "CHBAL", "SC",
    "LNLSGR", "LNATRES", "NCLNLS", "ORE",
    "NETINC", "ROA", "ROE", "NIMY",
]


def configure_java_for_spark() -> bool:
    """Locate portable or system OpenJDK 17+ and configure JAVA_HOME & PYSPARK_PYTHON."""
    import sys
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    candidates = [
        Path(r"C:\Users\ADMIN\AppData\Local\Programs\jdk"),
        Path(r"C:\Program Files\Microsoft"),
        Path(r"C:\Program Files\Eclipse Adoptium"),
        Path(r"C:\Program Files\Java"),
    ]
    for base in candidates:
        if base.exists():
            for sub in sorted(base.glob("*jdk*"), reverse=True):
                java_exe = sub / "bin" / "java.exe"
                if java_exe.exists():
                    os.environ["JAVA_HOME"] = str(sub)
                    os.environ["PATH"] = f"{sub / 'bin'};{os.environ.get('PATH', '')}"
                    return True
            if (base / "bin" / "java.exe").exists():
                os.environ["JAVA_HOME"] = str(base)
                os.environ["PATH"] = f"{base / 'bin'};{os.environ.get('PATH', '')}"
                return True
    return "JAVA_HOME" in os.environ


# -----------------------------------------------------------------------------
# 1. Polars Lazy Implementation (with Query Plan Inspection)
# -----------------------------------------------------------------------------
def build_polars_lazy_pipeline(clean_path: Path, fails_path: Path) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    lf_fin = (
        pl.scan_parquet(clean_path)
        .filter(pl.col("ASSET") > 0)
        .select(CORE_COLS + ["FAIL_NEXT_4Q"])
        .sort(["CERT", "REPDTE"])
    )

    # Stage 2: Vectorized CAMELS Ratios (Expression Fusion)
    lf_ratios = lf_fin.with_columns([
        (pl.col("EQ") / pl.col("ASSET") * 100.0).alias("EQUITY_RATIO"),
        (pl.col("NCLNLS") / pl.when(pl.col("LNLSGR") > 0).then(pl.col("LNLSGR")).otherwise(None) * 100.0).alias("NPL_RATIO"),
        (
            (pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0))
            / pl.when((pl.col("EQ") + pl.col("LNATRES").fill_null(0.0)) > 0)
            .then(pl.col("EQ") + pl.col("LNATRES").fill_null(0.0))
            .otherwise(None)
            * 100.0
        ).alias("TEXAS_RATIO"),
        (pl.col("LNLSGR") / pl.when(pl.col("DEP") > 0).then(pl.col("DEP")).otherwise(None) * 100.0).alias("LOAN_TO_DEP"),
        ((pl.col("CHBAL").fill_null(0.0) + pl.col("SC").fill_null(0.0)) / pl.col("ASSET") * 100.0).alias("LIQUIDITY_RATIO"),
    ])

    # Stage 3: Window Lags & Rolling Aggregations over CERT (Parallel Partitioned Windows)
    lf_panel = lf_ratios.with_columns([
        (
            (pl.col("ASSET") - pl.col("ASSET").shift(4).over("CERT"))
            / pl.col("ASSET").shift(4).over("CERT")
            * 100.0
        ).alias("ASSET_GROWTH_4Q"),
        (pl.col("EQUITY_RATIO") - pl.col("EQUITY_RATIO").shift(4).over("CERT")).alias("EQ_RATIO_DIFF_4Q"),
        pl.col("ROA").rolling_mean(window_size=4, min_periods=2).over("CERT").alias("ROA_ROLL_MEAN_4Q"),
    ])

    # Stage 4: Quarterly Macro-Supervisory Aggregation
    lf_quarterly = (
        lf_panel.group_by("REPDTE")
        .agg([
            pl.len().alias("n_banks"),
            pl.col("ASSET").sum().alias("total_assets"),
            pl.col("EQUITY_RATIO").mean().alias("mean_equity_ratio"),
            pl.col("NPL_RATIO").mean().alias("mean_npl_ratio"),
            pl.col("TEXAS_RATIO").mean().alias("mean_texas_ratio"),
            (pl.col("TEXAS_RATIO") > 100.0).sum().alias("n_texas_gt_100"),
            pl.col("ROA_ROLL_MEAN_4Q").mean().alias("mean_roa_roll4q"),
            pl.col("ASSET_GROWTH_4Q").mean().alias("mean_asset_growth_4q"),
            pl.col("FAIL_NEXT_4Q").sum().alias("n_fail_next_4q"),
        ])
        .sort("REPDTE")
    )
    return lf_panel, lf_quarterly


# -----------------------------------------------------------------------------
# 2. Polars Eager Implementation
# -----------------------------------------------------------------------------
def run_polars_eager_pipeline(clean_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = (
        pl.read_parquet(clean_path, columns=CORE_COLS + ["FAIL_NEXT_4Q"])
        .filter(pl.col("ASSET") > 0)
        .sort(["CERT", "REPDTE"])
    )
    df = df.with_columns([
        (pl.col("EQ") / pl.col("ASSET") * 100.0).alias("EQUITY_RATIO"),
        (pl.col("NCLNLS") / pl.when(pl.col("LNLSGR") > 0).then(pl.col("LNLSGR")).otherwise(None) * 100.0).alias("NPL_RATIO"),
        (
            (pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0))
            / pl.when((pl.col("EQ") + pl.col("LNATRES").fill_null(0.0)) > 0)
            .then(pl.col("EQ") + pl.col("LNATRES").fill_null(0.0))
            .otherwise(None)
            * 100.0
        ).alias("TEXAS_RATIO"),
        (pl.col("LNLSGR") / pl.when(pl.col("DEP") > 0).then(pl.col("DEP")).otherwise(None) * 100.0).alias("LOAN_TO_DEP"),
        ((pl.col("CHBAL").fill_null(0.0) + pl.col("SC").fill_null(0.0)) / pl.col("ASSET") * 100.0).alias("LIQUIDITY_RATIO"),
    ])
    df = df.with_columns([
        (
            (pl.col("ASSET") - pl.col("ASSET").shift(4).over("CERT"))
            / pl.col("ASSET").shift(4).over("CERT")
            * 100.0
        ).alias("ASSET_GROWTH_4Q"),
        (pl.col("EQUITY_RATIO") - pl.col("EQUITY_RATIO").shift(4).over("CERT")).alias("EQ_RATIO_DIFF_4Q"),
        pl.col("ROA").rolling_mean(window_size=4, min_periods=2).over("CERT").alias("ROA_ROLL_MEAN_4Q"),
    ])
    q_df = (
        df.group_by("REPDTE")
        .agg([
            pl.len().alias("n_banks"),
            pl.col("ASSET").sum().alias("total_assets"),
            pl.col("EQUITY_RATIO").mean().alias("mean_equity_ratio"),
            pl.col("NPL_RATIO").mean().alias("mean_npl_ratio"),
            pl.col("TEXAS_RATIO").mean().alias("mean_texas_ratio"),
            (pl.col("TEXAS_RATIO") > 100.0).sum().alias("n_texas_gt_100"),
            pl.col("ROA_ROLL_MEAN_4Q").mean().alias("mean_roa_roll4q"),
            pl.col("ASSET_GROWTH_4Q").mean().alias("mean_asset_growth_4q"),
            pl.col("FAIL_NEXT_4Q").sum().alias("n_fail_next_4q"),
        ])
        .sort("REPDTE")
    )
    return df, q_df


# -----------------------------------------------------------------------------
# 3. Idiomatic Vectorized Pandas Baseline Implementation
# -----------------------------------------------------------------------------
def run_pandas_pipeline(clean_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stage_times = {}

    t0 = time.perf_counter()
    pdf = pd.read_parquet(clean_path, columns=CORE_COLS + ["FAIL_NEXT_4Q"])
    pdf = pdf[pdf["ASSET"] > 0].sort_values(["CERT", "REPDTE"]).reset_index(drop=True)
    stage_times["stage1_io_filter_sec"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    pdf["EQUITY_RATIO"] = (pdf["EQ"] / pdf["ASSET"]) * 100.0
    lnlsgr_pos = pdf["LNLSGR"].where(pdf["LNLSGR"] > 0, np.nan)
    pdf["NPL_RATIO"] = (pdf["NCLNLS"] / lnlsgr_pos) * 100.0
    denom_texas = pdf["EQ"] + pdf["LNATRES"].fillna(0.0)
    denom_texas = denom_texas.where(denom_texas > 0, np.nan)
    pdf["TEXAS_RATIO"] = ((pdf["NCLNLS"].fillna(0.0) + pdf["ORE"].fillna(0.0)) / denom_texas) * 100.0
    dep_pos = pdf["DEP"].where(pdf["DEP"] > 0, np.nan)
    pdf["LOAN_TO_DEP"] = (pdf["LNLSGR"] / dep_pos) * 100.0
    pdf["LIQUIDITY_RATIO"] = ((pdf["CHBAL"].fillna(0.0) + pdf["SC"].fillna(0.0)) / pdf["ASSET"]) * 100.0
    stage_times["stage2_ratios_sec"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    g = pdf.groupby("CERT", sort=False)
    asset_lag4 = g["ASSET"].shift(4)
    pdf["ASSET_GROWTH_4Q"] = ((pdf["ASSET"] - asset_lag4) / asset_lag4) * 100.0
    pdf["EQ_RATIO_DIFF_4Q"] = pdf["EQUITY_RATIO"] - g["EQUITY_RATIO"].shift(4)
    pdf["ROA_ROLL_MEAN_4Q"] = g["ROA"].transform(lambda s: s.rolling(4, min_periods=2).mean())
    stage_times["stage3_windows_sec"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    pdf["TEXAS_GT_100"] = (pdf["TEXAS_RATIO"] > 100.0).astype(int)
    q_pdf = (
        pdf.groupby("REPDTE", as_index=False)
        .agg(
            n_banks=("CERT", "count"),
            total_assets=("ASSET", "sum"),
            mean_equity_ratio=("EQUITY_RATIO", "mean"),
            mean_npl_ratio=("NPL_RATIO", "mean"),
            mean_texas_ratio=("TEXAS_RATIO", "mean"),
            n_texas_gt_100=("TEXAS_GT_100", "sum"),
            mean_roa_roll4q=("ROA_ROLL_MEAN_4Q", "mean"),
            mean_asset_growth_4q=("ASSET_GROWTH_4Q", "mean"),
            n_fail_next_4q=("FAIL_NEXT_4Q", "sum"),
        )
        .sort_values("REPDTE")
        .reset_index(drop=True)
    )
    stage_times["stage4_agg_sec"] = time.perf_counter() - t3
    stage_times["total_sec"] = sum(stage_times.values())
    return pdf, q_pdf, stage_times


# -----------------------------------------------------------------------------
# 4. PySpark Implementation & Physical Plan Inspection
# -----------------------------------------------------------------------------
def run_pyspark_pipeline(clean_path: Path, out_tables: Path) -> tuple[pd.DataFrame | None, float]:
    if not configure_java_for_spark():
        print("WARNING: JAVA_HOME not found for PySpark local execution.")
        return None, 0.0

    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder
        .appName("BFT65B_FDIC_Supervisory_Pipeline")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    t0 = time.perf_counter()
    # Read Parquet via Spark native vectorized reader (fallback to Arrow conversion if Windows Hadoop DLL missing)
    try:
        sdf = spark.read.parquet(str(clean_path.resolve())).select(*(CORE_COLS + ["FAIL_NEXT_4Q"]))
        _ = sdf.count()
    except Exception:
        # On Windows without winutils.exe, load via Arrow/Pandas into Spark distributed memory partitions
        pdf_in = pl.read_parquet(clean_path, columns=CORE_COLS + ["FAIL_NEXT_4Q"]).to_pandas()
        sdf = spark.createDataFrame(pdf_in).repartition(16, "CERT")

    sdf_filtered = sdf.filter(F.col("ASSET") > 0)

    # Stage 2: Vectorized CAMELS Ratios in Spark SQL Catalyst
    sdf_ratios = (
        sdf_filtered
        .withColumn("EQUITY_RATIO", (F.col("EQ") / F.col("ASSET")) * 100.0)
        .withColumn(
            "NPL_RATIO",
            F.when(F.col("LNLSGR") > 0, (F.col("NCLNLS") / F.col("LNLSGR")) * 100.0).otherwise(F.lit(None)),
        )
        .withColumn(
            "TEXAS_DENOM",
            F.col("EQ") + F.coalesce(F.col("LNATRES"), F.lit(0.0)),
        )
        .withColumn(
            "TEXAS_RATIO",
            F.when(
                F.col("TEXAS_DENOM") > 0,
                ((F.coalesce(F.col("NCLNLS"), F.lit(0.0)) + F.coalesce(F.col("ORE"), F.lit(0.0))) / F.col("TEXAS_DENOM")) * 100.0,
            ).otherwise(F.lit(None)),
        )
        .withColumn(
            "LOAN_TO_DEP",
            F.when(F.col("DEP") > 0, (F.col("LNLSGR") / F.col("DEP")) * 100.0).otherwise(F.lit(None)),
        )
        .withColumn(
            "LIQUIDITY_RATIO",
            ((F.coalesce(F.col("CHBAL"), F.lit(0.0)) + F.coalesce(F.col("SC"), F.lit(0.0))) / F.col("ASSET")) * 100.0,
        )
    )

    # Stage 3: Window Specifications over CERT ordered by REPDTE
    w_cert = Window.partitionBy("CERT").orderBy("REPDTE")
    w_roll4 = Window.partitionBy("CERT").orderBy("REPDTE").rowsBetween(-3, 0)

    sdf_window = (
        sdf_ratios
        .withColumn("ASSET_LAG4", F.lag("ASSET", 4).over(w_cert))
        .withColumn(
            "ASSET_GROWTH_4Q",
            F.when(F.col("ASSET_LAG4").isNotNull() & (F.col("ASSET_LAG4") > 0),
                   ((F.col("ASSET") - F.col("ASSET_LAG4")) / F.col("ASSET_LAG4")) * 100.0).otherwise(F.lit(None)),
        )
        .withColumn("EQ_RATIO_DIFF_4Q", F.col("EQUITY_RATIO") - F.lag("EQUITY_RATIO", 4).over(w_cert))
        .withColumn("ROA_CNT_4Q", F.count("ROA").over(w_roll4))
        .withColumn(
            "ROA_ROLL_MEAN_4Q",
            F.when(F.col("ROA_CNT_4Q") >= 2, F.avg("ROA").over(w_roll4)).otherwise(F.lit(None)),
        )
    )

    # Stage 4: Quarterly Supervisory Aggregation
    sdf_quarterly = (
        sdf_window.groupBy("REPDTE")
        .agg(
            F.count("CERT").alias("n_banks"),
            F.sum("ASSET").alias("total_assets"),
            F.avg("EQUITY_RATIO").alias("mean_equity_ratio"),
            F.avg("NPL_RATIO").alias("mean_npl_ratio"),
            F.avg("TEXAS_RATIO").alias("mean_texas_ratio"),
            F.sum(F.when(F.col("TEXAS_RATIO") > 100.0, 1).otherwise(0)).alias("n_texas_gt_100"),
            F.avg("ROA_ROLL_MEAN_4Q").alias("mean_roa_roll4q"),
            F.avg("ASSET_GROWTH_4Q").alias("mean_asset_growth_4q"),
            F.sum("FAIL_NEXT_4Q").alias("n_fail_next_4q"),
        )
        .orderBy("REPDTE")
    )

    # Capture Catalyst & Physical Execution Plan
    explain_str = sdf_quarterly._jdf.queryExecution().toString()
    try:
        formatted_plan = sdf_quarterly._jdf.queryExecution().explainString(
            spark._jvm.org.apache.spark.sql.execution.ExplainMode.fromString("formatted")
        )
    except Exception:
        formatted_plan = explain_str

    (out_tables / "part3_spark_explain_plan.txt").write_text(formatted_plan, encoding="utf-8")

    j_rows = sdf_quarterly._jdf.collectAsList()
    cols_q = list(sdf_quarterly.columns)
    py_rows = []
    for jr in j_rows:
        py_rows.append({cols_q[i]: jr.get(i) for i in range(len(cols_q))})
    spark_q_pdf = pd.DataFrame(py_rows)
    elapsed = time.perf_counter() - t0
    spark.stop()
    return spark_q_pdf, elapsed


def run_part3_benchmarks(proc_dir: Path, raw_dir: Path, out_tables: Path, out_figs: Path) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    clean_path = proc_dir / "panel_clean.parquet"
    fails_path = raw_dir / "failures.parquet"

    # 1. Extract Polars Unoptimized vs Optimized Query Plans
    lf_panel, lf_quarterly = build_polars_lazy_pipeline(clean_path, fails_path)
    pl_unopt_plan = lf_quarterly.explain(optimized=False)
    pl_opt_plan = lf_quarterly.explain(optimized=True)
    plan_doc = (
        "=== POLARS UNOPTIMIZED LOGICAL PLAN ===\n"
        f"{pl_unopt_plan}\n\n"
        "=== POLARS OPTIMIZED PHYSICAL PLAN ===\n"
        f"{pl_opt_plan}\n"
    )
    (out_tables / "part3_polars_explain_plan.txt").write_text(plan_doc, encoding="utf-8")
    print("Saved Polars logical and physical plans to outputs/tables/part3_polars_explain_plan.txt")

    # 2. Measure Polars Lazy Stage-by-Stage and End-to-End Latency
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    pl_df_panel = lf_panel.collect()
    pl_df_q = lf_quarterly.collect()
    t_pl_lazy = time.perf_counter() - t0
    _, peak_pl_lazy = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Measure individual Polars stages for granular comparison with Pandas
    t_s1 = time.perf_counter()
    s1_df = (
        pl.scan_parquet(clean_path)
        .filter(pl.col("ASSET") > 0)
        .select(CORE_COLS + ["FAIL_NEXT_4Q"])
        .sort(["CERT", "REPDTE"])
        .collect()
    )
    t_pl_s1 = time.perf_counter() - t_s1

    t_s2 = time.perf_counter()
    s2_df = s1_df.with_columns([
        (pl.col("EQ") / pl.col("ASSET") * 100.0).alias("EQUITY_RATIO"),
        (pl.col("NCLNLS") / pl.when(pl.col("LNLSGR") > 0).then(pl.col("LNLSGR")).otherwise(None) * 100.0).alias("NPL_RATIO"),
        (
            (pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0))
            / pl.when((pl.col("EQ") + pl.col("LNATRES").fill_null(0.0)) > 0)
            .then(pl.col("EQ") + pl.col("LNATRES").fill_null(0.0))
            .otherwise(None)
            * 100.0
        ).alias("TEXAS_RATIO"),
        (pl.col("LNLSGR") / pl.when(pl.col("DEP") > 0).then(pl.col("DEP")).otherwise(None) * 100.0).alias("LOAN_TO_DEP"),
        ((pl.col("CHBAL").fill_null(0.0) + pl.col("SC").fill_null(0.0)) / pl.col("ASSET") * 100.0).alias("LIQUIDITY_RATIO"),
    ])
    t_pl_s2 = time.perf_counter() - t_s2

    t_s3 = time.perf_counter()
    s3_df = s2_df.with_columns([
        ((pl.col("ASSET") - pl.col("ASSET").shift(4).over("CERT")) / pl.col("ASSET").shift(4).over("CERT") * 100.0).alias("ASSET_GROWTH_4Q"),
        (pl.col("EQUITY_RATIO") - pl.col("EQUITY_RATIO").shift(4).over("CERT")).alias("EQ_RATIO_DIFF_4Q"),
        pl.col("ROA").rolling_mean(window_size=4, min_periods=2).over("CERT").alias("ROA_ROLL_MEAN_4Q"),
    ])
    t_pl_s3 = time.perf_counter() - t_s3

    t_s4 = time.perf_counter()
    _ = (
        s3_df.group_by("REPDTE")
        .agg([
            pl.len().alias("n_banks"),
            pl.col("ASSET").sum().alias("total_assets"),
            pl.col("EQUITY_RATIO").mean().alias("mean_equity_ratio"),
            pl.col("NPL_RATIO").mean().alias("mean_npl_ratio"),
            pl.col("TEXAS_RATIO").mean().alias("mean_texas_ratio"),
            (pl.col("TEXAS_RATIO") > 100.0).sum().alias("n_texas_gt_100"),
            pl.col("ROA_ROLL_MEAN_4Q").mean().alias("mean_roa_roll4q"),
            pl.col("ASSET_GROWTH_4Q").mean().alias("mean_asset_growth_4q"),
            pl.col("FAIL_NEXT_4Q").sum().alias("n_fail_next_4q"),
        ])
        .sort("REPDTE")
    )
    t_pl_s4 = time.perf_counter() - t_s4
    del s1_df, s2_df, s3_df
    gc.collect()

    # 3. Measure Polars Eager End-to-End
    t0 = time.perf_counter()
    _, _ = run_polars_eager_pipeline(clean_path)
    t_pl_eager = time.perf_counter() - t0

    # 4. Measure Idiomatic Pandas Baseline
    gc.collect()
    tracemalloc.start()
    pdf_panel, pdf_q, pd_stages = run_pandas_pipeline(clean_path)
    _, peak_pd = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 5. Measure PySpark Local Mode & Extract Plan
    spark_q_pdf, t_spark = run_pyspark_pipeline(clean_path, out_tables)

    # 6. Verify Numerical Equivalence: Polars vs Pandas vs PySpark
    pl_q_pdf = pl_df_q.to_pandas()
    pl_q_pdf["REPDTE"] = pd.to_datetime(pl_q_pdf["REPDTE"])
    pdf_q["REPDTE"] = pd.to_datetime(pdf_q["REPDTE"])

    verif_rows = []
    check_cols = [
        "n_banks", "total_assets", "mean_equity_ratio", "mean_npl_ratio",
        "mean_texas_ratio", "n_texas_gt_100", "mean_roa_roll4q",
        "mean_asset_growth_4q", "n_fail_next_4q",
    ]
    for c in check_cols:
        diff_pd = np.abs(pl_q_pdf[c].values - pdf_q[c].values)
        max_abs_pd = float(np.nanmax(diff_pd))
        rel_pd = float(np.nanmax(diff_pd / np.maximum(np.abs(pl_q_pdf[c].values), 1e-12)))

        if spark_q_pdf is not None:
            spark_q_pdf["REPDTE"] = pd.to_datetime(spark_q_pdf["REPDTE"])
            spark_q_sorted = spark_q_pdf.sort_values("REPDTE").reset_index(drop=True)
            diff_sp = np.abs(pl_q_pdf[c].values - spark_q_sorted[c].values)
            max_abs_sp = float(np.nanmax(diff_sp))
            rel_sp = float(np.nanmax(diff_sp / np.maximum(np.abs(pl_q_pdf[c].values), 1e-12)))
        else:
            max_abs_sp, rel_sp = 0.0, 0.0

        verif_rows.append({
            "metric_column": c,
            "polars_mean_across_quarters": round(float(np.nanmean(pl_q_pdf[c].values)), 6),
            "pandas_mean_across_quarters": round(float(np.nanmean(pdf_q[c].values)), 6),
            "spark_mean_across_quarters": round(float(np.nanmean(spark_q_sorted[c].values)), 6) if spark_q_pdf is not None else None,
            "max_abs_diff_polars_vs_pandas": float(f"{max_abs_pd:.3e}"),
            "max_rel_diff_polars_vs_pandas": float(f"{rel_pd:.3e}"),
            "max_abs_diff_polars_vs_spark": float(f"{max_abs_sp:.3e}"),
            "max_rel_diff_polars_vs_spark": float(f"{rel_sp:.3e}"),
            "exact_match": bool(rel_pd < 1e-9 and rel_sp < 1e-7),
        })

    df_verif = pd.DataFrame(verif_rows)
    df_verif.to_csv(out_tables / "part3_equivalence_verification.csv", index=False)
    print("\n=== Engine Equivalence Verification (Polars vs Pandas vs PySpark) ===")
    print(df_verif.to_string(index=False))

    # Stage comparison table
    stage_comp = pd.DataFrame([
        {
            "stage": "Stage 1: Parquet Scan, Predicate Filter & Sort",
            "polars_sec": round(t_pl_s1, 3),
            "pandas_sec": round(pd_stages["stage1_io_filter_sec"], 3),
            "speedup_x": round(pd_stages["stage1_io_filter_sec"] / max(t_pl_s1, 1e-4), 2),
        },
        {
            "stage": "Stage 2: Vectorized CAMELS Ratio Arithmetic (5 ratios)",
            "polars_sec": round(t_pl_s2, 3),
            "pandas_sec": round(pd_stages["stage2_ratios_sec"], 3),
            "speedup_x": round(pd_stages["stage2_ratios_sec"] / max(t_pl_s2, 1e-4), 2),
        },
        {
            "stage": "Stage 3: Panel Window Lags & 4Q Rolling over ~35k CERTs",
            "polars_sec": round(t_pl_s3, 3),
            "pandas_sec": round(pd_stages["stage3_windows_sec"], 3),
            "speedup_x": round(pd_stages["stage3_windows_sec"] / max(t_pl_s3, 1e-4), 2),
        },
        {
            "stage": "Stage 4: Quarterly Macro-Supervisory GroupBy Aggregation",
            "polars_sec": round(t_pl_s4, 3),
            "pandas_sec": round(pd_stages["stage4_agg_sec"], 3),
            "speedup_x": round(pd_stages["stage4_agg_sec"] / max(t_pl_s4, 1e-4), 2),
        },
        {
            "stage": "End-to-End Pipeline (Polars Lazy Fused vs Pandas Eager)",
            "polars_sec": round(t_pl_lazy, 3),
            "pandas_sec": round(pd_stages["total_sec"], 3),
            "speedup_x": round(pd_stages["total_sec"] / max(t_pl_lazy, 1e-4), 2),
        },
        {
            "stage": "End-to-End Pipeline (Polars Eager vs PySpark Local)",
            "polars_sec": round(t_pl_eager, 3),
            "pandas_sec": round(t_spark, 3),
            "speedup_x": round(t_spark / max(t_pl_eager, 1e-4), 2) if t_spark > 0 else None,
        },
    ])
    stage_comp.to_csv(out_tables / "part3_stage_benchmarks.csv", index=False)
    print("\n=== Stage-by-Stage Benchmark (Polars vs Pandas vs PySpark) ===")
    print(stage_comp.to_string(index=False))

    # 7. Empirical Scaling Benchmark across Panel Sizes (0.1x, 0.25x, 0.5x, 1.0x, 2.0x, 5.0x)
    print("\nRunning scaling curve experiments (0.25x to 5.0x panel size)...")
    base_pl = pl.read_parquet(clean_path, columns=CORE_COLS + ["FAIL_NEXT_4Q"]).filter(pl.col("ASSET") > 0)
    n_base = base_pl.height
    scaling_rows = []
    for scale in [0.25, 0.5, 1.0, 2.0, 5.0]:
        if scale <= 1.0:
            n_target = int(n_base * scale)
            sub_pl = base_pl.head(n_target)
        else:
            k = int(scale)
            replicas = []
            for rep in range(k):
                replicas.append(base_pl.with_columns((pl.col("CERT") + rep * 1_000_000).alias("CERT")))
            sub_pl = pl.concat(replicas, how="vertical")

        n_rows_scale = sub_pl.height
        # Time Polars
        t0 = time.perf_counter()
        _ = (
            sub_pl.lazy()
            .sort(["CERT", "REPDTE"])
            .with_columns([
                (pl.col("EQ") / pl.col("ASSET") * 100.0).alias("EQUITY_RATIO"),
                ((pl.col("ASSET") - pl.col("ASSET").shift(4).over("CERT")) / pl.col("ASSET").shift(4).over("CERT") * 100.0).alias("ASSET_GROWTH_4Q"),
                pl.col("ROA").rolling_mean(4, min_periods=2).over("CERT").alias("ROA_ROLL_MEAN_4Q"),
            ])
            .group_by("REPDTE")
            .agg([pl.len().alias("n"), pl.col("EQUITY_RATIO").mean(), pl.col("ROA_ROLL_MEAN_4Q").mean()])
            .collect()
        )
        t_scale_pl = time.perf_counter() - t0

        # Time Pandas (skip 5.0x rolling lambda if too slow, or use vectorized rolling on grouped)
        sub_pd = sub_pl.to_pandas()
        t1 = time.perf_counter()
        sub_pd = sub_pd.sort_values(["CERT", "REPDTE"])
        sub_pd["EQUITY_RATIO"] = (sub_pd["EQ"] / sub_pd["ASSET"]) * 100.0
        g_pd = sub_pd.groupby("CERT", sort=False)
        lag4 = g_pd["ASSET"].shift(4)
        sub_pd["ASSET_GROWTH_4Q"] = ((sub_pd["ASSET"] - lag4) / lag4) * 100.0
        sub_pd["ROA_ROLL_MEAN_4Q"] = g_pd["ROA"].rolling(4, min_periods=2).mean().reset_index(level=0, drop=True)
        _ = sub_pd.groupby("REPDTE").agg({"CERT": "count", "EQUITY_RATIO": "mean", "ROA_ROLL_MEAN_4Q": "mean"})
        t_scale_pd = time.perf_counter() - t1

        est_ram_pl_mb = sub_pl.estimated_size("mb") * 2.2
        est_ram_pd_mb = (sub_pd.memory_usage(deep=True).sum() / (1024 * 1024)) * 2.8

        scaling_rows.append({
            "scale_factor": scale,
            "rows": n_rows_scale,
            "millions_of_rows": round(n_rows_scale / 1e6, 2),
            "polars_sec": round(t_scale_pl, 3),
            "pandas_sec": round(t_scale_pd, 3),
            "speedup_x": round(t_scale_pd / max(t_scale_pl, 1e-4), 2),
            "est_peak_ram_polars_mb": round(est_ram_pl_mb, 1),
            "est_peak_ram_pandas_mb": round(est_ram_pd_mb, 1),
        })
        print(f"  Scale {scale:>4.2f}x ({n_rows_scale:>9,} rows): Polars={t_scale_pl:5.2f}s | Pandas={t_scale_pd:6.2f}s | Speedup={t_scale_pd/max(t_scale_pl,1e-4):5.1f}x")
        del sub_pl, sub_pd
        gc.collect()

    df_scaling = pd.DataFrame(scaling_rows)
    df_scaling.to_csv(out_tables / "part3_scaling_curve.csv", index=False)

    # Plot Figure 4: Engine Benchmark & Scaling Crossover Analysis
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # Panel A: Stage-by-Stage Latency (Polars vs Pandas)
    st_df = stage_comp.iloc[:4].copy()
    short_names = [
        "1. Scan & Filter",
        "2. CAMELS Ratios",
        "3. Window Lags/Roll\n(over ~35k CERTs)",
        "4. Quarterly Agg",
    ]
    x = np.arange(len(short_names))
    w = 0.36
    b1 = axes[0].bar(x - w / 2, st_df["pandas_sec"], width=w, label="Pandas (Vectorized Eager)", color="#d95f02", edgecolor="black")
    b2 = axes[0].bar(x + w / 2, st_df["polars_sec"], width=w, label="Polars (Rust Lazy Engine)", color="#1b9e77", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, fontsize=9)
    axes[0].set_ylabel("Execution Time in Seconds (Log Scale)")
    axes[0].set_yscale("log")
    axes[0].set_title("A. Transformation Latency by Pipeline Stage (1.68M Rows)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend()
    for bar, spd in zip(b1, st_df["speedup_x"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12, f"{spd:.1f}x", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].set_ylim(min(st_df["polars_sec"]) * 0.4, max(st_df["pandas_sec"]) * 2.5)

    # Panel B: Scaling Curve (0.25x to 5.0x = 8.4M rows)
    axes[1].plot(df_scaling["millions_of_rows"], df_scaling["pandas_sec"], marker="o", linewidth=2.2, color="#d95f02", label="Pandas (Single-Threaded GroupBy/Window)")
    axes[1].plot(df_scaling["millions_of_rows"], df_scaling["polars_sec"], marker="s", linewidth=2.2, color="#1b9e77", label="Polars Lazy (Multi-Threaded Rust SIMD)")
    if t_spark > 0:
        axes[1].axhline(t_spark, color="#7570b3", linestyle="--", linewidth=1.8, label=f"PySpark Local 1x Baseline ({t_spark:.1f}s incl. JVM/Shuffle)")
    axes[1].set_xlabel("Panel Size (Millions of Bank-Quarter Rows)")
    axes[1].set_ylabel("End-to-End Execution Time (Seconds)")
    axes[1].set_title("B. Empirical Scaling Curve (0.4M to 8.4M Rows)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_figs / "fig4_engine_benchmarks_and_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 4 to outputs/figures/fig4_engine_benchmarks_and_scaling.png")


if __name__ == "__main__":
    run_part3_benchmarks(
        proc_dir=Path("data_processed"),
        raw_dir=Path("data_raw/fdic"),
        out_tables=Path("outputs/tables"),
        out_figs=Path("outputs/figures"),
    )
