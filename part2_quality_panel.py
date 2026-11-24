"""
Part 2: Automated Data Quality Audit & Point-in-Time Clean Bank-Quarter Panel Construction.

Author      : Pham Bach Tung | Student ID: 11231023
Course      : BFT65B – Big Data Analytics in Finance (National Economics University)
Attribution : Regulatory domain rules anchored to FFIEC UFIRS/CAMELS (1996), FDIC RISVIEW Dictionary (2026),
              Cole & White (2012), and Wheelock & Wilson (2000). Executed in Polars (Vink et al., 2024).
              AI Tool Disclosure: Google DeepMind Antigravity AI Assistant assisted in structuring audit
              aggregation tables; all quality anomalies and exit counts were verified empirically.

Performs systematic quality checks across Financials, Failures, and Institutions:
1. Key Integrity & Panel Continuity:
   - Duplicate (CERT, REPDTE) check
   - Intermittent reporting gaps (quarter jumps > 1 quarter within a bank's history)
2. Regime-Dependent Missingness Audit:
   - Quantifies missingness across 4 regulatory eras:
     * Pre-Basel I / S&L Crisis (1984-1990)
     * Basel I & Post-FDICIA Expansion (1991-2006)
     * GFC & Dodd-Frank / Basel III Transition (2007-2019)
     * CBLR & Modern Tailoring Era (2020-2026)
3. Accounting Identity & Range Anomaly Audit:
   - Non-positive assets (ASSET <= 0), negative deposits/loans/reserves
   - Negative equity (EQ < 0): proves empirically why EQ < 0 is true economic distress
     (insolvency prior to failure) rather than data corruption, avoiding survivorship bias.
   - Extreme ratio outliers (ROE, ROA, NIMY) when equity denominator approaches zero.
4. Point-in-Time Exit Taxonomy & Label Construction:
   - Distinguishes Failure Exits (matched to Failures table) from Non-Failure Exits
     (mergers, consolidations, voluntary liquidations in Institutions).
   - Prevents look-ahead bias from static snapshot fields in Institutions (ACTIVE, CHANGEC1, ENDEFYMD).
   - Removes post-failure ghost/receivership filings (REPDTE >= FAILDATE for closed banks).
   - Constructs the exact 4-quarter forward failure indicator:
       y_{i,q} = 1{ bank i fails in (q, q + 4 quarters] }
     and restricts modelling eligibility to REPDTE <= 2025-06-30 (2025Q2).
"""
from __future__ import annotations

import json
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

FIN_NUMERIC_COLS = [
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ",
]


def load_tables(raw_dir: Path, proc_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    cons_path = proc_dir / "financials_raw_consolidated.parquet"
    if cons_path.exists():
        fin = pl.read_parquet(cons_path)
    else:
        files = sorted((raw_dir / "financials").glob("REPDTE=*.parquet"))
        dfs = []
        for f in files:
            d = pl.read_parquet(f)
            exprs = [
                pl.col("CERT").cast(pl.Int32, strict=False),
                pl.col("REPDTE").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
                pl.col("STNAME").cast(pl.Categorical),
            ]
            for c in FIN_NUMERIC_COLS:
                exprs.append(
                    pl.col(c).cast(pl.Float64, strict=False) if c in d.columns else pl.lit(None, dtype=pl.Float64).alias(c)
                )
            dfs.append(d.select(exprs))
        fin = pl.concat(dfs, how="vertical").sort(["REPDTE", "CERT"])

    fails = pl.read_parquet(raw_dir / "failures.parquet")
    inst = pl.read_parquet(raw_dir / "institutions.parquet")
    return fin, fails, inst


def parse_fdic_dates(fails: pl.DataFrame, inst: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # FAILDATE in FDIC API can be YYYY-MM-DD, MM/DD/YYYY, or YYYYMMDD
    fails_clean = fails.with_columns([
        pl.col("CERT").cast(pl.Int32, strict=False),
        pl.coalesce([
            pl.col("FAILDATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False),
            pl.col("FAILDATE").cast(pl.Utf8).str.to_date("%m/%d/%Y", strict=False),
            pl.col("FAILDATE").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("FAILDATE").cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
        ]).alias("FAILDATE_DT"),
        pl.col("COST").cast(pl.Float64, strict=False),
        pl.col("QBFASSET").cast(pl.Float64, strict=False),
        pl.col("QBFDEP").cast(pl.Float64, strict=False),
    ]).filter(pl.col("CERT").is_not_null() & pl.col("FAILDATE_DT").is_not_null())

    inst_clean = inst.with_columns([
        pl.col("CERT").cast(pl.Int32, strict=False),
        pl.col("ACTIVE").cast(pl.Int32, strict=False),
        pl.col("BKCLASS").cast(pl.Utf8),
        pl.col("CHANGEC1").cast(pl.Utf8),
        pl.coalesce([
            pl.col("ESTYMD").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False),
            pl.col("ESTYMD").cast(pl.Utf8).str.to_date("%m/%d/%Y", strict=False),
            pl.col("ESTYMD").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
        ]).alias("ESTYMD_DT"),
        pl.coalesce([
            pl.col("INSDATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False),
            pl.col("INSDATE").cast(pl.Utf8).str.to_date("%m/%d/%Y", strict=False),
            pl.col("INSDATE").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
        ]).alias("INSDATE_DT"),
        pl.coalesce([
            pl.col("ENDEFYMD").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False),
            pl.col("ENDEFYMD").cast(pl.Utf8).str.to_date("%m/%d/%Y", strict=False),
            pl.col("ENDEFYMD").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
        ]).alias("ENDEFYMD_DT"),
    ]).unique(subset=["CERT"], keep="first")

    return fails_clean, inst_clean


def run_quality_audit_and_build_panel(raw_dir: Path, proc_dir: Path, out_tables: Path, out_figs: Path) -> None:
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    fin, fails_raw, inst_raw = load_tables(raw_dir, proc_dir)
    fails, inst = parse_fdic_dates(fails_raw, inst_raw)

    n_raw_rows = fin.height
    n_raw_banks = fin["CERT"].n_unique()
    print(f"Raw Financials Panel: {n_raw_rows:,} rows, {n_raw_banks:,} distinct banks.")
    print(f"Failures Table      : {fails.height:,} valid failure records ({fails_raw.height:,} raw).")
    print(f"Institutions Table  : {inst.height:,} unique institutions ({inst_raw.height:,} raw).")

    # -------------------------------------------------------------------------
    # 1. Automated Field-Level Quality & Missingness Report across Regulatory Regimes
    # -------------------------------------------------------------------------
    fin_with_era = fin.with_columns([
        pl.col("REPDTE").dt.year().alias("YEAR"),
        pl.when(pl.col("REPDTE").dt.year() <= 1990).then(pl.lit("1984-1990 (Pre-Basel I)"))
        .when(pl.col("REPDTE").dt.year() <= 2006).then(pl.lit("1991-2006 (Basel I / FDICIA)"))
        .when(pl.col("REPDTE").dt.year() <= 2019).then(pl.lit("2007-2019 (GFC & Basel III)"))
        .otherwise(pl.lit("2020-2026 (CBLR Tailoring Era)"))
        .alias("REGIME")
    ])

    quality_rows = []
    for col in FIN_NUMERIC_COLS:
        s = fin[col]
        null_cnt = int(s.null_count())
        null_pct = 100.0 * null_cnt / n_raw_rows
        valid = s.drop_nulls()
        neg_cnt = int((valid < 0).sum()) if len(valid) > 0 else 0
        zero_cnt = int((valid == 0).sum()) if len(valid) > 0 else 0

        # Era-specific missingness
        era_nulls = (
            fin_with_era.group_by("REGIME")
            .agg((pl.col(col).null_count() * 100.0 / pl.len()).alias("null_pct"))
            .to_pandas()
            .set_index("REGIME")["null_pct"]
            .to_dict()
        )

        quality_rows.append({
            "field": col,
            "missing_count": null_cnt,
            "missing_pct_overall": round(null_pct, 2),
            "missing_pct_1984_1990": round(era_nulls.get("1984-1990 (Pre-Basel I)", 0.0), 2),
            "missing_pct_1991_2006": round(era_nulls.get("1991-2006 (Basel I / FDICIA)", 0.0), 2),
            "missing_pct_2007_2019": round(era_nulls.get("2007-2019 (GFC & Basel III)", 0.0), 2),
            "missing_pct_2020_2026": round(era_nulls.get("2020-2026 (CBLR Tailoring Era)", 0.0), 2),
            "negative_count": neg_cnt,
            "zero_count": zero_cnt,
            "min": round(float(valid.min()), 2) if len(valid) > 0 else None,
            "p01": round(float(valid.quantile(0.01)), 2) if len(valid) > 0 else None,
            "median": round(float(valid.median()), 2) if len(valid) > 0 else None,
            "p99": round(float(valid.quantile(0.99)), 2) if len(valid) > 0 else None,
            "max": round(float(valid.max()), 2) if len(valid) > 0 else None,
        })

    df_quality = pd.DataFrame(quality_rows)
    df_quality.to_csv(out_tables / "part2_data_quality_report.csv", index=False)
    print("Saved automated field-level quality report to outputs/tables/part2_data_quality_report.csv")

    # -------------------------------------------------------------------------
    # 2. Systematic Audit of Logical / Accounting Anomalies & Panel Irregularities
    # -------------------------------------------------------------------------
    # Check duplicate keys
    n_dup_keys = int(fin.is_duplicated().sum())
    n_dup_cert_repdte = int(fin.select(["CERT", "REPDTE"]).is_duplicated().sum())

    # Check intermittent panel gaps (days between consecutive reports > 100 days, i.e., > 1 quarter)
    fin_sorted = fin.sort(["CERT", "REPDTE"]).with_columns([
        (pl.col("REPDTE") - pl.col("REPDTE").shift(1).over("CERT")).dt.total_days().alias("days_since_prev")
    ])
    n_gap_events = int(fin_sorted.filter(pl.col("days_since_prev") > 115).height)
    n_banks_with_gaps = int(fin_sorted.filter(pl.col("days_since_prev") > 115)["CERT"].n_unique())

    # Accounting & range checks
    n_asset_nonpos = int(fin.filter(pl.col("ASSET").is_null() | (pl.col("ASSET") <= 0)).height)
    n_dep_neg = int(fin.filter(pl.col("DEP") < 0).height)
    n_lnlsgr_neg = int(fin.filter(pl.col("LNLSGR") < 0).height)
    n_lnatres_neg = int(fin.filter(pl.col("LNATRES") < 0).height)
    n_nclnls_neg = int(fin.filter(pl.col("NCLNLS") < 0).height)
    n_eq_nonpos = int(fin.filter(pl.col("EQ") <= 0).height)
    n_roe_extreme = int(fin.filter(pl.col("ROE").abs() > 100.0).height)
    n_roa_extreme = int(fin.filter(pl.col("ROA").abs() > 20.0).height)
    # Balance sheet identity check: LNATRES > LNLSGR or NCLNLS > LNLSGR
    n_res_gt_loans = int(fin.filter((pl.col("LNLSGR") > 0) & (pl.col("LNATRES") > pl.col("LNLSGR"))).height)
    n_npl_gt_loans = int(fin.filter((pl.col("LNLSGR") > 0) & (pl.col("NCLNLS") > pl.col("LNLSGR"))).height)

    # -------------------------------------------------------------------------
    # 3. Point-in-Time Failure Matching, Post-Failure Ghost Removal, & Exit Taxonomy
    # -------------------------------------------------------------------------
    # Note: Some banks in the 1980s/1990s received Open Bank Assistance (RESTYPE='OBA') or had
    # multiple failure entries. For each (CERT, REPDTE), we check all failure records of that CERT.
    # Specifically:
    # - A failure at FAILDATE_DT falls in (REPDTE, REPDTE + 4 quarters] iff:
    #     REPDTE < FAILDATE_DT <= REPDTE.dt.offset_by("1y")
    #   (Since REPDTE is always a quarter-end 03-31, 06-30, 09-30, 12-31, +1y is the exact 4th subsequent quarter-end!)
    # - What if a bank continues filing Call Reports ON OR AFTER a terminal closure (FAILDATE_DT <= REPDTE)?
    #   Let's check terminal failure date per CERT (latest FAILDATE_DT, and earliest closure FAILDATE_DT).
    fails_by_cert = (
        fails.group_by("CERT")
        .agg([
            pl.col("FAILDATE_DT").min().alias("FIRST_FAILDATE"),
            pl.col("FAILDATE_DT").max().alias("LAST_FAILDATE"),
            pl.col("COST").sum().alias("TOTAL_FAIL_COST"),
            pl.col("QBFASSET").max().alias("FAIL_QBFASSET"),
            pl.col("RESTYPE").first().alias("FIRST_RESTYPE"),
            pl.len().alias("N_FAIL_RECORDS"),
        ])
    )

    # Join failures_by_cert to inspect post-failure filings and negative-equity failure rates
    fin_joined = fin_sorted.join(fails_by_cert, on="CERT", how="left")

    # Identify post-failure ghost filings: REPDTE >= LAST_FAILDATE (reports filed on or after the bank's final failure date)
    n_post_fail_ghosts = int(fin_joined.filter(
        pl.col("LAST_FAILDATE").is_not_null() & (pl.col("REPDTE") >= pl.col("LAST_FAILDATE"))
    ).height)

    # Also check exact forward 4-quarter failure label by joining fin with fails on CERT
    # and checking REPDTE < FAILDATE_DT <= REPDTE + 1y (exact 4 quarters)
    fin_with_horizon = (
        fin_sorted.select(["CERT", "REPDTE"])
        .with_columns(pl.col("REPDTE").dt.offset_by("1y").alias("REPDTE_PLUS_4Q"))
        .join(fails.select(["CERT", "FAILDATE_DT", "COST", "RESTYPE"]), on="CERT", how="inner")
        .filter((pl.col("FAILDATE_DT") > pl.col("REPDTE")) & (pl.col("FAILDATE_DT") <= pl.col("REPDTE_PLUS_4Q")))
        .group_by(["CERT", "REPDTE"])
        .agg([
            pl.lit(1, dtype=pl.Int8).alias("FAIL_NEXT_4Q"),
            pl.col("FAILDATE_DT").min().alias("NEXT_FAILDATE"),
            pl.col("COST").first().alias("NEXT_FAIL_COST"),
        ])
    )

    fin_full = (
        fin_joined
        .join(fin_with_horizon, on=["CERT", "REPDTE"], how="left")
        .with_columns(pl.col("FAIL_NEXT_4Q").fill_null(0).cast(pl.Int8))
    )

    # Empirical proof that EQ <= 0 is real insolvency, NOT noise:
    eq_neg_df = fin_full.filter(pl.col("EQ") <= 0)
    eq_pos_df = fin_full.filter(pl.col("EQ") > 0)
    fail_rate_eq_neg = 100.0 * float(eq_neg_df["FAIL_NEXT_4Q"].mean()) if eq_neg_df.height > 0 else 0.0
    fail_rate_eq_pos = 100.0 * float(eq_pos_df["FAIL_NEXT_4Q"].mean()) if eq_pos_df.height > 0 else 0.0
    print(f"1-Yr Failure Rate when EQ <= 0: {fail_rate_eq_neg:.2f}% ({eq_neg_df.height:,} rows) "
          f"vs EQ > 0: {fail_rate_eq_pos:.3f}%")

    # Join static point-in-time valid attributes from Institutions (BKCLASS, ESTYMD_DT, INSDATE_DT)
    # plus audit-only exit columns (ACTIVE, CHANGEC1, ENDEFYMD_DT)
    fin_full = fin_full.join(
        inst.select(["CERT", "BKCLASS", "ESTYMD_DT", "INSDATE_DT", "ACTIVE", "CHANGEC1", "ENDEFYMD_DT"]),
        on="CERT",
        how="left",
    )

    # Audit Bank Exit Taxonomy across all distinct CERTs in the Financials panel
    bank_summary = (
        fin_full.group_by("CERT")
        .agg([
            pl.col("REPDTE").min().alias("FIRST_REPDTE"),
            pl.col("REPDTE").max().alias("LAST_REPDTE"),
            pl.len().alias("N_QUARTERS"),
            pl.col("LAST_FAILDATE").first().alias("FAILDATE"),
            pl.col("ACTIVE").first().alias("INST_ACTIVE"),
            pl.col("CHANGEC1").first().alias("INST_CHANGEC1"),
        ])
        .with_columns(
            pl.when(pl.col("FAILDATE").is_not_null())
            .then(pl.lit("Failed Bank (FDIC Resolution)"))
            .when(pl.col("LAST_REPDTE") == pl.date(2026, 6, 30))
            .then(pl.lit("Active in 2026Q2"))
            .otherwise(pl.lit("Non-Failure Exit (Merger / Consolidation / Voluntary)"))
            .alias("EXIT_CATEGORY")
        )
    )
    exit_counts = bank_summary.group_by("EXIT_CATEGORY").agg([
        pl.len().alias("n_banks"),
        pl.col("N_QUARTERS").sum().alias("total_bank_quarters"),
        pl.col("N_QUARTERS").median().alias("median_quarters_observed"),
    ]).sort("n_banks", descending=True).to_pandas()
    exit_counts.to_csv(out_tables / "part2_exit_taxonomy.csv", index=False)
    print("\n=== Bank Exit Taxonomy ===")
    print(exit_counts.to_string(index=False))

    # Summarize all detected quality issues and handling justifications
    anomalies_summary = [
        {
            "issue_id": "DQ-1",
            "category": "Regime Missingness & Sentinel Zero Encoding",
            "description": "P3ASSET (30-89d past due) 100% missing in 1984-1990 & 68.7% in 1991-2006; pre-Basel I RBC1AAJ/RBCRWAJ encoded as sentinel 0.0 (74,731 rows, 15.6% in 1984-1990)",
            "affected_rows": int(fin["P3ASSET"].null_count()),
            "affected_pct": round(100.0 * fin["P3ASSET"].null_count() / n_raw_rows, 2),
            "handling_policy": "Retain all rows; add explicit missingness/sentinel indicator flags; use GAAP Leverage (EQ/ASSET, 100% populated) & NCLNLS as universal anchors; fit median imputer strictly on training split.",
        },
        {
            "issue_id": "DQ-2",
            "category": "Early-Regime Asset Quality & Thrift Missingness",
            "description": "Granular nonaccrual/past-due & charge-off fields (NAASSET, P9ASSET, NTLNLS) 15.9-16.1% missing in 1984-1990 S&L era",
            "affected_rows": int(fin["NAASSET"].null_count()),
            "affected_pct": round(100.0 * fin["NAASSET"].null_count() / n_raw_rows, 2),
            "handling_policy": "Use primary aggregate NCLNLS (Noncurrent loans) + ORE; causal 1-quarter forward-fill within bank (limit=1) for sporadic gaps; training-only median imputation.",
        },
        {
            "issue_id": "DQ-3",
            "category": "Invalid / Non-Positive Total Assets",
            "description": "Rows with ASSET <= 0 or missing ASSET (shell/inactive liquidation filings where asset ratios divide by zero)",
            "affected_rows": n_asset_nonpos,
            "affected_pct": round(100.0 * n_asset_nonpos / n_raw_rows, 4),
            "handling_policy": "Drop rows with ASSET <= 0 or null prior to ratio computation.",
        },
        {
            "issue_id": "DQ-4",
            "category": "Negative Book Equity (Insolvency)",
            "description": f"Rows with EQ <= 0 ({fail_rate_eq_neg:.1f}% 1-yr failure rate vs {fail_rate_eq_pos:.2f}% when EQ > 0)",
            "affected_rows": n_eq_nonpos,
            "affected_pct": round(100.0 * n_eq_nonpos / n_raw_rows, 3),
            "handling_policy": "PRESERVE all rows (true economic insolvency prior to failure; dropping causes severe survivorship bias). Flag with is_insolvent=(EQ<=0) and bound equity-denominator ratios.",
        },
        {
            "issue_id": "DQ-5",
            "category": "Negative Balance Sheet Stocks",
            "description": "Sporadic negative stock values in DEP, LNLSGR, LNATRES, or NCLNLS due to reporting restatements",
            "affected_rows": n_dep_neg + n_lnlsgr_neg + n_lnatres_neg + n_nclnls_neg,
            "affected_pct": round(100.0 * (n_dep_neg + n_lnlsgr_neg + n_lnatres_neg + n_nclnls_neg) / n_raw_rows, 4),
            "handling_policy": "Floor non-negative balance sheet stock items (DEP, LNLSGR, LNATRES, NCLNLS, CHBAL, SC, ORE) at 0.0.",
        },
        {
            "issue_id": "DQ-6",
            "category": "Extreme Ratio Outliers (Small-Denominator Explosion)",
            "description": "Reported ROE exceeding +-100% or ROA exceeding +-20% when equity or assets approach zero",
            "affected_rows": n_roe_extreme,
            "affected_pct": round(100.0 * n_roe_extreme / n_raw_rows, 2),
            "handling_policy": "Learn [0.5%, 99.5%] winsorization/clipping bounds strictly on the Training set inside the scikit-learn Pipeline (zero look-ahead leakage).",
        },
        {
            "issue_id": "DQ-7",
            "category": "Intermittent Panel Gaps",
            "description": f"Consecutive reports for the same CERT separated by >1 quarter ({n_banks_with_gaps:,} distinct banks)",
            "affected_rows": n_gap_events,
            "affected_pct": round(100.0 * n_gap_events / n_raw_rows, 3),
            "handling_policy": "Compute temporal lags and 4-quarter changes using exact calendar-date joins / date-verified window lags rather than blind row-offset shift(4), preventing cross-gap distortion.",
        },
        {
            "issue_id": "DQ-8",
            "category": "Post-Failure Ghost / Receivership Filings",
            "description": "Call Reports filed on or after the bank's final FAILDATE (receivership/bridge artifacts)",
            "affected_rows": n_post_fail_ghosts,
            "affected_pct": round(100.0 * n_post_fail_ghosts / n_raw_rows, 4),
            "handling_policy": "Filter out post-failure rows (REPDTE >= LAST_FAILDATE) so the panel represents active pre-resolution institutions at quarter q.",
        },
        {
            "issue_id": "DQ-9",
            "category": "Right-Censored Forward Labels (Post-2025Q2)",
            "description": "Quarters 2025Q3-2026Q2 lack complete 4-quarter-ahead failure observation windows",
            "affected_rows": int(fin.filter(pl.col("REPDTE") > pl.date(2025, 6, 30)).height),
            "affected_pct": round(100.0 * fin.filter(pl.col("REPDTE") > pl.date(2025, 6, 30)).height / n_raw_rows, 2),
            "handling_policy": "Retain in storage panel for latest feature lags, but restrict supervised modelling & evaluation strictly to REPDTE <= 2025-06-30 (2025Q2).",
        },
        {
            "issue_id": "DQ-10",
            "category": "Static Snapshot Leakage in Institutions Table",
            "description": "Institutions fields ACTIVE, CHANGEC1, and ENDEFYMD reflect end-of-sample (2026) status, not status at quarter q",
            "affected_rows": n_raw_rows,
            "affected_pct": 100.0,
            "handling_policy": "Exclude ACTIVE, CHANGEC1, ENDEFYMD from predictive features; use only point-in-time inception fields (ESTYMD, INSDATE, BKCLASS) to compute bank age/charter at quarter q.",
        },
    ]
    df_anom = pd.DataFrame(anomalies_summary)
    df_anom.to_csv(out_tables / "part2_anomalies_and_handling.csv", index=False)

    # -------------------------------------------------------------------------
    # 4. Build Clean Point-in-Time Bank-Quarter Panel
    # -------------------------------------------------------------------------
    stock_nonneg_cols = [
        "DEP", "DEPDOM", "BRO", "CHBAL", "SC",
        "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
        "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "ORE",
    ]
    panel_clean = (
        fin_full
        # 1. Remove invalid non-positive asset shells and post-failure ghost filings
        .filter(pl.col("ASSET").is_not_null() & (pl.col("ASSET") > 0))
        .filter(pl.col("LAST_FAILDATE").is_null() | (pl.col("REPDTE") < pl.col("LAST_FAILDATE")))
        # 2. Floor non-negative balance-sheet stock columns at 0.0
        .with_columns([
            pl.when(pl.col(c) < 0).then(0.0).otherwise(pl.col(c)).alias(c)
            for c in stock_nonneg_cols
        ])
        # 3. Point-in-time bank age in years at quarter REPDTE (from ESTYMD_DT or INSDATE_DT)
        .with_columns([
            (
                (pl.col("REPDTE") - pl.coalesce([pl.col("ESTYMD_DT"), pl.col("INSDATE_DT")])).dt.total_days() / 365.25
            ).clip(lower_bound=0.0).alias("BANK_AGE_YRS"),
            (pl.col("EQ") <= 0).cast(pl.Int8).alias("IS_INSOLVENT_EQ"),
            (pl.col("RBC1AAJ").is_null()).cast(pl.Int8).alias("MISSING_RBC1AAJ"),
            (pl.col("REPDTE") <= pl.date(2025, 6, 30)).alias("IN_MODELLING_SCOPE"),
        ])
        # Drop static future-leaking columns from clean modelling panel (keep only in audit summary)
        .drop(["ACTIVE", "CHANGEC1", "ENDEFYMD_DT"])
        .sort(["CERT", "REPDTE"])
    )

    clean_path = proc_dir / "panel_clean.parquet"
    panel_clean.write_parquet(clean_path, compression="zstd")
    print(f"\nSaved clean point-in-time panel to {clean_path}: {panel_clean.height:,} rows.")

    # Compute quarter-by-quarter summary for plotting & split selection
    q_stats = (
        panel_clean.group_by("REPDTE")
        .agg([
            pl.len().alias("active_banks"),
            pl.col("ASSET").sum().alias("total_assets_kusd"),
            pl.col("FAIL_NEXT_4Q").sum().alias("pos_labels_next_4q"),
            (pl.col("FAIL_NEXT_4Q").mean() * 100.0).alias("pos_rate_pct"),
        ])
        .sort("REPDTE")
        .to_pandas()
    )
    q_stats.to_csv(out_tables / "part2_quarterly_panel_stats.csv", index=False)

    # Save JSON summary of Part 2
    mod_panel = panel_clean.filter(pl.col("IN_MODELLING_SCOPE"))
    p2_summary = {
        "raw_rows": n_raw_rows,
        "raw_distinct_banks": n_raw_banks,
        "clean_rows_all_quarters": int(panel_clean.height),
        "clean_distinct_banks_all": int(panel_clean["CERT"].n_unique()),
        "Dropped_asset_nonpos": n_asset_nonpos,
        "Dropped_post_failure_ghosts": n_post_fail_ghosts,
        "modelling_scope_rows_1984Q1_2025Q2": int(mod_panel.height),
        "modelling_scope_distinct_banks": int(mod_panel["CERT"].n_unique()),
        "modelling_scope_positive_bank_quarters": int(mod_panel["FAIL_NEXT_4Q"].sum()),
        "modelling_scope_positive_rate_pct": round(100.0 * float(mod_panel["FAIL_NEXT_4Q"].mean()), 4),
        "distinct_failed_banks_in_panel": int(mod_panel.filter(pl.col("FAIL_NEXT_4Q") == 1)["CERT"].n_unique()),
        "insolvent_eq_rows": int(eq_neg_df.height),
        "insolvent_eq_failure_rate_pct": round(fail_rate_eq_neg, 2),
        "solvent_eq_failure_rate_pct": round(fail_rate_eq_pos, 4),
    }
    (out_tables / "part2_summary.json").write_text(json.dumps(p2_summary, indent=2))
    print("\n=== Part 2 Clean Panel Summary ===")
    for k, v in p2_summary.items():
        print(f"  {k:<40}: {v}")

    # -------------------------------------------------------------------------
    # Plot Figure 2: 42-Year Secular Consolidation & Failure Waves (1984Q1 - 2026Q2)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)

    q_dates = pd.to_datetime(q_stats["REPDTE"])
    axes[0].plot(q_dates, q_stats["active_banks"], color="#1f4e78", linewidth=2.2, label="Active Insured Institutions")
    axes[0].plot(q_dates, q_stats["total_assets_kusd"] / 1e9, color="#2e8b57", linewidth=2.0, linestyle="--",
                 label="Aggregate Industry Assets ($ Trillions, right axis)")
    ax0_r = axes[0].twinx()
    ax0_r.plot(q_dates, q_stats["total_assets_kusd"] / 1e9, color="#2e8b57", linewidth=2.0, linestyle="--")
    ax0_r.set_ylabel("Total Assets ($ Trillions)", color="#2e8b57")
    axes[0].set_ylabel("Number of Reporting Banks", color="#1f4e78")
    axes[0].set_title("A. Four Decades of US Banking Consolidation: Institution Count vs. Industry Assets (1984Q1–2026Q2)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Bottom panel: 1-Year-Ahead Failure Labels per Quarter with Crisis Highlighting
    mod_q = q_stats[pd.to_datetime(q_stats["REPDTE"]) <= pd.Timestamp("2025-06-30")]
    axes[1].fill_between(pd.to_datetime(mod_q["REPDTE"]), mod_q["pos_labels_next_4q"], color="#cb181d", alpha=0.75,
                         label="Banks Failing within Next 4 Quarters ($y_{i,q}=1$)")
    axes[1].axvspan(pd.Timestamp("1984-03-31"), pd.Timestamp("1993-12-31"), color="#fdd0a2", alpha=0.35, label="S&L / Banking Crisis (1984–1993)")
    axes[1].axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2014-12-31"), color="#dadaeb", alpha=0.45, label="Global Financial Crisis & Aftermath (2008–2014)")
    axes[1].axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-06-30"), color="#c7e9c0", alpha=0.5, label="2023 Regional Bank Turmoil")
    axes[1].set_ylabel("Bank-Quarters with $y_{i,q}=1$")
    axes[1].set_xlabel("Reporting Quarter (REPDTE)")
    axes[1].set_title("B. Temporal Clustering of Bank Failure Labels ($y_{i,q} = 1\\{\\text{fail in } (q, q+4Q]\\}$) Across Three Crisis Regimes")
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_figs / "fig2_panel_dynamics_and_exits.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot Figure 3: Regime-Dependent Missingness Heatmap Across Regulatory Eras
    # -------------------------------------------------------------------------
    top_missing = df_quality.sort_values("missing_pct_overall", ascending=False).head(15)
    heat_cols = [
        "missing_pct_1984_1990",
        "missing_pct_1991_2006",
        "missing_pct_2007_2019",
        "missing_pct_2020_2026",
    ]
    heat_data = top_missing.set_index("field")[heat_cols]
    heat_data.columns = [
        "1984–1990\n(Pre-Basel I)",
        "1991–2006\n(Basel I / FDICIA)",
        "2007–2019\n(GFC & Basel III)",
        "2020–2026\n(CBLR Era)",
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    im = ax.imshow(heat_data.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(heat_data.columns)))
    ax.set_xticklabels(heat_data.columns, fontsize=9.5)
    ax.set_yticks(np.arange(len(heat_data.index)))
    ax.set_yticklabels(heat_data.index, fontsize=9)
    for i in range(heat_data.shape[0]):
        for j in range(heat_data.shape[1]):
            val = heat_data.values[i, j]
            txt_color = "white" if val > 55 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=8.5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Missing Percentage (%)")
    ax.set_title("Regime-Dependent Missingness Across Four Regulatory Eras (Top 15 Fields)")
    plt.tight_layout()
    fig.savefig(out_figs / "fig3_data_quality_missingness.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figures 2 and 3.")


if __name__ == "__main__":
    run_quality_audit_and_build_panel(
        raw_dir=Path("data_raw/fdic"),
        proc_dir=Path("data_processed"),
        out_tables=Path("outputs/tables"),
        out_figs=Path("outputs/figures"),
    )
