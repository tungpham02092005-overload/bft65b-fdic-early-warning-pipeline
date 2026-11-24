# A Big-Data Pipeline for Bank Failure Early Warning
**Course:** BFT65B — Big Data Analytics in Finance (National Economics University)  
**Assessment:** Final Examination — Individual Research Report  
**Author / Student Name:** Pham Bach Tung  
**Student ID:** 11231023  
**GitHub Repository:** `https://github.com/tungpham02092005-overload/bft65b-fdic-early-warning-pipeline`  

---

## 1. Executive Summary & Project Scope

This repository implements an end-to-end, reproducible, and strictly point-in-time **Big-Data Engineering & Early-Warning System (EWS) Pipeline** for banking supervision using four decades of US Federal Deposit Insurance Corporation (FDIC) Call Reports (`1984Q1–2026Q2`). Designed from the perspective of a central bank supervision data engineering team (State Bank of Vietnam — SBV context), the pipeline addresses three foundational research questions:

- **Q1 (Storage & Scalable Compute):** How should four decades of quarterly bank reports ($1,682,616$ rows across $24,064$ insured institutions) be stored and processed on a single laptop, and at what exact data volume or granularity does a distributed Apache Spark cluster become justified?
- **Q2 (Data Quality & Point-in-Time Panel Integrity):** What structural quality anomalies exist in raw regulatory filings (regime-dependent missingness across Basel I / FDICIA / CBLR eras, cross-quarter schema drift, negative book equity insolvency vs. invalid zero-asset shells, intermittent reporting gaps, post-failure receivership ghost filings, and non-failure merger exits), and how are they resolved without look-ahead bias?
- **Q3 (Leak-Free Validation & Cost-Sensitive Early Warning):** Can the pipeline's output support an honestly validated 1-year-ahead early-warning model ($y_{i,q} = \mathbf{1}\{\text{bank } i \text{ fails in } (q, q+4\text{ quarters}]\}$) under extreme class imbalance and temporal regime shifts?

### Key Empirical Summary (for First-Page Verification)

| Parameter | Empirical Value |
|---|---|
| **Data Source & Method** | FDIC BankFind Suite Public REST API (`https://api.fdic.gov/banks`), Resumable Quarter-Partitioned Downloader (`src/part1_download.py` & `fdic_download_starter.py`) |
| **Download Date** | `2026-11-11` (`2026-11-11 13:08:34 UTC`) |
| **`Financials` Raw Scope & Rows** | `1984Q1` (`1984-03-31`) to `2026Q2` (`2026-06-30`) — **170 quarters**, **1,682,616 rows**, **24,064 distinct banks (`CERT`)**, **0 duplicate `(CERT, REPDTE)` pairs** |
| **`Failures` Table Rows** | **4,117 total historical failures** (`3,629` with modern `CERT`; `3,241` failed banks matched to the 1984Q1–2026Q2 panel) |
| **`Institutions` Table Rows** | **27,835 unique institutions** (`4,306` active in 2026Q2; `16,517` non-failure merger/consolidation exits; `3,241` FDIC failure resolutions in panel) |
| **Clean Point-in-Time Panel** | **1,677,817 rows** (`1984Q1–2026Q2`, after removing `4,799` post-failure receivership ghost filings); **1,660,316 rows** in supervised modelling scope (`1984Q1–2025Q2`) |
| **Development Period** | **`1984Q1 – 2009Q4`** (`104` quarters): **1,292,456 bank-quarters**, **23,919 distinct banks**, **11,350 positive failure labels ($y_{i,q}=1$, 0.8782%)**, **2,991 distinct failed banks** |
| **Embargo / Purge Gap** | **`2010Q1 – 2010Q4`** (**4 full quarters** purged to prevent forward-label overlap): **31,477 bank-quarters** (`457` positive labels across `208` failing banks) |
| **Held-Out Test Period** | **`2011Q1 – 2025Q2`** (`58` quarters, evaluated **once**): **336,383 bank-quarters**, **7,743 distinct banks**, **652 positive failure labels ($y_{i,q}=1$, 0.1938%)**, **199 distinct failed banks** ($\ge 50$ required) |
| **Right-Censored Post-2025Q2** | **`2025Q3 – 2026Q2`** (`4` quarters, **17,501 rows**): retained in storage for trailing lags; excluded from supervised modelling due to incomplete 4-quarter forward windows |

---

## 2. Repository Structure

```text
.
├── README.md                          # Complete documentation & end-to-end reproduction guide
├── requirements.txt                   # Pinned Python 3.11 dependencies
├── .gitignore                         # Excludes data_raw/, data_processed/, *.parquet, .venv/
├── fdic_download_starter.py           # Course starter script for FDIC API acquisition
├── run_pipeline.py                    # Single-command end-to-end pipeline orchestrator (Part 1 -> Part 5)
├── part1_download.py                  # Part 1A: Resumable parallel FDIC API downloader & verifier
├── part1_storage_benchmark.py         # Part 1B: Storage format, codec, schema & partitioning benchmark
├── part2_quality_panel.py             # Part 2: Automated data quality audit & point-in-time panel builder
├── part3_scalable_processing.py       # Part 3: Polars Lazy/Eager vs Pandas vs PySpark + scaling curve
├── part4_feature_pipeline.py          # Part 4: 39 CAMELS + temporal features & training-only pipeline
├── part5_modelling.py                 # Part 5: Purged temporal CV, cost threshold, single test & Spark MLlib
├── src/                               # Modular package copy of Part 1 -> Part 5 pipeline scripts
│   ├── part1_download.py
│   ├── part1_storage_benchmark.py
│   ├── part2_quality_panel.py
│   ├── part3_scalable_processing.py
│   ├── part4_feature_pipeline.py
│   └── part5_modelling.py
└── outputs/
    ├── figures/                       # Publication-ready 300-DPI figures (fig1..fig6.png)
    └── tables/                        # Empirical CSV/JSON benchmark & audit tables + query plans
```

> **Important Submission Note:** Per course rules, raw and processed data files (`data_raw/` and `data_processed/`) are excluded via `.gitignore` and are **not** committed to Git. All summary tables, query plans, and figures in `outputs/` are deterministically regenerated by running `python run_pipeline.py`.

---

## 3. End-to-End Reproducibility Guide

### Prerequisites
- **Python:** `3.11+`
- **Java Runtime (for local PySpark & Spark MLlib demo):** OpenJDK `17` (64-bit). If `JAVA_HOME` is not set, `src/part3_scalable_processing.py` automatically searches standard Windows/Linux JDK paths.
- **Hardware:** Any modern laptop with $\ge 8\text{ GB}$ RAM (tested on Windows 11 x64). No paid cloud or cluster is required.

### Step 1: Clone & Set Up Environment

```bash
git clone <YOUR_PUBLIC_GITHUB_REPO_URL>
cd bft65b-fdic-early-warning-pipeline

python -m venv .venv
# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1
# Or on Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Run the Entire Pipeline End-to-End (Single Command)

```bash
python run_pipeline.py
```

If you have already downloaded `data_raw/fdic` (either via `src/part1_download.py` or `fdic_download_starter.py`), you can skip the network check or rely on the automatic manifest resumability:

```bash
python run_pipeline.py --skip-download
```

### Step 3: Or Run Individual Stages Step-by-Step

```bash
# Part 1A: Download all 170 quarters (1984Q1-2026Q2) + failures + institutions & verify completeness
python src/part1_download.py --workers 6 --pause 0.15
# (Optional: verify compatibility with the course starter script)
python fdic_download_starter.py

# Part 1B: Benchmark CSV vs Gzip CSV vs Parquet (Uncompressed, LZ4, Snappy, ZSTD, GZIP, Float32/64, Partitioned)
python src/part1_storage_benchmark.py

# Part 2: Run automated data quality audit, exit taxonomy, and build clean point-in-time panel
python src/part2_quality_panel.py

# Part 3: Benchmark Polars Lazy/Eager vs Pandas vs PySpark, extract query plans, and run scaling curve (up to 8.4M rows)
python src/part3_scalable_processing.py

# Part 4: Build 39 CAMELS + temporal features, fit training-only preprocessing pipeline, and verify leakage gate
python src/part4_feature_pipeline.py

# Part 5: Run Purged Expanding-Window Temporal CV, estimate data-driven cost threshold, evaluate once on Test set, and run Spark MLlib demo
python src/part5_modelling.py
```

---

## 4. Academic Integrity, Code Attribution, AI Tool Disclosure & References

In strict compliance with **Section 4.3 (Academic Integrity)** and **Section 5 (Submission Rules)** of the BFT65B Final Examination guidelines at National Economics University (NEU), I hereby declare that every external resource—including starter scripts, open-source software libraries, regulatory data dictionaries, academic papers, and artificial intelligence coding assistants—used in the preparation of this pipeline and report is explicitly cited below. All data downloads (`1,682,616` Financials rows, `4,117` Failures rows, `27,835` Institutions rows), data-quality audits, storage and engine benchmarks, feature pipelines, and model evaluations were executed deterministically (`SEED = 42`) on the authentic FDIC BankFind dataset with zero data fabrication and zero look-ahead leakage.

### 4.1 Code, Starter Script & Open-Source Software Attribution

| Component / Script / Library | Source / Authors & Version | Role & Adaptation in This Repository |
|---|---|---|
| **Course Starter Script (`fdic_download_starter.py`)** | BFT65B Course Instructor, National Economics University (2026) | Retained verbatim as reference; its REST query structure and 33-field Call Report list were extended in `src/part1_download.py` with multi-threaded resumability, exponential backoff, manifest checksums, and strict PyArrow schema casting. |
| **FDIC BankFind Suite Public REST API** | Federal Deposit Insurance Corporation (`https://api.fdic.gov/banks`) | Primary data provider for `/financials` (170 quarters, 1984Q1–2026Q2), `/failures` (1934–2026), and `/institutions` endpoints plus RISVIEW field definitions. |
| **Polars DataFrame Engine & PyArrow** | Vink et al. (2024), `polars==1.2.1`; Apache Arrow `pyarrow==14.0.2` | Core columnar storage and out-of-core Lazy/Eager query engine in `src/part1_storage_benchmark.py`, `src/part2_quality_panel.py`, `src/part3_scalable_processing.py`, and `src/part4_feature_pipeline.py`. |
| **Apache Spark (PySpark SQL & MLlib)** | Zaharia et al. (2016); Armbrust et al. (2015), `pyspark==3.5.3` (OpenJDK 17) | Catalyst/AQE distributed query execution and numerical equivalence verification in `src/part3_scalable_processing.py`; distributed `Imputer -> VectorAssembler -> StandardScaler -> LR/RF` MLlib pipeline in `src/part5_modelling.py`. |
| **pandas, NumPy & SciPy** | McKinney (2010), `pandas==2.2.3`; Harris et al. (2020), `numpy==1.26.4`; Virtanen et al. (2020), `scipy==1.12.0` | Vectorized single-threaded baseline in Part 3 benchmarks; nonparametric Mann–Whitney $U$ and two-sample Kolmogorov–Smirnov (KS) feature discrimination tests in Part 4. |
| **scikit-learn & LightGBM** | Pedregosa et al. (2011), `scikit-learn==1.4.2`; Ke et al. (2017), `lightgbm==4.3.0` | Training-only preprocessing `Pipeline` (custom `TrainingQuantileClipper`, `SimpleImputer`, `StandardScaler`), `LogisticRegression`, `RandomForestClassifier`, and `LGBMClassifier` in Parts 4–5. |
| **Matplotlib, ReportLab & python-docx** | Hunter (2007), `matplotlib==3.9.0`; `reportlab==4.4.10`; `python-docx==1.2.0` | Automated generation of 300-DPI figures (`outputs/figures/fig1..fig6.png`) and programmatic compilation of PDF and DOCX reports (`src/generate_report.py`). |

### 4.2 Transparent Module-by-Module AI Tool Usage Disclosure

| Pipeline Stage & Script | AI Tool Used | Scope of AI Assistance | Author Verification & Empirical Audit |
|---|---|---|---|
| **Part 1: Acquisition & Storage (`src/part1_download.py`, `src/part1_storage_benchmark.py`)** | Google DeepMind Antigravity AI Assistant | Assisted in writing `ThreadPoolExecutor` retry boilerplate and formatting the 11-layout Parquet/CSV benchmark loop. | Verified `1,682,616` downloaded rows == live API total across all 170 quarters; discovered & fixed cross-quarter `Int64`/`Float64` schema drift. |
| **Part 2: Data Quality & Panel (`src/part2_quality_panel.py`)** | Google DeepMind Antigravity AI Assistant | Assisted in structuring Polars aggregation queries for missingness by regulatory era and exit taxonomy tables. | Verified pre-Basel I sentinel `0.0` encoding (`74,731` rows), `58.7x` higher failure risk of `EQ<=0` rows, and removal of `4,799` post-failure ghost rows. |
| **Part 3: Scalable Processing (`src/part3_scalable_processing.py`)** | Google DeepMind Antigravity AI Assistant | Assisted in translating Polars `.over('CERT')` window expressions into PySpark Window SQL and configuring local OpenJDK 17 on Windows. | Verified numerical equivalence across Polars, pandas, and PySpark (`< 1.3e-14` max relative error) and inspected Catalyst/AQE physical plans. |
| **Part 4 & Part 5: Feature Pipeline & Modelling (`src/part4_feature_pipeline.py`, `src/part5_modelling.py`)** | Google DeepMind Antigravity AI Assistant | Assisted in implementing the custom scikit-learn `TrainingQuantileClipper`, Purged Expanding-Window CV loop, and Spark MLlib pipeline. | Executed automated zero-leakage gate (`part4_leakage_audit.json`), verified 4-quarter purge gap (2010), and audited single Test evaluation. |
| **Part 6 & Report Compilation (`src/generate_report.py`, `README.md`)** | Google DeepMind Antigravity AI Assistant | Assisted in English academic editing and ReportLab / `python-docx` table layout code. | Cross-checked every reported number in PDF, DOCX, and `README.md` against generated CSV/JSON artifacts in `outputs/tables/`. |

### 4.3 Complete Bibliography & References (Data, Literature, Regulations & Software)

#### A. Regulatory Frameworks & Primary Data Documentation
1. **Federal Deposit Insurance Corporation (FDIC).** (2026). *BankFind Suite: Public REST API & Research Information System (RISVIEW) Data Dictionary*. `https://api.fdic.gov/banks/docs/`.
2. **Federal Financial Institutions Examination Council (FFIEC).** (1996). *Uniform Financial Institutions Rating System (UFIRS / CAMELS)*, Federal Register Vol. 61, No. 245, pp. 67021–67029.
3. **FDIC.** (1997). *History of the Eighties — Lessons for the Future: Volume I, An Examination of the Banking Crises of the 1980s and Early 1990s*. Washington, DC: FDIC Division of Research and Statistics.
4. **FDIC.** (2017). *Crisis and Response: An FDIC History, 2008–2013*. Washington, DC: FDIC.
5. **Board of Governors of the Federal Reserve System.** (2023). *Review of the Federal Reserve's Supervision and Regulation of Silicon Valley Bank*. Washington, DC.
6. **State Bank of Vietnam (SBV).** (2018). *Circular No. 52/2018/TT-NHNN stipulating credit institution and foreign bank branch rating (CAMELS framework in Vietnam)*, amended by Circular 24/2023/TT-NHNN.
7. **National Assembly of Vietnam.** (2024). *Law on Credit Institutions No. 32/2024/QH15* (Provisions on early intervention, special control, and mandatory transfer of weak banks).

#### B. Academic Literature on Bank Failure Early Warning & Financial Machine Learning
8. **Cole, R. A., & White, L. J.** (2012). "Déjà Vu All Over Again: The Causes of U.S. Commercial Bank Failures This Time Around." *Journal of Financial Services Research*, 42(1–2), 5–29.
9. **Wheelock, D. C., & Wilson, P. W.** (2000). "Why Do Banks Disappear? The Determinants of U.S. Bank Failures and Acquisitions." *The Review of Economics and Statistics*, 82(1), 127–138.
10. **Lane, W. R., Looney, S. W., & Wansley, J. W.** (1986). "An Application of the Cox Proportional Hazards Model to Bank Failure." *Journal of Banking & Finance*, 10(4), 511–531.
11. **Martin, D.** (1977). "Early Warning of Bank Failure: A Logit Regression Approach." *Journal of Banking & Finance*, 1(3), 249–276.
12. **Estrella, A., Park, S., & Peristiani, S.** (2000). "Capital Ratios as Predictors of Bank Failure." *FRBNY Economic Policy Review*, 6(2), 33–52.
13. **Barr, R. S., Seiford, L. M., & Siems, T. F.** (1994). "Forecasting Bank Failure: A Non-Parametric Frontier Estimation Approach." *Recherches Économiques de Louvain*, 60(4), 417–429.
14. **López de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley. (Chapter 7: Cross-Validation in Finance — Purging and Embargoing overlapping forward labels).
15. **Elkan, C.** (2001). "The Foundations of Cost-Sensitive Learning." *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI)*, 973–978.
16. **Saito, T., & Rehmsmeier, M.** (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLOS ONE*, 10(3), e0118432.

#### C. Starter Code & Big-Data / Machine-Learning Software Citations
17. **BFT65B Course Instructor.** (2026). `fdic_download_starter.py` — Starter script for FDIC BankFind API acquisition, National Economics University.
18. **Vink, R., et al.** (2024). *Polars: Blazingly Fast DataFrames in Rust and Apache Arrow* (v1.2.1). `https://pola.rs/`.
19. **Zaharia, M., et al.** (2016). "Apache Spark: A Unified Engine for Big Data Processing." *Communications of the ACM*, 59(11), 56–65; **Armbrust, M., et al.** (2015). "Spark SQL: Relational Data Processing in Spark." *ACM SIGMOD*, 1383–1394.
20. **McKinney, W.** (2010). "Data Structures for Statistical Computing in Python." *Proceedings of the 9th Python in Science Conference (SciPy)*, 56–61.
21. **Pedregosa, F., et al.** (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830.
22. **Ke, G., et al.** (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 3146–3154.
