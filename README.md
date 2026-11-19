# Explainable Machine Learning for Phishing URL Detection

A reproducible, **offline research prototype** for classifying phishing URLs using lexical and structural features, comparing Logistic Regression and Random Forest, interpreting predictions with SHAP, and measuring generalization to an independently collected dataset.

**Course project:** CS 6399 · Angelo State University · Fall 2026  
**Repository:** https://github.com/SawMageOnGog/phishing-url-detection

> **Research limitation:** This project is not a browser extension, deployed filtering service, or security guarantee. It analyzes URL strings without visiting the websites. Results on one dataset do not imply reliable real-world phishing protection.

## Research questions

1. How accurately can models classify phishing and legitimate URLs using URL-only features?
2. What precision/recall tradeoffs and feature contributions are observed?
3. How well do selected models transfer to a separate URL collection, particularly for URLs without domain overlap?

## Method

- Input: labeled URL strings from **URL-Phish version 2** (primary training/testing source).
- Cleaning: validate URLs, normalize source records, remove invalid/duplicate observations, and audit input counts.
- Representation: **22 lexical/structural URL features** (see [`FEATURES.md`](FEATURES.md)). No webpage fetching or DNS lookups.
- Split: a repeatable, **registrable-domain-separated** training/holdout split.
- Models: Logistic Regression and Random Forest from scikit-learn.
- Selection: three-fold, training-only grouped cross-validation; choose by mean average precision and evaluate selected configurations on the unchanged holdout.
- Interpretability: SHAP explanations for selected saved models on a deliberately balanced sample of 48 test cases.
- Transfer: evaluate the *frozen* selected models on **PhiUSIIL**, with opposite dataset label definitions remapped and URL/domain overlap audited.
- Follow-up: analyze false-positive/false-negative outcomes and feature-distribution differences across datasets.

## Results (saved research runs through November 18, 2026)

| Metric | Logistic Regression — primary holdout | Random Forest — primary holdout | Random Forest — PhiUSIIL domain-disjoint |
|---|---:|---:|---:|
| Accuracy | 88.46% | 94.67% | 75.12% |
| Phishing recall | 91.47% | 89.45% | 32.56% |
| Precision | 56.02% | 77.13% | 85.77% |
| False-positive rate | 12.05% | 4.45% | 2.80% |

The **primary test set had 23,015 records**, and the **external domain-disjoint subset had 190,782 records**. Random Forest missed **349 phishing cases** on the original holdout and **43,949** in the much larger external subset. These counts must not be compared without accounting for sample sizes and class balance. Among phishing-labeled URLs, median URL length differed (60 characters primary vs. 29 external) and median URL entropy differed (approximately 4.50 vs. 3.95). These are descriptive differences, **not proof of causality**.

The original and external datasets differ in collection and distribution; the relatively low external recall is a central project finding, not a deployment-ready result. SHAP attributions describe model behavior on the selected cases rather than causal feature effects or population-wide importance.

## Project structure

```text
PhishingURLProject/
├── run.py                # Baseline preparation, training, evaluation, evidence export
├── validate.py           # Grouped cross-validation and selected model exports
├── explain.py            # SHAP global and local explanations
├── generalize.py         # Independent PhiUSIIL evaluation and overlap auditing
├── analyze_errors.py     # Error counts, examples, feature-distribution analysis
├── config.json           # Original baseline configuration and dataset metadata
├── requirements.txt      # Project dependencies
├── src/
│   ├── data.py           # Dataset cleaning and domain-separated splitting
│   ├── features.py       # URL feature extraction
│   ├── models.py         # Baseline model definitions
│   ├── evaluate.py       # Metrics, predictions, models and charts
│   └── reporting.py      # Research summaries
├── tests/                # Unit and integration tests
├── FEATURES.md           # Detailed feature definitions
├── VALIDATION.md         # Validation notes
├── REPORT4_SHAP.md       # SHAP notes
└── REPORT4_GENERALIZATION.md
```

Raw data and timestamped experiment outputs are **not committed** to this repository. They are created or downloaded separately under `data/raw/` and `outputs/`.

## Installation (macOS / Python)

The primary development machine successfully ran the test suite on **Python 3.14.5**, **scikit-learn 1.9.0**, **SHAP 0.51.0**, **NumPy 2.4.4**, and **pandas 3.0.3**. Use an isolated virtual environment. The supplied `requirements.txt` must be aligned with the verified environment to reproduce the saved model loading without warnings; review its pins before installing.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

**Last confirmed on the project owner's Mac:** 18 tests passed, 0 failures (November 18, 2026). The test suite uses small synthetic fixtures; count warnings for those fixtures do not indicate test failures. Passing unit tests alone does not validate fresh end-to-end reproduction of every saved experiment.

## Obtain the datasets

1. Download **URL-Phish v2** directly from [Mendeley Data](https://data.mendeley.com/datasets/65z9twcx3r/2), DOI [`10.17632/65z9twcx3r.2`](https://doi.org/10.17632/65z9twcx3r.2). Place the original dataset CSV file(s) in `data/raw/`.
2. For the external evaluation, download [PhiUSIIL phishing URL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) from UCI and provide its CSV path to `generalize.py`.

**Important:** `run.py` discovers all CSV files recursively under `data/raw/` unless given `--csv`. If both datasets are present there, **always explicitly specify the URL-Phish CSV file(s)** with `--csv`, so the baseline does not accidentally ingest the external PhiUSIIL dataset. Keep downloaded data outside Git.

## Run the research pipeline

Execute commands from the repository root in the activated environment. Each command uses the actual output directory produced by the preceding step. The timestamps below are placeholders: replace them with your own directory names.

**1. Baseline**

```bash
python run.py --csv data/raw/YOUR_URL_PHISH_V2.csv
```

This creates `outputs/<baseline-run>/` containing preparation artifacts, trained models, metrics and a summary. `--prepare-only` runs the preparation phase without training models.

**2. Cross-validation and model selection**

```bash
python validate.py --baseline outputs/<baseline-run> --folds 3 --model both
```

Use the actual validation output directory created by this command for subsequent steps.

**3. SHAP explanations**

```bash
python explain.py --baseline outputs/<baseline-run> --validation outputs/<validation-run> --samples 48 --background 40 --model both
```

**4. External transfer evaluation**

```bash
python generalize.py --dataset /path/to/PhiUSIIL_Phishing_URL_Dataset.csv --baseline outputs/<baseline-run> --validation outputs/<validation-run>
```

PhiUSIIL maps `0 = phishing, 1 = legitimate`, whereas this project's convention is `1 = phishing, 0 = legitimate`. `generalize.py` performs the necessary conversion. Do **not** retrain the selected models for this evaluation.

**5. Error and dataset-shift analysis**

```bash
python analyze_errors.py --baseline outputs/<baseline-run> --validation outputs/<validation-run> --external outputs/<external-run> --output outputs/<new-error-analysis-run>
```

The output directory must be new or empty. This script expects the primary and external evaluation counts used for the reported experiment (23,015 and 190,782); it will stop if they differ. Outputs include outcome tables, numerical-only error examples, distribution summaries, plots, and a manifest.

## Test suite

```bash
python -m unittest discover -s tests -v
```

Tests cover cleaning, canonicalization, feature calculations, domain grouping/splits, model evaluation, candidate configuration, external label mapping, and failure-analysis accounting. The latest verified run contains **18 passing tests**.

## Interpretation and reproducibility notes

- The decision threshold in these runs is **0.5**, not a calibrated real-world risk threshold.
- Data overlap is audited at the exact-URL and registrable-domain levels; the domain-disjoint subset is the more stringent transfer test.
- For the saved research outputs, no new models were fitted during SHAP, external testing, or error analysis.
- The project records manifests, configuration, software environment information, and per-run evidence under `outputs/`. These outputs are local and excluded from Git.
- Model artifacts are version-sensitive. Align the installed scikit-learn version with the saved artifacts before loading them.
- The code currently provides a **command-line interface**, not a graphical user interface.

## References and dataset attribution

- Dam Minh, L., & Tran Cong, H. (2026). *URL-Phish: A feature-engineered dataset for phishing detection (Version 2)*. Mendeley Data. https://doi.org/10.17632/65z9twcx3r.2
- Linh, D. M., & Hung, T. C. (2025). A feature-engineered dataset of benign and phishing URLs for machine learning and large language models evaluation. *Data in Brief, 63*, 112162. https://doi.org/10.1016/j.dib.2025.112162
- Prasad, A., & Chandra, S. (2024). *PhiUSIIL phishing URL (website)*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

See the course progress reports for the complete related-work discussion and remaining sources.
