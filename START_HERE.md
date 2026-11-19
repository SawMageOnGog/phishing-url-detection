# Phishing URL Detection Project

This project produces actual preparation and baseline evidence for Progress Report 3.
It implements the approved Python-script workflow: URL-Phish version 2, locally
extracted URL features, Logistic Regression and Random Forest, and a fixed,
domain-separated test set. SHAP and PhiUSIIL evaluation remain later work.

## 1. Set up Python on your Mac

Extract this project folder somewhere convenient. Open Terminal, type `cd `,
drag the extracted PhishingURLProject folder into Terminal, and press Return.
Then run:

```bash
python3 --version
```

Use Python 3.11, 3.12, or 3.13 with these pinned dependencies. If your version is
older, or is 3.14+, send the version before proceeding so we can use a compatible
environment without changing your existing projects.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The virtual environment keeps this project's packages separate. To resume in a
new Terminal window, return to the project folder and run the activation command.
If installation or a test fails, stop and send the error; do not continue as if
the checks passed.

## 2. Download the primary dataset

Open the original publisher page:

https://data.mendeley.com/datasets/65z9twcx3r/2

Confirm **Version 2**, DOI **10.17632/65z9twcx3r.2**. Use Download All or the file
download controls. Extract the downloaded archive if needed. Put the original
CSV file(s) under `data/raw/`. Do not copy explanatory example CSVs, generated
outputs, or another dataset into that folder. If the archive contains training,
validation, and test partitions, their raw URL/label rows can be combined here
because this project deliberately rebuilds its own domain-separated split.
Preserve the downloaded archive separately as a source record.

The publisher describes 116,600 URLs: 100,000 legitimate (0) and 16,600 phishing
(1). The script compares actual input counts with that description and warns
about differences; the publication's numbers are never substituted for measured
counts. Dataset availability and contents must be confirmed on your machine.

This bundle does not redistribute the source dataset. The source is CC BY 4.0;
retain attribution in your project and report.

## 3. Run the experiment

```bash
python run.py
```

This discovers the CSV file(s) under `data/raw/`. To specify particular files:

```bash
python run.py --csv "data/raw/your_downloaded_file.csv"
```

If the source uses a different column name, stop and check it first. The parser
will print the available names. A confirmed name can be supplied explicitly:

```bash
python run.py --csv "data/raw/your_downloaded_file.csv" --url-column url --label-column label
```

A preparation-only diagnostic command is available:

```bash
python run.py --prepare-only
```

Do not describe preparation-only output as completed model training.

The scripts print progress, fail on errors, and keep each run in a separate UTC
 timestamped directory under `outputs/`. Raw input files are read only. No DNS
lookups or visits to dataset websites occur. Model training uses the CPU; no GPU
setup is required. Runtime will be measured on your Mac, not estimated as a result.

## 4. Send back the evidence

After a successful full run, Terminal prints the exact path to:

`outputs/<run_timestamp>/report3_evidence.zip`

Upload that ZIP in our chat. Also capture a Terminal screenshot of the successful
run so we can include genuine execution evidence if useful. The ZIP contains
aggregate summaries, defanged examples, logs, settings, and figures, without the
large source data, prepared rows, complete predictions, or serialized models.
The logs include local paths and environment information; review them before
sharing beyond this project.

If a run fails, send the error or `outputs/<run_timestamp>/run.log` instead.
No finished research results are included in this starter project.

## Code structure

- `run.py`: orchestrates one run, logging, provenance, and the evidence bundle.
- `src/data.py`: loads, profiles, cleans, validates, and splits the data.
- `src/features.py`: defines and calculates 22 lexical/structural URL features.
- `src/models.py`: fixed Logistic Regression and Random Forest baselines.
- `src/evaluate.py`: metrics, serialized models, predictions, and charts.
- `src/reporting.py`: writes a summary from actual results.
- `tests/`: focused feature, cleaning, split, metric, and integration checks.
- `FEATURES.md`: exact definitions and parsing/cleaning rules.
- `config.json`: experiment settings and dataset identity.

`src/explain.py` and `src/generalization.py` are not implemented yet. Their
absence is intentional and should be reflected in the report's status table.

## Interpretation

Keep the model settings fixed for this first baseline. Looking at the test set
and then repeatedly changing settings would make it less useful as an unbiased
comparison. Later tuning should use training-only cross-validation and be
reported as a separate experiment. High scores do not prove real-world phishing
coverage, and poor scores still provide valid results for analysis.

The first fixed fold of a five-fold grouped splitter is used only to create an
approximately 80/20 partition. This is not a completed five-fold evaluation.
A majority-class accuracy reference is included to contextualize imbalance.

PhiUSIIL source for the later generalization phase:
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

PhiUSIIL uses **1=legitimate and 0=phishing**, opposite to URL-Phish. Do not feed it
into this baseline command. Future evaluation must map labels, extract the same
features, check overlap with training domains/URLs, and reuse training-fitted
preprocessing without retraining on the secondary evaluation data.
