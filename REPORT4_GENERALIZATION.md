# Progress Report 4 — PhiUSIIL external generalization stage

This package preserves earlier baseline, validation and SHAP artifacts. The secondary dataset is **not** bundled; obtain the original PhiUSIIL CSV from the UCI Machine Learning Repository (dataset 967): https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Place its `PhiUSIIL_Phishing_URL_Dataset.csv` in `data/raw/`. Do not substitute preengineered numeric columns for this project's 22 URL-derived features. This stage does not visit external URLs or retrain models.

Run from the project root:

```bash
python3 -m unittest discover -s tests -v
python3 generalize.py \
  --dataset data/raw/PhiUSIIL_Phishing_URL_Dataset.csv \
  --baseline outputs/20261023T021333_607052Z \
  --validation outputs/report4_cv_20261106T031758_233969Z
```

**Critical label mapping:** PhiUSIIL's official `0=phishing, 1=legitimate` is inverted to our project's `1=phishing, 0=legitimate`. The source file hash, mapping, validity/deduplication counts, URL overlap, and domain overlap are saved in the resulting `outputs/report4_external_...` folder. Three separately reported evaluation populations are provided: all valid unique URLs; no exact URL overlap; and no original-dataset registrable-domain overlap. Use the most stringent domain-disjoint results when discussing transfer, but also report how many records this excludes. Raw dataset is not modified.

The baseline and cross-validation configurations are not modified. These results assess the selected models on a *different* dataset; they are not an estimate of real-world safety, and any comparisons should discuss dataset bias and label prevalence.
