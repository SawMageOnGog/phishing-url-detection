# Progress Report 4 — SHAP explainability

The selected Logistic Regression and Random Forest models are reused from the completed November 5 validation run; **no retraining** occurs. Outputs are stored in a new timestamped directory and existing experiment outputs are never overwritten.

From the `PhishingURLProject` directory:

```bash
python3 -m pip install 'shap>=0.48,<0.52'
python3 explain.py \
  --baseline outputs/20261023T021333_607052Z \
  --validation outputs/report4_cv_20261106T031758_233969Z \
  --samples 48 \
  --background 40 \
  --model both
```

For each model, the run exports the explanation sample, original feature values, SHAP contributions, global mean absolute SHAP feature importance, four example waterfall plots (when available), an additivity check and the run manifest. Logistic Regression explanations use **log-odds**; Random Forest explanations use **predicted probability**. Do not compare raw SHAP magnitude across these two units. Selected cases intentionally cover both correct and incorrect predictions; they are not a representative prevalence sample. SHAP feature attribution explains a model's decision and does not establish malicious causation.

The extra outputs are exploratory explainability evidence. The held-out test set was already examined in Progress Report 3 and therefore is not a new independent test. Upload the new `outputs/report4_shap_...` folder for analysis and inclusion in Report 4.
