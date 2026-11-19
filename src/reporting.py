"""Write plain-language evidence notes from actual outputs, never estimated results."""
from pathlib import Path

def write_summary(out, profile, split, config, results):
    lines = ['# Actual run summary', '',
        'These values describe this run, not the dataset publication or another machine.', '',
        f"- Input rows: {profile['input_rows']:,}", f"- Retained rows: {profile['retained_rows']:,}",
        f"- Excluded rows: {profile['excluded_rows']:,}",
        f"- Duplicate URL rows removed: {profile['duplicate_rows_removed']:,}",
        f"- Rows quarantined for conflicting labels: {profile['conflicting_rows_removed']:,}",
        f"- Retained legitimate: {profile['class_counts_after_cleaning']['legitimate']:,}",
        f"- Retained phishing: {profile['class_counts_after_cleaning']['phishing']:,}",
        f"- Recomputed lexical features: {len(profile['features'])}",
        f"- Train rows: {split['train_rows']:,}", f"- Test rows: {split['test_rows']:,} ({split['test_fraction']:.2%})",
        f"- Shared registrable domains: {split['domain_overlap']}", f"- Shared canonical URLs: {split['canonical_url_overlap']}", '',
        '## Baseline model results', '']
    if results is None:
        lines += ['Preparation only. No model was trained or evaluated in this run.']
    else:
        lines += ['Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Fit seconds',
                  '--- | --- | --- | --- | --- | --- | --- | ---']
        for row in results.itertuples():
            lines.append(f'{row.model} | {row.accuracy:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.roc_auc:.4f} | {row.false_positive_rate:.4f} | {row.train_seconds:.2f}')
    lines += ['', '## Method and limits', '',
        'The positive class is phishing (1). A score of at least 0.5 is classified as phishing. Scores are model outputs and have not been calibrated as real-world risk probabilities.',
        'The fixed held-out fold comes from StratifiedGroupKFold with five folds and seed 42. Only the fixed held-out fold was evaluated; this is not a completed five-fold cross-validation experiment. The exact test fraction may differ from 20% because domains stay intact.',
        'Logistic Regression includes training-fitted standardization and balanced class weights. Random Forest uses 300 trees, minimum leaf size 2, square-root feature sampling, balanced subsample weights, and four workers. These are initial fixed settings, not tuned optima.',
        'The majority-class reference always predicts the class most common in the training partition. It contextualizes accuracy on imbalanced data; it is not a third trained research model.',
        'Source numerical features are ignored. All 22 model features are recomputed with the same implementation from stored URL text. This feature set is our project schema, not a replication of the source dataset’s 22 features.',
        'The registrable-domain grouping uses tldextract 5.3.0’s bundled public suffix snapshot, with private domains grouped under their public parent. This is conservative for shared hosting. It does not establish campaign-level independence.',
        'Removing duplicates and using separate domains reduces leakage but does not establish real-world deployment performance. URL-Phish’s benign collection sources and class imbalance can introduce source bias.',
        'Hyperparameter tuning, full cross-validation evaluation, SHAP, and cross-dataset evaluation have not been completed by this baseline command. PhiUSIIL is reserved for a separate later experiment and is not read by this command.',
        'If the counts differ from the published version 2 description, review input filenames and provenance before citing these results.', '',
        '## Files for Progress Report 3', '',
        '- data_summary.json: source hashes, counts, missing-value profile, cleaning decisions, and feature names.',
        '- feature_samples.csv: actual parsed examples with defanged URLs and feature values.',
        '- split_summary.json: class counts and verified zero train/test domain and URL overlap.',
        '- metrics.csv and confusion-matrix/curve PNGs: actual baseline evaluation, if completed.',
        '- run.log and environment.json: execution and environment evidence.',
        '- prediction_error_examples.csv: examples for analysis, not SHAP explanations.',
        '- report3_evidence.zip: compact bundle to upload in our chat.', '']
    Path(out, 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
