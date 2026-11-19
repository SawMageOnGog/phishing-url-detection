#!/usr/bin/env python3
"""Report 4 SHAP analysis of previously selected models; never retrains or visits URLs."""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.features import FEATURES

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('shap_analysis')
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression_cv_selected.joblib',
    'Random Forest': 'random_forest_cv_selected.joblib',
}


def sample_cases(test, score, seed, max_samples):
    """Deterministically sample all confusion-matrix groups if available."""
    rng = np.random.default_rng(seed)
    actual = test['label'].to_numpy(dtype=int)
    predicted = (score >= 0.5).astype(int)
    groups = {
        'true_negative': np.where((actual == 0) & (predicted == 0))[0],
        'false_positive': np.where((actual == 0) & (predicted == 1))[0],
        'false_negative': np.where((actual == 1) & (predicted == 0))[0],
        'true_positive': np.where((actual == 1) & (predicted == 1))[0],
    }
    # Round-robin sampling distributes the sample across the four error classes.
    picks = []
    shuffled = {k: rng.permutation(v).tolist() for k, v in groups.items()}
    while len(picks) < max_samples and any(shuffled.values()):
        for k in groups:
            if shuffled[k] and len(picks) < max_samples:
                picks.append((shuffled[k].pop(), k))
    return picks, {k: int(len(v)) for k, v in groups.items()}


def phishing_values(explanation, count, n_features):
    """Normalize SHAP API binary-class shapes to (rows, features), output probability or log-odds."""
    val = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if val.shape == (count, n_features, 2):
        val = val[:, :, 1]
        if base.ndim == 2:
            base = base[:, 1]
        elif base.ndim == 1 and len(base) == 2:
            base = np.repeat(base[1], count)
    elif val.shape != (count, n_features):
        raise ValueError(f'Unrecognized SHAP shape {val.shape}')
    if base.ndim == 0:
        base = np.repeat(float(base), count)
    if len(base) != count:
        raise ValueError(f'Unexpected base_values shape {base.shape}')
    return val, base


def explain_one(model_name, model, train, test, out, seed=42, samples=48, background_size=40):
    out.mkdir(parents=True, exist_ok=False)
    probabilities = model.predict_proba(test[FEATURES])[:, list(model.classes_).index(1)]
    picks, cases = sample_cases(test, probabilities, seed, samples)
    if not picks:
        raise ValueError('No test observations sampled.')
    indices, group_names = zip(*picks)
    subset = test.iloc[list(indices)].reset_index(drop=True)
    raw = subset[FEATURES].astype(float)
    background = train[FEATURES].sample(n=min(background_size, len(train)), random_state=seed).astype(float)

    if isinstance(model, Pipeline):
        scaler = model.named_steps['scale']
        classifier = model.named_steps['classifier']
        transformed = pd.DataFrame(scaler.transform(raw), columns=FEATURES)
        bg_transformed = pd.DataFrame(scaler.transform(background), columns=FEATURES)
        explainer = shap.LinearExplainer(classifier, bg_transformed)
        explanation = explainer(transformed)
        unit = 'log-odds'
        predicted_link = np.log(np.clip(probabilities[list(indices)], 1e-12, 1 - 1e-12) /
                                np.clip(1 - probabilities[list(indices)], 1e-12, 1 - 1e-12))
    else:
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation='interventional', model_output='probability')
        explanation = explainer(raw, check_additivity=False)
        unit = 'phishing probability'
        predicted_link = probabilities[list(indices)]
    values, base = phishing_values(explanation, len(raw), len(FEATURES))
    expected = base + values.sum(axis=1)
    max_diff = float(np.max(np.abs(expected - predicted_link)))
    if max_diff > 0.025:
        raise ValueError(f'SHAP additivity failed for {model_name}: maximum difference {max_diff:.6f}')

    # Input feature values in original URL-feature units, even for scaled LR inputs.
    rows = []
    for j, (original_index, category) in enumerate(picks):
        rows.append({
            'test_index': int(original_index), 'case': category,
            'actual_label': int(subset.iloc[j]['label']),
            'predicted_label': int(probabilities[original_index] >= 0.5),
            'phishing_probability': float(probabilities[original_index]),
            'shap_base': float(base[j]), 'shap_sum_plus_base': float(expected[j]),
            'shap_output_unit': unit,
        })
    pd.DataFrame(rows).to_csv(out / 'explained_cases.csv', index=False)
    pd.DataFrame(values, columns=FEATURES).to_csv(out / 'shap_values.csv', index=False)
    pd.DataFrame(raw).to_csv(out / 'explained_features.csv', index=False)

    importance = pd.DataFrame({'feature': FEATURES, 'mean_absolute_shap': np.mean(np.abs(values), axis=0)})
    importance.sort_values('mean_absolute_shap', ascending=False).to_csv(out / 'global_feature_importance.csv', index=False)
    top = importance.nlargest(15, 'mean_absolute_shap').sort_values('mean_absolute_shap')
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top['feature'], top['mean_absolute_shap'])
    ax.set_xlabel(f'Mean absolute SHAP value ({unit})')
    ax.set_title(f'{model_name}: held-out URL explanation sample (n={len(raw)})')
    fig.tight_layout()
    fig.savefig(out / 'global_feature_importance.png', dpi=160)
    plt.close(fig)

    for cat in ('true_positive', 'true_negative', 'false_positive', 'false_negative'):
        if cat not in group_names:
            continue
        j = group_names.index(cat)
        local = shap.Explanation(values=values[j], base_values=base[j],
                                 data=raw.iloc[j].to_numpy(), feature_names=FEATURES)
        shap.plots.waterfall(local, max_display=12, show=False)
        plt.title(f'{model_name}: {cat.replace("_", " ")}', fontsize=10)
        plt.tight_layout()
        plt.savefig(out / f'local_{cat}.png', dpi=140, bbox_inches='tight')
        plt.close('all')
    summary = {'model': model_name, 'sample_size': int(len(raw)), 'case_counts_in_test': cases,
               'case_counts_in_sample': {x: group_names.count(x) for x in cases},
               'random_seed': seed, 'background_size': len(background), 'shap_output_unit': unit,
               'maximum_additivity_difference': max_diff,
               'warning': 'SHAP describes model behavior, not causal risk. Sample is intentionally case-balanced, not representative of prevalence.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    LOG.info('%s: explained %s cases, max additivity difference %.6f', model_name, len(raw), max_diff)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--background', type=int, default=40)
    parser.add_argument('--model', choices=['both', 'logistic', 'forest'], default='both')
    args = parser.parse_args()
    if not 4 <= args.samples <= 200 or not 10 <= args.background <= 100:
        parser.error('Samples must be 4–200 and background must be 10–100.')
    baseline, validation = args.baseline.resolve(), args.validation.resolve()
    if json.loads((baseline / 'run_status.json').read_text())['status'] != 'completed':
        parser.error('Baseline is not complete.')
    if json.loads((validation / 'validation_status.json').read_text())['status'] != 'completed':
        parser.error('Validation is not complete.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = ROOT / 'outputs' / ('report4_shap_' + stamp)
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / 'shap.log')])
    try:
        prepared = pd.read_csv(baseline / 'prepared_data.csv', low_memory=False)
        assignments = pd.read_csv(baseline / 'split_assignments.csv', low_memory=False)
        assert len(prepared) == len(assignments)
        for key in ('source_file','source_row','domain_group','label'):
            if not prepared[key].astype(str).reset_index(drop=True).equals(assignments[key].astype(str).reset_index(drop=True)):
                raise ValueError(f'Data alignment failure: {key}')
        train = prepared[assignments['partition'].eq('train')].reset_index(drop=True)
        test = prepared[assignments['partition'].eq('test')].reset_index(drop=True)
        if set(train['domain_group']) & set(test['domain_group']):
            raise ValueError('Domain overlap between train/test.')
        names = list(MODEL_FILES)
        if args.model == 'logistic': names = ['Logistic Regression']
        if args.model == 'forest': names = ['Random Forest']
        summaries = []
        for name in names:
            model = joblib.load(validation / 'evidence' / MODEL_FILES[name])
            summaries.append(explain_one(name, model, train, test, out / name.lower().replace(' ', '_'),
                                         samples=args.samples, background_size=args.background))
        (out / 'manifest.json').write_text(json.dumps({
            'status': 'completed', 'baseline': str(baseline), 'validation': str(validation),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'models': summaries,
            'note': 'Saved fitted models were reused; no training or test-set tuning performed.'}, indent=2) + '\n')
        (out / 'status.json').write_text('{"status":"completed"}\n')
        print('\nSHAP COMPLETE: ' + str(out))
    except Exception:
        LOG.exception('SHAP analysis failed; do not report partial plots as final evidence.')
        (out / 'status.json').write_text('{"status":"failed"}\n')
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
