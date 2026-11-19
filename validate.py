#!/usr/bin/env python3
"""Report 4: training-only grouped CV, limited model tuning, and held-out comparison.

Point --baseline at a completed baseline output directory. The original files are never changed.
"""
import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

from src.features import FEATURES
from src.models import build_models
from src.evaluate import calculate_metrics

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('validation')


def candidates(config):
    """Small, documented search. Baseline candidate is always included."""
    lr = build_models(config)['Logistic Regression']
    rf = build_models(config)['Random Forest']
    lr_alt = clone(lr).set_params(classifier__C=0.1)
    rf_alt = clone(rf).set_params(min_samples_leaf=5)
    return {
        'Logistic Regression': [('baseline_C1', lr), ('regularized_C0.1', lr_alt)],
        'Random Forest': [('baseline_leaf2', rf), ('regularized_leaf5', rf_alt)],
    }


def checked_splits(x, y, groups, seed, folds):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, val) in enumerate(splitter.split(x, y, groups), start=1):
        if len(set(groups.iloc[train]) & set(groups.iloc[val])):
            raise AssertionError('Validation domain leakage detected.')
        if len(set(y.iloc[train])) != 2 or len(set(y.iloc[val])) != 2:
            raise ValueError(f'CV fold {fold} lacks one of the classes; select a different fold count or dataset.')
        yield fold, train, val


def validate_frame(frame, assignments, config, out, folds=3, selected_models=None):
    """Save per-fold validation, select by mean AP, and evaluate the unchanged held-out set."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if len(frame) != len(assignments):
        raise ValueError('Prepared data and split assignments have different row counts.')
    for col in ('source_file', 'source_row', 'domain_group', 'label'):
        if not frame[col].astype(str).reset_index(drop=True).equals(assignments[col].astype(str).reset_index(drop=True)):
            raise ValueError(f'Misalignment of prepared data and split assignments: {col}')
    if len(FEATURES) != 22 or not set(FEATURES).issubset(frame.columns):
        raise ValueError('Missing or changed expected 22 URL features.')
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite feature detected.')
    if not set(assignments['partition']) == {'train', 'test'}:
        raise ValueError('Expected only train and test partitions.')
    train = frame.loc[assignments['partition'].eq('train')].reset_index(drop=True)
    test = frame.loc[assignments['partition'].eq('test')].reset_index(drop=True)
    if set(train['domain_group']) & set(test['domain_group']):
        raise ValueError('Original train/test domain separation is invalid.')
    x, y, groups = train[FEATURES], train['label'].astype(int), train['domain_group']
    tx, ty = test[FEATURES], test['label'].astype(int)
    splits = list(checked_splits(x, y, groups, config['seed'], folds))
    selection_rows = []
    fold_rows = []
    test_rows = []
    for model_name, configs in candidates(config).items():
        if selected_models and model_name not in selected_models:
            continue
        for candidate_id, prototype in configs:
            LOG.info('Validating %s / %s (%s folds)', model_name, candidate_id, folds)
            for fold, tr, va in splits:
                start = time.perf_counter()
                model = clone(prototype)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always', ConvergenceWarning)
                    model.fit(x.iloc[tr], y.iloc[tr])
                if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                    raise RuntimeError(f'Non-convergence: {model_name} / {candidate_id} fold {fold}')
                score = model.predict_proba(x.iloc[va])[:, list(model.classes_).index(1)]
                measures = calculate_metrics(y.iloc[va], score, config['decision_threshold'])
                fold_rows.append({'model': model_name, 'candidate': candidate_id, 'fold': fold,
                    'train_rows': int(len(tr)), 'validation_rows': int(len(va)),
                    'train_domains': int(groups.iloc[tr].nunique()), 'validation_domains': int(groups.iloc[va].nunique()),
                    'validation_phishing': int(y.iloc[va].sum()),
                    'fit_predict_seconds': round(time.perf_counter() - start, 3), **measures})
            subset = [r for r in fold_rows if r['model'] == model_name and r['candidate'] == candidate_id]
            selection_rows.append({'model': model_name, 'candidate': candidate_id,
                'mean_average_precision': float(np.mean([r['average_precision'] for r in subset])),
                'std_average_precision': float(np.std([r['average_precision'] for r in subset], ddof=1)),
                'mean_recall': float(np.mean([r['recall'] for r in subset])),
                'mean_fpr': float(np.mean([r['false_positive_rate'] for r in subset]))})
        ranked = sorted((r for r in selection_rows if r['model'] == model_name),
                        key=lambda r: (-r['mean_average_precision'], r['candidate']))
        best = ranked[0]
        best_id = best['candidate']
        LOG.info('Selected %s / %s using mean validation average precision %.4f',
                 model_name, best_id, best['mean_average_precision'])
        prototype = next(p for cid, p in configs if cid == best_id)
        fitted = clone(prototype).fit(x, y)
        scores = fitted.predict_proba(tx)[:, list(fitted.classes_).index(1)]
        test_rows.append({'model': model_name, 'selected_candidate': best_id,
                          'selection_criterion': 'highest mean training-only CV average precision',
                          **calculate_metrics(ty, scores, config['decision_threshold'])})
        slug = model_name.lower().replace(' ', '_')
        joblib.dump(fitted, out / f'{slug}_cv_selected.joblib')
    pd.DataFrame(fold_rows).to_csv(out / 'validation_folds.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(out / 'validation_candidate_summary.csv', index=False)
    pd.DataFrame(test_rows).to_csv(out / 'cv_selected_test_metrics.csv', index=False)
    return fold_rows, selection_rows, test_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True, help='Completed baseline output directory containing prepared_data.csv and split_assignments.csv')
    p.add_argument('--folds', type=int, default=3, choices=[3, 4, 5])
    p.add_argument('--model', choices=['both', 'logistic', 'forest'], default='both')
    args = p.parse_args()
    baseline = args.baseline.resolve()
    if not baseline.is_dir():
        p.error('Baseline folder does not exist.')
    status = json.loads((baseline / 'run_status.json').read_text())
    if status.get('status') != 'completed' or status.get('mode') != 'baseline_models':
        p.error('A successfully completed baseline_models run is required.')
    config = json.loads((baseline / 'config_used.json').read_text())
    if config['decision_threshold'] != 0.5:
        p.error('Expected the documented 0.5 decision threshold.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = ROOT / 'outputs' / ('report4_cv_' + stamp)
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / 'validation.log')])
    try:
        frame = pd.read_csv(baseline / 'prepared_data.csv', low_memory=False)
        assignments = pd.read_csv(baseline / 'split_assignments.csv', low_memory=False)
        model_names = {'both': None, 'logistic': {'Logistic Regression'}, 'forest': {'Random Forest'}}[args.model]
        # validate_frame owns a fresh directory; keep logs outside that subfolder.
        evidence = out / 'evidence'
        _, selection, test_rows = validate_frame(frame, assignments, config, evidence, args.folds, model_names)
        manifest = {'status': 'completed', 'baseline_directory': str(baseline),
                    'baseline_config': config, 'validation_folds': args.folds, 'model_request': args.model,
                    'selection_metric': 'mean average precision', 'held_out_threshold': 0.5,
                    'caution': 'The original held-out test was already examined in Report 3. This is an extension, not a new independent holdout.',
                    'completed_utc': datetime.now(timezone.utc).isoformat()}
        (out / 'validation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (out / 'validation_status.json').write_text(json.dumps({'status':'completed'}) + '\n')
        print('\nVALIDATION COMPLETE: ' + str(out))
        print(pd.DataFrame(selection).to_string(index=False))
        print(pd.DataFrame(test_rows).to_string(index=False))
    except Exception:
        LOG.exception('Validation failed; do not report partial results as completed.')
        (out / 'validation_status.json').write_text(json.dumps({'status':'failed'}) + '\n')
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
