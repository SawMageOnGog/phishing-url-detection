"""Consistent positive-class metrics and saved publication-style figures."""
import logging
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve)
from .features import FEATURES, defang
from .models import build_models

LOG = logging.getLogger(__name__)

def calculate_metrics(y, scores, threshold):
    predictions = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {'accuracy': float(accuracy_score(y, predictions)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'f1': float(f1_score(y, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, scores)),
        'average_precision': float(average_precision_score(y, scores)),
        'false_positive_rate': float(fp / (fp + tn)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def train_evaluate(frame, train_idx, test_idx, config, out):
    out = Path(out)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    x_train, y_train = train[FEATURES], train['label']
    x_test, y_test = test[FEATURES], test['label']
    rows, model_scores = [], {}
    examples = []
    for name, model in build_models(config).items():
        slug = name.lower().replace(' ', '_')
        LOG.info('Training %s', name)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x_train, y_train)
        if any(issubclass(w.category, ConvergenceWarning) for w in recorded):
            raise RuntimeError(f'{name} did not converge. Stop and review the log before reporting these results.')
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        classes = model.classes_.tolist()
        scores = model.predict_proba(x_test)[:, classes.index(1)]
        predict_seconds = time.perf_counter() - started
        metrics = calculate_metrics(y_test, scores, config['decision_threshold'])
        metrics.update(model=name, train_seconds=fit_seconds, predict_seconds=predict_seconds)
        rows.append(metrics)
        model_scores[name] = scores
        predictions = (scores >= config['decision_threshold']).astype(int)
        # Keep a limited set of errors for later SHAP review; no claim of explanations yet.
        errors = test.loc[y_test.to_numpy() != predictions, ['url', 'label', 'source_file', 'source_row']].head(10).copy()
        errors['model'] = name
        positions = test.index.get_indexer(errors.index)
        errors['prediction'] = predictions[positions]
        errors['phishing_probability'] = scores[positions]
        errors['url'] = errors['url'].map(defang)
        examples.append(errors.rename(columns={'url': 'url_defanged'}))
        pd.DataFrame({'source_file': test['source_file'].to_numpy(), 'source_row': test['source_row'].to_numpy(),
            'label': y_test.to_numpy(), 'prediction': predictions, 'phishing_probability': scores}).to_csv(out / f'{slug}_predictions.csv', index=False)
        joblib.dump(model, out / f'{slug}.joblib')
        fig, ax = plt.subplots(figsize=(5.5, 4.8), layout='constrained')
        ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=[0, 1],
            display_labels=['Legitimate', 'Phishing'], colorbar=False, cmap='Blues', ax=ax, values_format=',d')
        ax.set_title(name + '\nDomain-separated held-out test set')
        fig.savefig(out / f'{slug}_confusion_matrix.png', dpi=200)
        plt.close(fig)
        LOG.info('%s: precision %.4f | recall %.4f | F1 %.4f | ROC-AUC %.4f | FPR %.4f',
            name, metrics['precision'], metrics['recall'], metrics['f1'], metrics['roc_auc'], metrics['false_positive_rate'])
    # Trivial reference baseline is not an additional trained research model.
    majority = int(y_train.value_counts().idxmax())
    rows.append({'model': 'Majority-class reference', **calculate_metrics(y_test,
        np.full(len(y_test), float(majority)), config['decision_threshold']), 'train_seconds': 0.0, 'predict_seconds': 0.0})
    results = pd.DataFrame(rows)
    results.to_csv(out / 'metrics.csv', index=False)
    pd.concat(examples, ignore_index=True).to_csv(out / 'prediction_error_examples.csv', index=False)
    for plot_type in ['roc', 'precision_recall']:
        fig, ax = plt.subplots(figsize=(6, 4.8), layout='constrained')
        for name, scores in model_scores.items():
            if plot_type == 'roc':
                a, b, _ = roc_curve(y_test, scores)
                ax.plot(a, b, label=name)
            else:
                precision, recall, _ = precision_recall_curve(y_test, scores)
                ax.plot(recall, precision, label=name)
        if plot_type == 'roc':
            ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance reference')
            ax.set(xlabel='False-positive rate', ylabel='True-positive rate', title='ROC curves')
        else:
            ax.axhline(float(y_test.mean()), linestyle='--', color='gray', label='Phishing prevalence')
            ax.set(xlabel='Recall', ylabel='Precision', title='Precision-recall curves')
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc='best');ax.grid(alpha=.2)
        fig.savefig(out / f'{plot_type}_curves.png', dpi=200)
        plt.close(fig)
    return results
