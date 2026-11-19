"""Load explicit URL-Phish labels, profile, quarantine, deduplicate, and group-split."""
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .features import FEATURES, extract_features, defang

LOG = logging.getLogger(__name__)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def class_counts(values):
    values = pd.Series(values)
    return {'legitimate': int((values == 0).sum()), 'phishing': int((values == 1).sum())}

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def resolve_column(frame, requested):
    matches = [c for c in frame.columns if str(c).strip().lower() == requested.lower()]
    if len(matches) != 1:
        raise ValueError(f'Expected one {requested!r} column. Available columns: {list(frame.columns)}. '
                         'Use --url-column or --label-column if the names differ; do not guess label meanings.')
    return matches[0]

def prepare(paths, out, config, url_column='url', label_column='label'):
    out = Path(out)
    frames, files, missing_summary = [], [], []
    for path in paths:
        path = Path(path)
        raw = pd.read_csv(path, dtype=str, encoding='utf-8-sig', keep_default_na=False)
        u, y = resolve_column(raw, url_column), resolve_column(raw, label_column)
        if u == y:
            raise ValueError('URL and label columns must differ.')
        files.append({'filename': path.name, 'sha256': sha256(path), 'rows': len(raw),
                      'columns': list(raw.columns), 'url_column': u, 'label_column': y})
        missing_summary.append({'filename': path.name,
            'blank_or_whitespace_cells': {str(c): int(raw[c].str.strip().eq('').sum()) for c in raw.columns}})
        frame = raw[[u, y]].rename(columns={u: 'url', y: 'label'}).copy()
        frame['source_file'] = path.name
        frame['source_row'] = np.arange(2, len(frame) + 2)  # CSV header is row 1.
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise ValueError('The input contains no records.')
    frame['url'] = frame['url'].str.strip()
    labels = frame['label'].str.strip()
    missing_labels = labels.eq('')
    numeric_labels = pd.to_numeric(labels, errors='coerce')
    bad = ~missing_labels & ~numeric_labels.isin([0, 1])
    if bad.any():
        raise ValueError(f'Unexpected labels: {labels[bad].value_counts().to_dict()}. '
                         'This command accepts URL-Phish only: 0=legitimate, 1=phishing.')
    frame['label'] = numeric_labels
    counts_before = class_counts(numeric_labels)
    records, rejected = [], []
    for n, row in enumerate(frame.itertuples(index=False), start=1):
        reason = ''
        if pd.isna(row.label):
            reason = 'missing_label'
        elif not row.url:
            reason = 'missing_url'
        else:
            try:
                features, metadata = extract_features(row.url)
            except (ValueError, UnicodeError) as exc:
                reason = 'invalid_url: ' + str(exc)
        if reason:
            rejected.append({'source_file': row.source_file, 'source_row': int(row.source_row), 'reason': reason})
        else:
            records.append({'url': row.url, 'label': int(row.label), 'source_file': row.source_file,
                            'source_row': int(row.source_row), **metadata, **features})
        if n % 10000 == 0:
            LOG.info('Parsed %s / %s records', f'{n:,}', f'{len(frame):,}')
    if not records:
        raise ValueError('No valid labeled URLs remain after parsing.')
    clean = pd.DataFrame(records)
    conflicts = clean.groupby('canonical_url', sort=False)['label'].nunique()
    conflict_keys = set(conflicts.index[conflicts > 1])
    conflict_mask = clean['canonical_url'].isin(conflict_keys)
    conflict_rows = clean.loc[conflict_mask]
    rejected.extend({'source_file': r.source_file, 'source_row': int(r.source_row),
                     'reason': 'conflicting_labels_for_canonical_url'} for r in conflict_rows.itertuples())
    clean = clean.loc[~conflict_mask].copy()
    duplicate_mask = clean.duplicated('canonical_url', keep='first')
    duplicate_count = int(duplicate_mask.sum())
    rejected.extend({'source_file': r.source_file, 'source_row': int(r.source_row),
                     'reason': 'duplicate_canonical_url'} for r in clean.loc[duplicate_mask].itertuples())
    clean = clean.loc[~duplicate_mask].reset_index(drop=True)
    if clean['label'].nunique() != 2:
        raise ValueError('Both classes must remain after cleaning.')
    if not np.isfinite(clean[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Extracted features contain NaN or infinity.')
    if len(clean) + len(rejected) != len(frame):
        raise AssertionError('Cleaning accounting does not match the input row count.')
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejected, columns=['source_file', 'source_row', 'reason']).to_csv(out / 'quarantine.csv', index=False)
    clean.to_csv(out / 'prepared_data.csv', index=False)
    reasons = pd.Series([r['reason'].split(':')[0] for r in rejected], dtype=str).value_counts().to_dict()
    summary = {'source': files, 'missing_values': missing_summary, 'input_rows': len(frame),
        'class_counts_before_cleaning': counts_before, 'retained_rows': len(clean),
        'class_counts_after_cleaning': class_counts(clean['label']),
        'excluded_rows': len(rejected), 'excluded_by_reason': {str(k): int(v) for k, v in reasons.items()},
        'duplicate_rows_removed': duplicate_count,
        'conflicting_canonical_urls': len(conflict_keys), 'conflicting_rows_removed': len(conflict_rows),
        'domain_groups': int(clean['domain_group'].nunique()), 'features': FEATURES,
        'label_mapping': config['label_mapping'], 'feature_source': 'Recomputed from URL text; source numerical features ignored',
        'published_expected_rows': config['published_expected_rows'],
        'row_count_matches_published_description': len(frame) == config['published_expected_rows'],
        'published_expected_phishing': config['published_expected_phishing'],
        'phishing_count_matches_published_description': counts_before['phishing'] == config['published_expected_phishing']}
    save_json(out / 'data_summary.json', summary)
    if not summary['row_count_matches_published_description'] or not summary['phishing_count_matches_published_description']:
        LOG.warning('Input counts differ from the version 2 description. Review source files before using results in the report.')
    # Class-balanced examples for review, with defanged URLs to avoid active links.
    examples = clean.groupby('label', group_keys=False).head(5).copy()
    examples['url_defanged'] = examples['url'].map(defang)
    examples[['source_file', 'source_row', 'label', 'url_defanged'] + FEATURES].to_csv(out / 'feature_samples.csv', index=False)
    LOG.info('Cleaning finished: %s retained, %s excluded', f'{len(clean):,}', f'{len(rejected):,}')
    return clean, summary

def create_split(frame, config, out):
    if frame['domain_group'].nunique() < config['split_folds']:
        raise ValueError('Too few distinct domain groups for the configured split.')
    splitter = StratifiedGroupKFold(n_splits=config['split_folds'], shuffle=True, random_state=config['seed'])
    # Fixed fold chosen before any model is fitted. Other folds are NOT scored here.
    chosen = config['held_out_fold']
    train_idx, test_idx = list(splitter.split(frame[FEATURES], frame['label'], frame['domain_group']))[chosen]
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    if train['label'].nunique() != 2 or test['label'].nunique() != 2:
        raise ValueError('Fixed split does not contain both classes in each partition. Review grouping before changing settings.')
    domains = set(train['domain_group']) & set(test['domain_group'])
    urls = set(train['canonical_url']) & set(test['canonical_url'])
    if domains or urls:
        raise AssertionError('Training/test leakage detected.')
    assignment = frame[['source_file', 'source_row', 'domain_group', 'label']].copy()
    assignment['partition'] = 'train'
    assignment.loc[test_idx, 'partition'] = 'test'
    assignment.to_csv(Path(out) / 'split_assignments.csv', index=False)
    summary = {'method': 'StratifiedGroupKFold fixed held-out fold; NOT a completed cross-validation evaluation',
        'group': 'Registrable domain from bundled Public Suffix List; private suffixes grouped under their public parent',
        'seed': config['seed'], 'n_splits': config['split_folds'], 'held_out_fold': chosen,
        'train_rows': len(train), 'test_rows': len(test), 'test_fraction': len(test) / len(frame),
        'train_class_counts': class_counts(train['label']), 'test_class_counts': class_counts(test['label']),
        'train_domains': int(train['domain_group'].nunique()), 'test_domains': int(test['domain_group'].nunique()),
        'domain_overlap': len(domains), 'canonical_url_overlap': len(urls)}
    save_json(Path(out) / 'split_summary.json', summary)
    LOG.info('Domain-separated split: %s train / %s test; overlapping domains: %s', len(train), len(test), len(domains))
    return train_idx, test_idx, summary
