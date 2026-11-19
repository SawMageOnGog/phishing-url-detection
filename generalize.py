#!/usr/bin/env python3
"""Evaluate saved classifiers on external PhiUSIIL URLs, with explicit overlap audits.

No retraining or test-set threshold optimization. URL-only, offline feature extraction.
"""
import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.features import FEATURES, extract_features
from src.evaluate import calculate_metrics

LOG = logging.getLogger('generalization')

def label_to_project(value):
    """Official UCI PhiUSIIL: 0=phishing, 1=legitimate; project: 1=phishing."""
    label = str(value).strip()
    if label not in ('0', '1'):
        raise ValueError('PhiUSIIL label must be exactly 0 or 1')
    return 1 - int(label)

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def prepare_external(path, baseline, output, chunk_size=5000):
    """Parse raw URLs, map labels, reject invalid/duplicated URLs, and flag source overlap."""
    baseline = pd.read_csv(baseline, usecols=['canonical_url','domain_group'], dtype=str)
    source_urls = set(baseline['canonical_url'].dropna())
    source_domains = set(baseline['domain_group'].dropna())
    seen = set()
    count = {'raw':0,'invalid':0,'duplicate':0,'valid_unique':0,
             'source_url_overlap':0,'source_domain_overlap':0,'independent':0}
    output = Path(output)
    rows = []
    reader = pd.read_csv(path, dtype=str, chunksize=chunk_size, keep_default_na=False, encoding='utf-8-sig')
    for part in reader:
        if 'URL' not in part.columns or 'label' not in part.columns:
            # UCI official uses uppercase URL and lowercase label? Check both case-insensitively.
            lookup={c.casefold():c for c in part.columns}
            if 'url' not in lookup or 'label' not in lookup:
                raise ValueError('Expected one URL and one label column in PhiUSIIL CSV')
            url_col,label_col=lookup['url'],lookup['label']
        else: url_col,label_col='URL','label'
        for url,label in zip(part[url_col],part[label_col]):
            count['raw'] += 1
            try:
                target=label_to_project(label)
                values,meta=extract_features(url)
            except (ValueError,UnicodeError,TypeError):
                count['invalid'] += 1
                continue
            canon=meta['canonical_url']
            if canon in seen:
                count['duplicate'] += 1
                continue
            seen.add(canon)
            count['valid_unique'] += 1
            url_overlap=canon in source_urls
            domain_overlap=meta['domain_group'] in source_domains
            count['source_url_overlap'] += int(url_overlap)
            count['source_domain_overlap'] += int(domain_overlap)
            count['independent'] += int(not url_overlap and not domain_overlap)
            rows.append({'label':target,'source_url_overlap':url_overlap,
                         'source_domain_overlap':domain_overlap,**values})
        if count['raw'] % 25000 < chunk_size:
            LOG.info('Processed %s source rows',count['raw'])
    df=pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No valid URLs after parsing')
    df.to_csv(output/'external_prepared_features.csv',index=False)
    return df,count

def evaluate_external(data,models,out,threshold=0.5):
    results=[]
    for name, model_path in models.items():
        model=joblib.load(model_path)
        for subset_name,subset in [('all_valid_unique',data),
                ('no_exact_url_overlap',data.loc[~data.source_url_overlap]),
                ('domain_disjoint',data.loc[~data.source_domain_overlap & ~data.source_url_overlap])]:
            if subset.empty or subset.label.nunique()!=2:
                LOG.warning('Skipped %s / %s: empty or only one class',name,subset_name)
                continue
            scores=np.asarray(model.predict_proba(subset[FEATURES])[:,1],dtype=float)
            metrics=calculate_metrics(subset.label.astype(int).to_numpy(),scores,threshold)
            results.append({'model':name,'subset':subset_name,'n':len(subset),
                            'legitimate':int((subset.label==0).sum()),'phishing':int((subset.label==1).sum()),**metrics})
            LOG.info('%s / %s: n=%d, recall=%.4f, FPR=%.4f',name,subset_name,len(subset),metrics['recall'],metrics['false_positive_rate'])
    pd.DataFrame(results).to_csv(out/'external_metrics.csv',index=False)
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,help='Original PhiUSIIL CSV from UCI')
    parser.add_argument('--baseline',required=True,help='October baseline output directory')
    parser.add_argument('--validation',required=True,help='Selected model validation output directory')
    parser.add_argument('--output',help='Optional new output directory')
    args=parser.parse_args()
    dataset=Path(args.dataset).expanduser().resolve()
    baseline=Path(args.baseline).expanduser().resolve()
    validation=Path(args.validation).expanduser().resolve()
    for p in [dataset,baseline/'prepared_data.csv',validation/'evidence'/'logistic_regression_cv_selected.joblib',validation/'evidence'/'random_forest_cv_selected.joblib']:
        if not p.is_file(): parser.error(f'Missing required file: {p}')
    out=Path(args.output).expanduser().resolve() if args.output else Path('outputs')/('report4_external_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'generalization.log')])
    try:
        data,counts=prepare_external(dataset,baseline/'prepared_data.csv',out)
        models={'Logistic Regression':validation/'evidence'/'logistic_regression_cv_selected.joblib',
                'Random Forest':validation/'evidence'/'random_forest_cv_selected.joblib'}
        rows=evaluate_external(data,models,out)
        manifest={'status':'completed','source_csv_name':dataset.name,'source_sha256':sha256(dataset),
                  'official_phiusiil_mapping':{'0':'phishing','1':'legitimate'},
                  'project_mapping':{'0':'legitimate','1':'phishing'},
                  'preprocessing_counts':counts,'models_retrained':False,'threshold':0.5,
                  'warning':'Cross-source domain overlap can be extensive. Domain-disjoint subset is the strictest comparison; source collection and label biases remain.'}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (out/'status.json').write_text('{"status":"completed"}\n')
        print('\nGENERALIZATION COMPLETE:',out.resolve())
        print(pd.DataFrame(rows).to_string(index=False))
    except Exception:
        LOG.exception('External evaluation failed')
        (out/'status.json').write_text('{"status":"failed"}\n')
        raise
if __name__=='__main__':main()
