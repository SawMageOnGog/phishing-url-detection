#!/usr/bin/env python3
"""Report 5: reproducible misclassification and dataset-shift audit.

Reads SAVED feature tables and selected models; never fetches URLs or retrains.
Raw URLs and source identifiers are intentionally omitted from exported error examples.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Exact feature order from src/features.py; importing that parser is unnecessary for
# saved feature tables and would require its URL-parsing dependencies.
FEATURES = ['url_length','hostname_length','path_length','query_length',
'subdomain_count','path_component_count','digit_count','digit_ratio',
'special_character_count','special_character_ratio','dot_count',
'hyphen_count','at_count','percent_count','query_parameter_count',
'is_https','hostname_is_ip','has_explicit_port','url_entropy',
'hostname_entropy','has_userinfo','fragment_length']

MODEL_FILES = {'Logistic Regression': 'logistic_regression_cv_selected.joblib',
               'Random Forest': 'random_forest_cv_selected.joblib'}
FEATURE_SUBSET = ['url_length','hostname_length','path_length','digit_ratio',
                  'url_entropy','hostname_entropy','is_https','subdomain_count']

def evaluate_subset(df, model, source, threshold=0.5):
    """Calculate rows and feature values for a selected-source evaluation subset."""
    if df.empty or not set(FEATURES).issubset(df.columns):
        raise ValueError('Missing required feature columns or empty evaluation subset')
    if not df['label'].isin([0, 1]).all():
        raise ValueError('Expected project labels 0=legitimate and 1=phishing')
    score = model.predict_proba(df[FEATURES])[:, 1]
    pred = (score >= threshold).astype(int)
    result = df[['label', *FEATURE_SUBSET]].reset_index(drop=True).copy()
    result['probability'] = score
    result['prediction'] = pred
    result['outcome'] = np.select([ (result.label==1)&(pred==1),
                                    (result.label==0)&(pred==0),
                                    (result.label==0)&(pred==1),
                                    (result.label==1)&(pred==0)],
                                  ['TP','TN','FP','FN'], default='unknown')
    result['source'] = source
    return result

def summaries(df, model_name):
    counts = df.outcome.value_counts()
    tp, tn, fp, fn = [int(counts.get(k,0)) for k in ('TP','TN','FP','FN')]
    return {'model':model_name,'source':df.source.iloc[0], 'n':len(df),
            'TP':tp,'TN':tn,'FP':fp,'FN':fn,
            'phishing_recall':tp/(tp+fn) if tp+fn else np.nan,
            'false_positive_rate':fp/(fp+tn) if fp+tn else np.nan}

def save_distributions(primary, external, out):
    rows=[]
    for feature in FEATURE_SUBSET:
        for source, frame in [('primary_holdout',primary),('external_domain_disjoint',external)]:
            for label in (0,1):
                a=frame.loc[frame.label==label,feature].dropna().astype(float)
                rows.append({'feature':feature,'source':source,'label':label,
                             'count':len(a),'mean':a.mean(),'median':a.median(),
                             'p10':a.quantile(.1),'p90':a.quantile(.9)})
    pd.DataFrame(rows).to_csv(out/'feature_distribution_summary.csv', index=False)
    for feature in FEATURE_SUBSET:
        fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        for axis,label,title in zip(axes,(0,1),('Legitimate','Phishing')):
            for source,data in [('Primary holdout',primary),('External domain-disjoint',external)]:
                series=data.loc[data.label==label,feature].astype(float)
                # Histogram density and fixed quantile clipping reduce domination by outliers.
                cutoff=max(primary[feature].quantile(.99), external[feature].quantile(.99))
                axis.hist(series.clip(upper=cutoff),bins=30,density=True,alpha=.5,label=source)
            axis.set_title(title);axis.set_xlabel(feature);axis.set_ylabel('Density')
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out/f'shift_{feature}.png',dpi=145)
        plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,help='October baseline run directory')
    p.add_argument('--validation',required=True,help='November selected-model run directory')
    p.add_argument('--external',required=True,help='PhiUSIIL generalization output directory')
    p.add_argument('--output',required=True,help='New/empty output directory')
    args=p.parse_args()
    base,valid,ext,out=map(Path,(args.baseline,args.validation,args.external,args.output))
    if out.exists() and any(out.iterdir()): p.error('Output directory must be new or empty')
    out.mkdir(parents=True,exist_ok=True)
    primary=pd.read_csv(base/'prepared_data.csv')
    partitions=pd.read_csv(base/'split_assignments.csv',usecols=['source_file','source_row','partition'])
    if primary.duplicated(['source_file','source_row']).any() or partitions.duplicated(['source_file','source_row']).any():
        raise ValueError('Ambiguous source row identity')
    primary=primary.merge(partitions,on=['source_file','source_row'],validate='one_to_one')
    primary=primary.loc[primary.partition=='test'].reset_index(drop=True)
    external=pd.read_csv(ext/'external_prepared_features.csv')
    for flag in ('source_url_overlap','source_domain_overlap'):
        if external[flag].dtype!='bool':
            external[flag]=external[flag].map({'True':True,'False':False,True:True,False:False})
        if external[flag].isna().any():raise ValueError(f'Unrecognized overlap flag: {flag}')
    external=external.loc[~external.source_domain_overlap & ~external.source_url_overlap].reset_index(drop=True)
    if len(primary)!=23015 or len(external)!=190782:
        raise ValueError(f'Unexpected evaluation subsets: primary={len(primary)}, external={len(external)}')
    summary=[];feature_rows=[]
    for model_name,filename in MODEL_FILES.items():
        model=joblib.load(valid/'evidence'/filename)
        for source,data in [('primary_holdout',primary),('external_domain_disjoint',external)]:
            evaluated=evaluate_subset(data,model,source)
            summary.append(summaries(evaluated,model_name))
            for outcome,group in evaluated.groupby('outcome'):
                for feature in FEATURE_SUBSET:
                    feature_rows.append({'model':model_name,'source':source,'outcome':outcome,
                                         'feature':feature,'n':len(group),
                                         'mean':group[feature].mean(),'median':group[feature].median()})
            # Representative errors (greatest-confidence wrong predictions); numerical features only.
            errors=evaluated.loc[evaluated.outcome.isin(['FP','FN'])].copy()
            errors['error_confidence']=np.where(errors.outcome=='FP',errors.probability,1-errors.probability)
            errors=(errors.sort_values('error_confidence',ascending=False)
                        .groupby('outcome',group_keys=False).head(25))
            errors.to_csv(out/f'error_examples_{model_name.lower().replace(" ","_")}_{source}.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'error_summary.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(out/'outcome_feature_summary.csv',index=False)
    save_distributions(primary,external,out)
    (out/'analysis_manifest.json').write_text(json.dumps({'status':'completed','threshold':.5,
        'models_retrained':False,'primary_n':len(primary),'external_domain_disjoint_n':len(external),
        'note':'Source-comparison figures describe observed distributions, not the cause of model errors. URL identifiers are omitted.'},indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))
    print('Saved:',out.resolve())
if __name__=='__main__':main()
