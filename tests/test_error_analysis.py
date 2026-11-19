import unittest
import numpy as np
import pandas as pd
from analyze_errors import FEATURES
from analyze_errors import evaluate_subset, summaries

class DummyModel:
    def predict_proba(self,x):
        p=np.array([.8,.2,.7,.3])[:len(x)]
        return np.column_stack([1-p,p])

class ErrorAnalysisTests(unittest.TestCase):
    def test_outcome_accounting(self):
        d=pd.DataFrame({k:[0]*4 for k in FEATURES})
        for k in ('url_length','hostname_length','path_length','digit_ratio','url_entropy','hostname_entropy','is_https','subdomain_count'):
            d[k]=[0]*4
        d['label']=[1,1,0,0]
        result=evaluate_subset(d,DummyModel(),'test')
        self.assertEqual(result.outcome.tolist(),['TP','FN','FP','TN'])
        m=summaries(result,'mock')
        self.assertEqual((m['TP'],m['TN'],m['FP'],m['FN']),(1,1,1,1))
        self.assertEqual(m['phishing_recall'],.5)
    def test_reject_bad_labels(self):
        d=pd.DataFrame({k:[0] for k in FEATURES});d['label']=[9]
        with self.assertRaises(ValueError):evaluate_subset(d,DummyModel(),'test')
if __name__=='__main__':unittest.main()
