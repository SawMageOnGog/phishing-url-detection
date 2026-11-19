"""Unit checks for Report 4 leakage prevention and candidate definitions."""
import unittest
import pandas as pd
from validate import candidates, checked_splits

class ValidationTests(unittest.TestCase):
    def test_disjoint_groups(self):
        groups = pd.Series([f'group_{i//2}' for i in range(60)])
        y = pd.Series([i % 2 for i in range(60)])
        x = pd.DataFrame({'a': range(60)})
        folds = list(checked_splits(x, y, groups, 42, 3))
        self.assertEqual(len(folds), 3)
        for _, tr, va in folds:
            self.assertTrue(set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va])))
    def test_candidate_configuration(self):
        config = {'seed': 42, 'logistic_regression': {'C':1.0, 'max_iter':2000,'class_weight':'balanced','solver':'lbfgs'},
                  'random_forest': {'n_estimators':300,'min_samples_leaf':2,'max_features':'sqrt','class_weight':'balanced_subsample','n_jobs':4}}
        models = candidates(config)
        self.assertEqual(len(models['Logistic Regression']), 2)
        self.assertEqual(len(models['Random Forest']), 2)
        self.assertEqual(models['Logistic Regression'][1][1].get_params()['classifier__C'], 0.1)
        self.assertEqual(models['Random Forest'][1][1].get_params()['min_samples_leaf'], 5)

if __name__ == '__main__':
    unittest.main()
