"""Focused checks with synthetic fixtures; these are not research results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.features import extract_features, entropy, FEATURES
from src.data import prepare, create_split
from src.evaluate import calculate_metrics, train_evaluate
from src.reporting import write_summary

ROOT = Path(__file__).resolve().parents[1]
def config():
    return json.loads((ROOT / 'config.json').read_text())

class FeatureTests(unittest.TestCase):
    def test_known_values(self):
        url = 'https://sub.example.co.uk/a/12?x=4&y=5#f'
        f, m = extract_features(url)
        self.assertEqual(m['domain_group'], 'example.co.uk')
        self.assertEqual(f['subdomain_count'], 1)
        self.assertEqual(f['digit_count'], 4)
        self.assertEqual(f['query_parameter_count'], 2)
        self.assertEqual(f['path_component_count'], 2)
        self.assertEqual(f['is_https'], 1)
        self.assertEqual(f['fragment_length'], 1)
        self.assertEqual(f['url_length'], len(url))
        self.assertAlmostEqual(f['digit_ratio'], 4 / len(url))
        self.assertEqual(list(f), FEATURES)
    def test_entropy(self):
        self.assertEqual(entropy(''), 0)
        self.assertEqual(entropy('aaaa'), 0)
        self.assertAlmostEqual(entropy('aabb'), 1)
    def test_ips_ports_and_userinfo(self):
        f, m = extract_features('http://user:pw@192.0.2.1:8080/login')
        self.assertEqual(f['hostname_is_ip'], 1)
        self.assertEqual(f['has_explicit_port'], 1)
        self.assertEqual(f['has_userinfo'], 1)
        self.assertEqual(m['domain_group'], '192.0.2.1')
        _, m = extract_features('http://[2001:db8::1]/')
        self.assertEqual(m['domain_group'], '2001:db8::1')
    def test_canonicalization(self):
        _, a = extract_features('HTTP://EXAMPLE.COM:80')
        _, b = extract_features('http://example.com/')
        self.assertEqual(a['canonical_url'], b['canonical_url'])
        _, c = extract_features('http://example.com/A?x=1&y=2')
        _, d = extract_features('http://example.com/a?y=2&x=1')
        self.assertNotEqual(c['canonical_url'], d['canonical_url'])
    def test_missing_scheme_and_private_hosts(self):
        f, m = extract_features('news.example.com/path')
        self.assertEqual(f['is_https'], 0)
        self.assertEqual(f['url_length'], len('news.example.com/path'))
        self.assertEqual(m['domain_group'], 'example.com')
        _, a = extract_features('https://alice.blogspot.com/a')
        _, b = extract_features('https://bob.blogspot.com/b')
        self.assertEqual(a['domain_group'], b['domain_group'])
    def test_malformed_urls_rejected(self):
        for url in ['', 'https://', 'javascript:alert(1)', 'https://bad host.com',
                    'https://example.com:99999', 'https://-bad.com/', 'https://example.invalidtld/']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                extract_features(url)
    def test_offline_features(self):
        with patch('socket.create_connection', side_effect=AssertionError('Network access attempted')):
            _, m = extract_features('https://news.example.co.uk/')
            self.assertEqual(m['domain_group'], 'example.co.uk')

class DataTests(unittest.TestCase):
    def test_cleaning_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'url': ['HTTP://EXAMPLE.COM:80', 'http://example.com/',
                'https://conflict.org/', 'https://conflict.org/', '',
                'https://bad host.com/', 'https://missinglabel.org/', 'https://good.org/', 'http://bad.net/login'],
                'label': [0, 0, 0, 1, 0, 1, '', 0, 1]}).to_csv(root / 'source.csv', index=False)
            frame, summary = prepare([root / 'source.csv'], root / 'output', config())
            self.assertEqual(summary['input_rows'], 9)
            self.assertEqual(summary['retained_rows'], 3)
            self.assertEqual(summary['duplicate_rows_removed'], 1)
            self.assertEqual(summary['conflicting_rows_removed'], 2)
            self.assertEqual(summary['excluded_rows'], 6)
            self.assertEqual(summary['retained_rows'] + summary['excluded_rows'], summary['input_rows'])
            self.assertEqual(frame['label'].nunique(), 2)
            samples = pd.read_csv(root / 'output' / 'feature_samples.csv')
            self.assertTrue(samples['url_defanged'].str.contains(r'\[\.\]').all())
    def test_unexpected_labels_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'url': ['https://example.org/'], 'label': ['phishing']}).to_csv(root / 'source.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Unexpected labels'):
                prepare([root / 'source.csv'], root / 'out', config())
    def test_domain_disjoint_split_is_repeatable(self):
        records = []
        for i in range(60):
            for sub in ['www', 'login']:
                features, metadata = extract_features(f'https://{sub}.domain{i}.com/path/{i}')
                records.append({'url': f'https://{sub}.domain{i}.com/path/{i}',
                    'source_file': 'fixture.csv', 'source_row': len(records) + 2,
                    'label': i % 2, **metadata, **features})
        frame = pd.DataFrame(records)
        with tempfile.TemporaryDirectory() as directory:
            a, b, summary = create_split(frame, config(), directory)
            a2, b2, _ = create_split(frame, config(), directory)
            np.testing.assert_array_equal(a, a2)
            np.testing.assert_array_equal(b, b2)
            self.assertEqual(summary['domain_overlap'], 0)
            self.assertEqual(summary['canonical_url_overlap'], 0)
            self.assertEqual(len(a) + len(b), len(frame))
            self.assertEqual(frame.iloc[a]['label'].nunique(), 2)
            self.assertEqual(frame.iloc[b]['label'].nunique(), 2)

class EvaluationTests(unittest.TestCase):
    def test_metrics_from_known_confusion_matrix(self):
        m = calculate_metrics([0, 0, 0, 1, 1, 1], [.1, .2, .8, .3, .7, .9], .5)
        self.assertEqual([m[x] for x in ['tn', 'fp', 'fn', 'tp']], [2, 1, 1, 2])
        self.assertAlmostEqual(m['false_positive_rate'], 1 / 3)
        self.assertAlmostEqual(m['precision'], 2 / 3)
        self.assertAlmostEqual(m['recall'], 2 / 3)
        self.assertAlmostEqual(m['f1'], 2 / 3)
    def test_end_to_end_and_train_only_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'url': f'https://site{i}.org/' + ('account/verify/1000' if i % 2 else 'about'), 'label': i % 2}
                    for i in range(60)]
            pd.DataFrame(rows).to_csv(root / 'fixture.csv', index=False)
            cfg = config()
            cfg['random_forest']['n_estimators'] = 5
            frame, profile = prepare([root / 'fixture.csv'], root / 'out', cfg)
            train, test, split = create_split(frame, cfg, root / 'out')
            results = train_evaluate(frame, train, test, cfg, root / 'out')
            write_summary(root / 'out', profile, split, cfg, results)
            import joblib
            model = joblib.load(root / 'out' / 'logistic_regression.joblib')
            self.assertEqual(model.named_steps['scale'].n_samples_seen_, len(train))
            np.testing.assert_allclose(model.named_steps['scale'].mean_, frame.iloc[train][FEATURES].mean().to_numpy())
            self.assertEqual(len(results), 3)
            for name in ['metrics.csv', 'SUMMARY.md', 'roc_curves.png', 'precision_recall_curves.png',
                         'logistic_regression_confusion_matrix.png', 'random_forest_confusion_matrix.png']:
                self.assertTrue((root / 'out' / name).is_file(), name)

if __name__ == '__main__':
    unittest.main()
