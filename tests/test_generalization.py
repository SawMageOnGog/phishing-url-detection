import unittest
from generalize import label_to_project

class GeneralizationTests(unittest.TestCase):
    def test_official_label_mapping(self):
        self.assertEqual(label_to_project('0'),1)
        self.assertEqual(label_to_project('1'),0)
    def test_invalid_labels_fail(self):
        for x in ('','phishing','2','1.0'):
            with self.assertRaises(ValueError): label_to_project(x)
if __name__=='__main__': unittest.main()
