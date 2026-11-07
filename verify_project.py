"""Meaningful checks after training: run python verify_project.py."""
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from ml_core import ROOT, MODEL_FILES, evaluate, read_csv_bytes, validate_test_data


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        class FixedModel:
            classes_ = np.array([0, 1])
            def predict(self, X):
                return np.array([0, 1, 0, 1])
            def predict_proba(self, X):
                return np.array([[.9, .1], [.2, .8], [.6, .4], [.1, .9]])
        metrics, matrix, _, _ = evaluate(FixedModel(), None, np.array([0, 0, 1, 1]))
        self.assertEqual(matrix.tolist(), [[1, 1], [1, 1]])
        for key in ['Accuracy', 'Precision', 'Recall', 'F1']:
            self.assertEqual(metrics[key], .5)
        self.assertEqual(metrics['MCC'], 0)
        self.assertEqual(metrics['AUC'], .75)

    def test_duplicate_csv_headers(self):
        with self.assertRaises(ValueError):
            read_csv_bytes(b'a,a,diagnosis\n1,2,0\n')

    def test_invalid_and_reordered_data(self):
        frame = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'diagnosis': [0, 1]})
        X, _ = validate_test_data(frame[['diagnosis', 'b', 'a']], ['a', 'b'])
        self.assertEqual(list(X.columns), ['a', 'b'])
        for broken in [frame.drop(columns='a'), frame.assign(a=np.nan),
                       frame.assign(diagnosis=2), frame.assign(extra=0), frame.iloc[:0]]:
            with self.assertRaises(ValueError):
                validate_test_data(broken, ['a', 'b'])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'model/manifest.json').read_text(encoding='utf-8'))
        cls.frame = pd.read_csv(ROOT / 'test_data.csv')

    def test_disjoint_split_and_reloaded_metrics(self):
        self.assertFalse(set(self.manifest['train_indices']) & set(self.manifest['test_indices']))
        self.assertEqual(self.manifest['rows'], 569)
        self.assertEqual(len(self.manifest['features']), 30)
        X, y = validate_test_data(self.frame, self.manifest['features'])
        saved = pd.read_csv(ROOT / 'results/metrics.csv').set_index('ML Model Name')
        for name, filename in MODEL_FILES.items():
            model = joblib.load(ROOT / 'model' / filename)
            actual, _, _, _ = evaluate(model, X, y)
            for metric, value in actual.items():
                self.assertAlmostEqual(value, saved.loc[name, metric], places=12)
            subset = y == 1
            self.assertIsNone(evaluate(model, X.loc[subset], y.loc[subset])[0]['AUC'])

    def test_streamlit_all_models(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        for name in MODEL_FILES:
            app.selectbox[0].select(name).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)

