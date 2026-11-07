"""Run this script in BITS Lab as well as locally; never fit on uploaded CSVs."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_core import ROOT, TARGET, MODEL_FILES, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['local', 'bits-lab'], default='local',
                        help='Execution label only; screenshot is still required for lab proof.')
    args = parser.parse_args()
    data = load_breast_cancer(as_frame=True)
    X = data.data
    # sklearn uses 0=malignant, 1=benign. Reverse this deliberately and document it.
    y = (1 - data.target).rename(TARGET)
    assert X.shape == (569, 30) and not X.isna().any().any()
    train_idx, test_idx = train_test_split(X.index, test_size=0.2, stratify=y, random_state=247)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    assert set(train_idx).isdisjoint(test_idx)
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=247)),
        'Decision Tree': make_pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=247)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        'Naive Bayes': make_pipeline(StandardScaler(), GaussianNB()),
        'Support Vector Machine': make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale')),
    }
    for directory in ['model', 'results', 'data']:
        (ROOT / directory).mkdir(exist_ok=True)
    pd.concat([X, y], axis=1).to_csv(ROOT / 'data/dataset.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / 'test_data.csv', index=False)
    rows, details = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, matrix, report, _ = evaluate(model, X_test, y_test)
        rows.append({'ML Model Name': name, **metrics})
        details[name] = {'metrics': metrics, 'confusion_matrix': matrix.tolist(),
                         'classification_report': report,
                         'training_accuracy': float(model.score(X_train, y_train)),
                         'configuration': str(model)}
        joblib.dump(model, ROOT / 'model' / MODEL_FILES[name])
        print(f'{name}: {metrics}', flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ROOT / 'results/metrics.csv', index=False)
    manifest = {
        'dataset': 'UCI Breast Cancer Wisconsin (Diagnostic)',
        'source': 'https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic',
        'features': list(X.columns), 'target': TARGET,
        'target_mapping': {'0': 'benign', '1': 'malignant'},
        'rows': len(X), 'train_rows': len(train_idx), 'test_rows': len(test_idx),
        'class_counts': {str(k): int(v) for k, v in y.value_counts().items()},
        'train_indices': train_idx.tolist(), 'test_indices': test_idx.tolist(),
        'random_seed': 247, 'test_fraction': 0.2,
        'metric_definition': 'Binary precision/recall/F1 with malignant=1; ROC-AUC from probabilities or SVM decision scores; zero_division=0.',
        'environment_label': args.environment,
        'platform': platform.platform(), 'python': sys.version,
        'sklearn': sklearn.__version__, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'test_csv_sha256': hashlib.sha256((ROOT / 'test_data.csv').read_bytes()).hexdigest(),
    }
    (ROOT / 'model/manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (ROOT / 'results/details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(f'Finished: {len(X_train)} training rows / {len(X_test)} test rows. Environment label: {args.environment}')


if __name__ == '__main__':
    main()

