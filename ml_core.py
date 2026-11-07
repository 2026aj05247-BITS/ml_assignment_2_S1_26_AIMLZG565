"""Shared validation and evaluation: no training is performed by the web app."""
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
TARGET = 'diagnosis'
METRICS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib',
}


def read_csv_bytes(content):
    """Check duplicate headers before pandas can silently rename them."""
    try:
        text = content.decode('utf-8-sig')
        header = next(csv.reader(io.StringIO(text)))
        if len(header) != len(set(header)):
            raise ValueError('Duplicate column names are not allowed.')
        return pd.read_csv(io.StringIO(text))
    except (UnicodeError, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('Upload a non-empty UTF-8 CSV with a header row.') from exc


def validate_test_data(frame, features):
    expected = set(features) | {TARGET}
    missing, extra = expected - set(frame.columns), set(frame.columns) - expected
    if missing or extra:
        raise ValueError(f'Column mismatch. Missing: {sorted(missing)}. Unexpected: {sorted(extra)}.')
    if not 1 <= len(frame) <= 10000:
        raise ValueError('Upload between 1 and 10,000 test rows.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate columns are not allowed.')
    try:
        X = frame.loc[:, features].apply(pd.to_numeric, errors='raise')
        y = pd.to_numeric(frame[TARGET], errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('All features and diagnosis must be numeric. diagnosis must be 0 or 1.') from exc
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Features must not contain missing or infinite values.')
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError('diagnosis must contain only 0 (benign) or 1 (malignant).')
    return X, y.astype(int)


def evaluate(model, X, y):
    prediction = model.predict(X)
    # Positive label is explicitly 1 = malignant, including the probability column.
    if hasattr(model, 'predict_proba'):
        positive_index = list(model.classes_).index(1)
        scores = model.predict_proba(X)[:, positive_index]
    else:
        scores = model.decision_function(X)
    both_classes = len(np.unique(y)) == 2
    metrics = {
        'Accuracy': float(accuracy_score(y, prediction)),
        'AUC': float(roc_auc_score(y, scores)) if both_classes else None,
        'Precision': float(precision_score(y, prediction, zero_division=0)),
        'Recall': float(recall_score(y, prediction, zero_division=0)),
        'F1': float(f1_score(y, prediction, zero_division=0)),
        'MCC': float(matthews_corrcoef(y, prediction)),
    }
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    report = classification_report(y, prediction, labels=[0, 1],
                                   target_names=['Benign (0)', 'Malignant (1)'],
                                   output_dict=True, zero_division=0)
    return metrics, matrix, report, prediction

