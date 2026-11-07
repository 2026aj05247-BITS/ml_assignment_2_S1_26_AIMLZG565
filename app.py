"""Streamlit evaluation interface. Loads trusted repository artifacts only."""
import json

import joblib
import pandas as pd
import streamlit as st

from ml_core import ROOT, MODEL_FILES, METRICS, evaluate, read_csv_bytes, validate_test_data

st.set_page_config(page_title='CellScope | Classifier comparison', page_icon='🔬', layout='wide')
st.title('CellScope · Classifier comparison')
st.caption('ML Assignment 2 · Wisconsin Diagnostic Breast Cancer · Educational experiment, not a diagnostic tool')


@st.cache_resource
def load_models(signature):
    return {name: joblib.load(ROOT / 'model' / filename) for name, filename in MODEL_FILES.items()}


try:
    manifest = json.loads((ROOT / 'model/manifest.json').read_text(encoding='utf-8'))
    signature = tuple((ROOT / 'model' / name).stat().st_mtime_ns for name in MODEL_FILES.values())
    models = load_models(signature)
except (OSError, ValueError) as exc:
    st.error(f'Model artifacts are unavailable. Run python train.py first. Details: {exc}')
    st.stop()

with st.sidebar:
    st.header('Test your classifiers')
    selection = st.selectbox('Select model', list(MODEL_FILES))
    st.download_button('Download held-out test CSV', (ROOT / 'test_data.csv').read_bytes(),
                       file_name='test_data.csv', mime='text/csv')
    upload = st.file_uploader('Upload labelled test data (CSV)', type=['csv'])
    st.caption('30 numeric features + diagnosis. 0 = benign; 1 = malignant. Upload test data only.')
    st.info('Saved models are used for evaluation. Uploaded data never trains or changes a model.')

if upload is not None and upload.size > 5 * 1024 * 1024:
    st.error('The CSV must be smaller than 5 MB.')
    st.stop()
try:
    content = upload.getvalue() if upload is not None else (ROOT / 'test_data.csv').read_bytes()
    frame = read_csv_bytes(content)
    X, y = validate_test_data(frame, manifest['features'])
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

st.success(f'Evaluating {len(X)} rows · ' + ('uploaded CSV' if upload is not None else 'bundled held-out test CSV'))
if y.nunique() < 2:
    st.warning('This CSV contains one class only: ROC-AUC is undefined and displayed as N/A. Other metrics may be uninformative; zero-division values use 0.')
st.caption('Positive class: malignant (1). Precision, recall and F1 use binary averaging. AUC is ROC-AUC.')
evaluations = {name: evaluate(model, X, y) for name, model in models.items()}
metrics, matrix, report, predictions = evaluations[selection]
for column, metric in zip(st.columns(6), METRICS):
    column.metric(metric, 'N/A' if metrics[metric] is None else f'{metrics[metric]:.4f}')

comparison_tab, errors_tab, data_tab = st.tabs(['Compare all five', 'Selected model details', 'Dataset & method'])
with comparison_tab:
    st.subheader('Same test rows, five classifiers')
    comparison = pd.DataFrame([{'ML Model Name': name, **result[0]} for name, result in evaluations.items()])
    st.dataframe(comparison.style.format({metric: '{:.4f}' for metric in METRICS}, na_rep='N/A'), hide_index=True, width='stretch')
    st.download_button('Download comparison CSV', comparison.to_csv(index=False), 'comparison.csv', 'text/csv')
    st.caption('Compare the trade-offs; this uploaded-data ranking does not tune or select training parameters.')
with errors_tab:
    st.subheader(selection)
    st.write('Confusion matrix — rows: actual class; columns: predicted class.')
    st.dataframe(pd.DataFrame(matrix, index=['Actual benign (0)', 'Actual malignant (1)'],
                              columns=['Predicted benign (0)', 'Predicted malignant (1)']), width='stretch')
    st.write('Classification report')
    st.dataframe(pd.DataFrame(report).transpose(), width='stretch')
    output = frame.copy()
    output['predicted_diagnosis'] = predictions
    st.download_button('Download predictions', output.to_csv(index=False), 'predictions.csv', 'text/csv')
with data_tab:
    st.write('569 observations, 30 real-valued input features; target excluded from the feature count.')
    st.markdown('[Original UCI dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)')
    st.write('Fixed stratified 80/20 split, seed 247. Scaling is fitted only on training data and saved inside each applicable pipeline. No test-set tuning is performed.')
    st.write('The scikit-learn source labels are reversed deliberately: this project uses malignant=1 and benign=0.')
    st.caption(f"Artifacts generated in: {manifest['environment_label']} · {manifest['timestamp_utc']}")
    st.dataframe(frame.head(15), hide_index=True, width='stretch')
    with st.expander('Required input columns'):
        st.write(manifest['features'] + [manifest['target']])

