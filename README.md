# CellScope: Machine Learning Classification Comparison

**Course:** Machine Learning (AIMLZG565) | **Assignment:** 2  
**Student:** Satyajit Roy | **BITS ID:** 2026AJ05247

## a. Problem statement

Classify cell samples as benign or malignant using numerical measurements of cell nuclei from the Wisconsin Diagnostic Breast Cancer dataset. Compare Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine on the same held-out test set using six evaluation metrics. An interactive Streamlit application evaluates uploaded labelled test data using saved models without retraining.

This project is an educational classification experiment, not a clinical diagnostic system.

## b. Dataset description

| Property | Description |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository |
| Total instances | **569** |
| Input features | **30 numerical features**, excluding the target and identifiers |
| Classification task | Binary classification: benign or malignant |
| Target column | `diagnosis`: **0 = benign; 1 = malignant** |
| Class distribution | 357 benign and 212 malignant instances |
| Missing values | None |
| Training set | 455 instances: 285 benign and 170 malignant |
| Test set | 114 instances: 72 benign and 42 malignant |
| Split | Stratified 80:20 split with `random_state=247` |

The dataset satisfies the assignment minimum of **500 instances and 12 input features**. The 114-row test CSV is the held-out evaluation subset, not the complete dataset.

The features describe ten cell-nucleus characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each characteristic has mean, standard-error and worst-value measurements, giving 30 features.

- **Source:** [UCI dataset page](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic).
- **Access:** The project loads scikit-learn's bundled copy of the UCI dataset. All 569 rows and 30 features are retained without resampling or feature selection.
- **Label encoding:** scikit-learn's original labels are reversed so malignant is the positive class (`1`).
- **Attribution:** Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). License: CC BY 4.0.

## c. GitHub Repository Link

[https://github.com/2026aj05247-BITS/ml_assignment_2_S1_26_AIMLZG565](https://github.com/2026aj05247-BITS/ml_assignment_2_S1_26_AIMLZG565)

## d. Models used

### Model configuration and preprocessing

| ML Model Name | Configuration | Preprocessing |
|---|---|---|
| Logistic Regression | `C=1`, `max_iter=3000`, `random_state=247` | StandardScaler |
| Decision Tree | `max_depth=5`, `min_samples_leaf=3`, `random_state=247` | No scaling |
| KNN | 7 neighbours, uniform weights, Euclidean distance | StandardScaler |
| Naive Bayes | GaussianNB, default variance smoothing | StandardScaler |
| Support Vector Machine | RBF kernel, `C=1`, `gamma='scale'` | StandardScaler |

All models use the same training and test split. Scaling is fitted only on the training set and stored with the classifier in a saved pipeline. The test set is not used for fitting preprocessing, training or hyperparameter tuning. These configurations are fixed baselines; no hyperparameter search was performed.

### Evaluation metrics

Precision, Recall and F1 use binary averaging with **malignant = 1** as the positive class. Accuracy is the fraction of correctly classified records. F1 balances precision and recall; MCC uses all four confusion-matrix counts and ranges from -1 to 1.

AUC denotes **ROC-AUC**, calculated from positive-class probabilities for Logistic Regression, Decision Tree, KNN and Gaussian Naive Bayes, and from decision-function scores for SVM. Metrics are shown to four decimal places; full-precision results are available in `results/metrics.csv`. Undefined precision/recall/F1 divisions use zero. For uploads containing only one class, ROC-AUC is undefined and shown as N/A.

### Model comparison on the held-out test set

<!-- RESULTS_START -->

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9737 | 0.9904 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.9211 | 0.9137 | 0.8511 | 0.9524 | 0.8989 | 0.8380 |
| KNN | 0.9649 | 0.9983 | 0.9318 | 0.9762 | 0.9535 | 0.9260 |
| Naive Bayes | 0.9474 | 0.9921 | 0.8750 | 1.0000 | 0.9333 | 0.8956 |
| Support Vector Machine | 0.9912 | 1.0000 | 0.9767 | 1.0000 | 0.9882 | 0.9814 |

<!-- RESULTS_END -->

### Observations about model performance

A false negative means a malignant sample was predicted as benign; a false positive means a benign sample was predicted as malignant.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Correctly classified 111 of 114 samples, with 2 false negatives and 1 false positive. Precision of 0.9756 and F1 of 0.9639 make it a strong linear baseline, with the second-highest F1 in this comparison. Compared with KNN, it produced fewer false positives but missed one additional malignant sample. Its strong results suggest that a linear decision boundary captures much of the separation on this split. |
| Decision Tree | Correctly classified 105 samples, with 2 false negatives and 7 false positives. It matched Logistic Regression's recall of 0.9524 but had lower precision (0.8511), F1 (0.8989) and AUC (0.9137). Training accuracy was 0.9890 versus test accuracy of 0.9211, suggesting weaker generalization on this split despite depth and leaf-size restrictions. This gap alone does not establish performance on other splits. |
| KNN | Correctly classified 110 samples, with 1 false negative and 3 false positives. Recall of 0.9762 exceeded Logistic Regression's recall, while precision fell to 0.9318. AUC of 0.9983 indicates strong score ranking even though the selected neighbour-voting rule still produced four classification errors. Scaling ensures that feature units do not dominate distance calculations. |
| Naive Bayes | Identified all 42 malignant samples, achieving recall of 1.0000, but classified 6 benign samples as malignant. These false positives reduced precision to 0.8750 and F1 to 0.9333. Gaussian Naive Bayes therefore favoured sensitivity over precision in this experiment. Its conditional-independence and Gaussian assumptions may limit performance; this experiment does not isolate their individual effects. |
| Support Vector Machine | Correctly classified 113 samples, with no false negatives and 1 false positive. It achieved the highest accuracy (0.9912), F1 (0.9882) and MCC (0.9814), while matching Naive Bayes' recall with fewer false positives. AUC of 1.0000 means perfect ranking on these test rows, not perfect classification at the default decision threshold: one false positive remains. |
| Overall, winner for this dataset? | **Support Vector Machine**, using highest malignant-class F1 as the primary comparison criterion, then recall and AUC as tie-breakers. It detected all malignant test samples with only one false positive. This conclusion applies to the fixed 114-row test set; it does not establish statistical superiority, clinical suitability or the same performance on new populations. |

## Streamlit application

The application includes:

- CSV upload for labelled test data and a download of the supplied held-out test CSV.
- A dropdown containing all five classifiers.
- All six evaluation metrics for the selected model and a comparison table for all five models.
- A labelled confusion matrix and classification report for the selected model.
- Downloads of comparison results and predictions.

On startup, the app evaluates the bundled `test_data.csv`. Uploading that same file should reproduce the same metrics. Uploading another valid test file recalculates results on its rows. **No uploaded data is used for training.**

The CSV must contain all 30 feature columns and the actual `diagnosis` target. Feature values must be numeric, finite and non-missing; target values must be 0 or 1. Missing, extra or duplicate columns are rejected. Column order is corrected automatically. The application accepts up to 10,000 rows and 5 MB. Use held-out data for meaningful evaluation.

**Live Streamlit application:** Deployment link will be added after deployment and verification.

## Reproduce and run

Use Python 3.12 and the versions in `requirements.txt`. Clone the repository and open its directory:

```bash
 git clone https://github.com/2026aj05247-BITS/ml_assignment_2_S1_26_AIMLZG565.git
 cd ml_assignment_2_S1_26_AIMLZG565
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_project.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Linux / BITS Virtual Lab

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --environment bits-lab
python update_results.py
python verify_project.py
python -m streamlit run app.py --server.address 0.0.0.0
```

The app can use the committed model files without training again. To regenerate the local training artifacts, run `python train.py --environment local` with the project environment, followed by `python update_results.py`. After regenerating results, check that the observations still match the measured values. Automated checks cover known metric calculations, CSV validation, disjoint train/test indices, saved-model results and all five Streamlit model selections.

Current reported artifacts were generated locally. BITS Virtual Lab execution and its required screenshot remain to be completed.

## Repository contents

| File or directory | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `ml_core.py` | Shared CSV validation and metric calculation |
| `train.py` | Dataset preparation, model training and evaluation |
| `requirements.txt` | Package dependencies |
| `README.md` | Problem, dataset, method, results and instructions |
| `test_data.csv` | Labelled held-out test set |
| `model/` | All five saved model pipelines and execution manifest |
| `results/` | Full-precision metrics, confusion matrices and classification reports |
| `data/dataset.csv` | Full dataset with the documented label encoding |
| `update_results.py` | Refresh the README metric table from measured results |
| `verify_project.py` | Automated validation checks |
