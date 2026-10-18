# explainable-academic-risk-prediction-oulad
# Explainable Machine Learning for Early Prediction of Academic Risk

### An OULAD-Based Decision-Support System

**MSc Computer Science | Coventry University | 7005SCN Individual Research Project**

This repository contains an explainable machine-learning research prototype designed to identify students at risk of academic failure or withdrawal using early educational data from the Open University Learning Analytics Dataset (OULAD).

The project combines predictive modelling, feature engineering, group-aware validation, SHAP explainability and a Streamlit application to investigate how academic-risk predictions can support responsible educational decision-making.

## Project Overview

Early identification of academic risk can help educational institutions recognise students who may benefit from timely academic support.

This project investigates whether machine-learning models can predict adverse academic outcomes using student characteristics, assessment records and virtual learning environment (VLE) engagement during the first 56 days of a module.

Three classification algorithms are evaluated:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost was selected as the preferred research model based on its recorded predictive performance. SHAP explanations are used to examine influential features and individual predictions.

The developed system is intended for human-reviewed academic decision support, not automatic academic decisions.

## Research Question

**How effectively can explainable machine-learning models predict students at risk of academic underperformance using educational data?**

## Dataset

The project uses the Open University Learning Analytics Dataset (OULAD), containing anonymised educational records from multiple module presentations.

Seven source tables are integrated: `courses`, `assessments`, `studentInfo`, `studentRegistration`, `studentAssessment`, `vle` and `studentVle`.

Students whose final recorded outcome was **Fail** or **Withdrawn** are classified as At Risk. Students who achieved **Pass** or **Distinction** are classified as Not At Risk.

Students who had already withdrawn on or before Day 56 are excluded.

| Dataset property | Value |
|---|---:|
| Original student-module observations | 32,593 |
| Early withdrawals excluded | 6,071 |
| Eligible observations | 26,522 |
| At-risk observations | 11,137 |
| Not-at-risk observations | 15,385 |
| Original predictors | 56 |
| Transformed features | 94 |

OULAD publication: https://doi.org/10.1038/sdata.2017.171

## Machine-Learning Methodology

The workflow consists of data integration, Day-56 temporal filtering, feature engineering, preprocessing, grouped validation, model comparison, independent threshold selection and final evaluation.

Numerical features are processed using median imputation and standardisation. Categorical features are processed using most-frequent imputation and one-hot encoding.

Student-grouped validation is used to prevent the same student identifier from appearing across development and evaluation partitions.

| Data partition | Observations |
|---|---:|
| Model development | 16,992 |
| Threshold validation | 4,233 |
| Independent hold-out test | 5,297 |
| Final model training | 21,225 |

The operating classification threshold was selected on separate validation data using the F2 score. The hold-out dataset was reserved for final evaluation.

## Model Performance

The following results were recorded on the independent hold-out dataset at the standard classification threshold of 0.50.

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 0.7544 | 0.7584 | 0.7627 |
| Precision | 0.7602 | 0.7728 | 0.7727 |
| Recall | 0.6207 | 0.6149 | 0.6295 |
| F1-score | 0.6834 | 0.6849 | 0.6938 |
| ROC-AUC | 0.8256 | 0.8238 | 0.8292 |
| PR-AUC | 0.8026 | 0.8019 | 0.8087 |

XGBoost achieved the highest recorded ROC-AUC and PR-AUC. However, its advantage over the alternative classifiers was relatively small.

### Selected Operating Threshold

The preferred operating threshold was **0.19**, selected to prioritise the identification of at-risk observations.

| Metric | XGBoost at 0.19 |
|---|---:|
| Accuracy | 0.6458 |
| Precision | 0.5510 |
| Recall | 0.9213 |
| F1-score | 0.6896 |
| F2-score | 0.8122 |
| ROC-AUC | 0.8292 |
| PR-AUC | 0.8087 |

The threshold identified 2,084 true-positive observations but also produced 1,698 false-positive alerts.

This demonstrates the practical trade-off between detecting more at-risk students and generating additional staff intervention demands.

## Explainable Artificial Intelligence

SHAP (SHapley Additive exPlanations) is used to investigate the preferred model's predictions.

The explainability analysis includes global feature importance, a SHAP summary plot and local explanations of true-positive, true-negative, false-positive and false-negative classifications.

The most influential features include assessment submission rate, last VLE activity day, weighted assessment mean score, recent VLE clicks and assessment mean score.

SHAP values describe how features contribute to the fitted model output. They do not establish causal explanations of student performance or withdrawal.

## Streamlit Application

The research prototype includes five application sections.

| Section | Function |
|---|---|
| Prediction | Estimates academic risk from input features |
| Performance | Presents model evaluation results |
| Global Explainability | Displays influential model features |
| Subgroups | Presents descriptive subgroup performance |
| Responsible Use | Explains the model's limitations and safeguards |

The Prediction section supports a complete CSV input record and a manual demonstration mode.

A complete CSV record must contain the expected 56 predictors. Manual demonstration mode exposes selected variables and fills remaining inputs using recorded training-derived defaults.

## Project Files

The following artefacts are documented in the dissertation. Some locally downloaded files have a `(2)` suffix; the names in GitHub may differ.

| Project file | Purpose |
|---|---|
| `app.py` | Streamlit application entry point |
| `app_utils.py` | Input validation and supporting functions |
| `best_model_bundle.joblib` | Saved fitted model and preprocessing metadata |
| `requirements.txt` | Python dependencies |
| `run_configuration.json` | Model configuration and operating threshold |
| `sample_input.csv` | Example complete input record |
| `feature_dictionary.csv` | Feature names and types |
| `artefact_test_results.csv` | Automated testing evidence |
| `manual_streamlit_test_plan.csv` | Manual testing plan |
| `error_analysis.csv` | Prediction-error analysis |

These are descriptive filenames. Use the actual filenames in the repository when running the application.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/gonelasriya2003/explainable-academic-risk-prediction-oulad.git
cd explainable-academic-risk-prediction-oulad
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Required Packages

If the dependency file is named `requirements.txt`, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The research environment records pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.6.1, XGBoost 3.4.1 and SHAP 0.52.0, alongside Streamlit and joblib.

### 4. Run the Streamlit Application

If the entry-point file is named `app.py` and is located in the repository root, run:

```bash
python -m streamlit run app.py
```

If the uploaded application retains the original name `app(2).py`, use:

```bash
python -m streamlit run "app(2).py"
```

Ensure the application imports, saved-model filenames and related file paths match the files in the repository. A utility module imported as `app_utils` must have a matching Python module filename.

The browser should open the Streamlit interface after the application starts successfully.

## Example Input

Use the supplied sample CSV to test the Prediction section.

The input must contain exactly one observation with the required 56 predictor columns. Its field names must match the feature dictionary and model's expected input schema.

Input validation checks file structure and selected logical relationships between fields.

The model output includes an estimated probability, an At Risk or Not At Risk classification, and explanatory feature contributions.

## Testing Status

The project records **16 passing automated checks**, including:

- Student-group separation
- Exclusion of final outcomes from predictors
- Saved-model loading consistency
- Handling missing and unseen values
- SHAP feature compatibility
- Python source compilation
- CSV input validation

Six additional Streamlit browser-level acceptance tests were documented but remained marked **TO TEST** in the available research evidence.

These cover valid input uploads, invalid-file rejection, logical input constraints and successful rendering of predictions and explanations.

The recorded automated results should not be interpreted as confirmation of complete end-to-end application testing.

## Limitations

This is a retrospective educational research prototype and has not been externally validated at another institution.

OULAD does not provide independently verified release timestamps for every assessment mark, leaving uncertainty about the availability of some grades at Day 56.

The validation design prevents repeated-student overlap but does not establish generalisation to completely unseen module presentations.

The selected threshold creates a substantial false-positive alert burden. Probability calibration and subgroup differences also require further investigation.

The prototype does not demonstrate that prediction-informed interventions improve retention or academic outcomes.

## Responsible Use

This system is intended for academic research and supportive educational decision-making.

Predictions should not be used as definitive judgements about student ability, effort, motivation or future performance.

Any real institutional deployment would require appropriate privacy, data-protection, research-ethics and governance safeguards.

Human review is essential. Risk predictions should inform supportive conversations rather than automatically determine academic sanctions or access to opportunities.

## Future Work

Future development should prioritise verifying the availability of assessment information at Day 56, testing unseen course presentations, completing Streamlit acceptance tests, evaluating calibration and subgroup uncertainty, and examining thresholds under real institutional support constraints.

Further research could also distinguish failure from withdrawal and investigate the educational effects of prediction-informed support.

## Research Reference

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, Article 170171.

https://doi.org/10.1038/sdata.2017.171

## Author

**Sriya Gonela**  
MSc Computer Science  
Coventry University  
Student ID: 16851544

**GitHub:** https://github.com/gonelasriya2003

---

**Project status:** Explainable academic-risk prediction research prototype. Not validated for autonomous or production institutional decision-making.
