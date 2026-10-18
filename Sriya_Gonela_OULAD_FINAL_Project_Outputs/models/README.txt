
EXPLAINABLE ACADEMIC RISK PROTOTYPE

Preferred model:
XGBoost

Prediction point:
Day 56

Operating threshold:
0.190

Model selection:
Five-fold grouped cross-validation on the
model-development data.

Primary model-selection metric:
PR-AUC, followed by F1 and Recall.

Threshold selection:
Separate grouped threshold-validation set
using F2.

Final evaluation:
Independent grouped hold-out test set.

Students withdrawn on or before
Day 56 are excluded.

date_unregistration is not a predictive feature.

Run locally:

pip install -r requirements.txt

streamlit run app.py

Use sample_input.csv to test complete CSV upload.

Manual mode is a demonstration interface because
features not shown remain at training defaults.

The displayed probability is a model estimate,
not a guaranteed real-world probability.

This prototype is intended for decision support only.
