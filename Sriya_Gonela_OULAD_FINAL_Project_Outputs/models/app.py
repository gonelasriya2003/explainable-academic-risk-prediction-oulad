
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from app_utils import (
    validate_feature_frame,
    update_manual_features
)


BASE_DIR = Path(__file__).resolve().parent


bundle = joblib.load(
    BASE_DIR / "best_model_bundle.joblib"
)


model = bundle["pipeline"]

features = bundle["feature_columns"]

numeric = bundle["numeric_columns"]

categorical = bundle["categorical_columns"]

numeric_defaults = bundle["numeric_defaults"]

numeric_ranges = bundle["numeric_ranges"]

categorical_defaults = bundle["categorical_defaults"]

categorical_options = bundle["categorical_options"]

feature_map = np.array(
    bundle["feature_map"]
)

threshold = float(
    bundle["operating_threshold"]
)


st.set_page_config(
    page_title="Academic Risk Decision Support",
    layout="wide"
)


st.title(
    "Explainable Academic Risk Decision-Support Prototype"
)


st.info(
    f"Model: {bundle['preferred_model_name']} | "
    f"Prediction point: Day {bundle['prediction_day']} | "
    f"Operating threshold: {threshold:.3f}"
)


st.warning(
    "This prototype supports professional judgement. "
    "It must not be used as an automated academic decision system."
)


tabs = st.tabs([
    "Prediction",
    "Performance",
    "Global Explainability",
    "Subgroups",
    "Responsible Use"
])


with tabs[0]:

    st.subheader(
        "Student Risk Prediction"
    )


    input_mode = st.radio(
        "Input method",
        [
            "Upload complete feature CSV",
            "Manual demonstration"
        ]
    )


    prediction_df = None


    if input_mode == "Upload complete feature CSV":

        uploaded = st.file_uploader(
            "Upload one complete student feature row",
            type=["csv"]
        )


        if uploaded is not None:

            uploaded_df = pd.read_csv(
                uploaded
            )


            valid, message = validate_feature_frame(
                uploaded_df,
                features
            )


            if not valid:

                st.error(
                    message
                )

            else:

                prediction_df = (
                    uploaded_df[
                        features
                    ]
                    .copy()
                )


    else:

        st.caption(
            "Manual mode is for demonstration. "
            "Features not shown remain at training-set defaults."
        )


        row = {}


        for col in features:

            if col in numeric:

                row[col] = numeric_defaults.get(
                    col,
                    0.0
                )

            else:

                row[col] = categorical_defaults.get(
                    col,
                    ""
                )


        manual_fields = [
            "code_module",
            "code_presentation",
            "gender",
            "highest_education",
            "age_band",
            "imd_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "assess_due_count",
            "assess_submitted_count",
            "assess_mean_score",
            "assess_weighted_mean_score",
            "vle_total_clicks",
            "vle_active_days",
            "vle_clicks_first14",
            "vle_clicks_days14_27",
            "vle_clicks_last14"
        ]


        manual_fields = [
            field
            for field in manual_fields
            if field in features
        ]


        left, right = st.columns(
            2
        )


        for i, field in enumerate(
            manual_fields
        ):

            area = (
                left
                if i % 2 == 0
                else right
            )


            with area:

                if field in categorical:

                    options = categorical_options[
                        field
                    ]


                    default = categorical_defaults[
                        field
                    ]


                    index = (
                        options.index(
                            default
                        )
                        if default in options
                        else 0
                    )


                    row[
                        field
                    ] = st.selectbox(
                        field,
                        options,
                        index=index
                    )


                else:

                    limits = numeric_ranges.get(
                        field,
                        {
                            "min":
                                numeric_defaults.get(
                                    field,
                                    0.0
                                ),

                            "max":
                                numeric_defaults.get(
                                    field,
                                    0.0
                                )
                        }
                    )


                    minimum = float(
                        limits[
                            "min"
                        ]
                    )


                    maximum = float(
                        limits[
                            "max"
                        ]
                    )


                    default = float(
                        numeric_defaults.get(
                            field,
                            0.0
                        )
                    )


                    if maximum > minimum:

                        default = min(
                            max(
                                default,
                                minimum
                            ),
                            maximum
                        )


                        row[
                            field
                        ] = st.number_input(
                            field,
                            min_value=minimum,
                            max_value=maximum,
                            value=default
                        )


                    else:

                        row[
                            field
                        ] = st.number_input(
                            field,
                            value=default
                        )


        row, manual_errors = update_manual_features(
            row
        )


        if manual_errors:

            for message in manual_errors:

                st.error(
                    message
                )

        else:

            prediction_df = pd.DataFrame(
                [row]
            )[
                features
            ]


    if (
        prediction_df is not None
        and
        st.button(
            "Predict Risk",
            type="primary"
        )
    ):

        probability = (
            model
            .predict_proba(
                prediction_df
            )[0, 1]
        )


        prediction = int(
            probability
            >=
            threshold
        )


        if prediction == 1:

            st.error(
                f"AT RISK — model-estimated risk probability: "
                f"{probability:.1%}"
            )

        else:

            st.success(
                f"NOT AT RISK — model-estimated risk probability: "
                f"{probability:.1%}"
            )


        st.caption(
            "This probability is a model estimate and should "
            "not be treated as a guaranteed real-world probability."
        )


        st.progress(
            float(
                min(
                    max(
                        probability,
                        0
                    ),
                    1
                )
            )
        )


        try:

            prep = model.named_steps[
                "preprocess"
            ]


            estimator = model.named_steps[
                "model"
            ]


            transformed = np.asarray(
                prep.transform(
                    prediction_df
                )
            )


            background = np.asarray(
                bundle[
                    "shap_background"
                ]
            )


            if estimator.__class__.__name__ == "LogisticRegression":

                explainer = shap.LinearExplainer(
                    estimator,
                    background
                )

            else:

                explainer = shap.TreeExplainer(
                    estimator
                )


            output = explainer(
                transformed
            )


            values = np.asarray(
                output.values
            )


            if values.ndim == 3:

                values = values[
                    :,
                    :,
                    1
                ]


            values = values[
                0
            ]


            rows = []


            for feature in list(
                dict.fromkeys(
                    feature_map
                )
            ):

                mask = (
                    feature_map
                    ==
                    feature
                )


                rows.append({
                    "Feature":
                        feature,

                    "Input value":
                        prediction_df.iloc[
                            0
                        ].get(
                            feature,
                            np.nan
                        ),

                    "SHAP contribution":
                        values[
                            mask
                        ].sum()
                })


            explanation_table = pd.DataFrame(
                rows
            )


            explanation_table[
                "Absolute"
            ] = np.abs(
                explanation_table[
                    "SHAP contribution"
                ]
            )


            explanation_table = (
                explanation_table
                .sort_values(
                    "Absolute",
                    ascending=False
                )
                .head(12)
            )


            st.subheader(
                "Individual Explanation"
            )


            st.dataframe(
                explanation_table[
                    [
                        "Feature",
                        "Input value",
                        "SHAP contribution"
                    ]
                ],
                use_container_width=True
            )


            st.bar_chart(
                explanation_table
                .set_index(
                    "Feature"
                )[
                    "SHAP contribution"
                ]
            )


            st.caption(
                "Positive SHAP values push the model towards "
                "At Risk. Negative values push it towards "
                "Not At Risk. SHAP does not establish causality."
            )


        except Exception as error:

            st.warning(
                f"SHAP explanation unavailable: {error}"
            )


with tabs[1]:

    st.subheader(
        "Model Performance"
    )


    st.write(
        "Model-selection cross-validation"
    )


    st.dataframe(
        pd.DataFrame(
            bundle[
                "cv_results"
            ]
        ).round(4),
        use_container_width=True
    )


    st.write(
        "Final hold-out results at threshold 0.50"
    )


    st.dataframe(
        pd.DataFrame(
            bundle[
                "test_results"
            ]
        ).round(4),
        use_container_width=True
    )


    st.write(
        "Preferred model at selected operating threshold"
    )


    st.dataframe(
        pd.DataFrame(
            bundle[
                "operational_result"
            ]
        ).round(4),
        use_container_width=True
    )


with tabs[2]:

    st.subheader(
        "Global Explainability"
    )


    shap_df = pd.DataFrame(
        bundle[
            "global_shap"
        ]
    )


    st.bar_chart(
        shap_df
        .head(15)
        .set_index(
            "feature"
        )[
            "mean_abs_shap"
        ]
    )


    st.dataframe(
        shap_df.head(20),
        use_container_width=True
    )


with tabs[3]:

    st.subheader(
        "Subgroup Performance"
    )


    subgroup_df = pd.DataFrame(
        bundle[
            "subgroup_results"
        ]
    )


    st.dataframe(
        subgroup_df.round(4),
        use_container_width=True
    )


    st.warning(
        "Differences in subgroup metrics do not by themselves "
        "prove fairness or discrimination."
    )


with tabs[4]:

    st.subheader(
        "Responsible Use"
    )


    st.markdown(
        """
        - Predictions are probabilistic and may be incorrect.
        - The operating threshold was selected without using the final test set.
        - False negatives may miss students who could benefit from support.
        - False positives may create unnecessary intervention.
        - Students already withdrawn by Day 56 are excluded.
        - Withdrawal date is not a predictive feature.
        - VLE activity is an observed behaviour, not a direct measure of motivation.
        - SHAP explains the fitted model and does not establish causality.
        - OULAD represents a distance-learning context.
        - Academic staff should consider wider contextual information.
        - No adverse academic decision should be made solely from this model.
        """
    )
