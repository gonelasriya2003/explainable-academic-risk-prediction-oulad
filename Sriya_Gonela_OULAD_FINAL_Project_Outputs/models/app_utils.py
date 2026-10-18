
import numpy as np


def validate_feature_frame(
    frame,
    required_features
):

    if len(frame) != 1:

        return (
            False,
            "The CSV must contain exactly one row."
        )


    missing = [
        col
        for col in required_features
        if col not in frame.columns
    ]


    if missing:

        return (
            False,
            "Missing features: "
            + ", ".join(missing)
        )


    return (
        True,
        "Valid feature row."
    )


def update_manual_features(row):

    errors = []


    due = float(
        row.get(
            "assess_due_count",
            0
        )
        or 0
    )


    submitted = float(
        row.get(
            "assess_submitted_count",
            0
        )
        or 0
    )


    if submitted > due:

        errors.append(
            "Submitted assessments cannot exceed assessments due."
        )


    if "assess_missed_count" in row:

        row[
            "assess_missed_count"
        ] = max(
            due - submitted,
            0
        )


    if "assess_submission_rate" in row:

        row[
            "assess_submission_rate"
        ] = (
            submitted / due
            if due > 0
            else np.nan
        )


    total = float(
        row.get(
            "vle_total_clicks",
            0
        )
        or 0
    )


    active_days = float(
        row.get(
            "vle_active_days",
            0
        )
        or 0
    )


    first14 = float(
        row.get(
            "vle_clicks_first14",
            0
        )
        or 0
    )


    middle14 = float(
        row.get(
            "vle_clicks_days14_27",
            0
        )
        or 0
    )


    last14 = float(
        row.get(
            "vle_clicks_last14",
            0
        )
        or 0
    )


    if (
        first14
        +
        middle14
        +
        last14
        >
        total
    ):

        errors.append(
            "Selected VLE time-window clicks cannot exceed total clicks."
        )


    if (
        total > 0
        and
        active_days <= 0
    ):

        errors.append(
            "Active days must be above zero when clicks exist."
        )


    if "vle_avg_clicks_active_day" in row:

        row[
            "vle_avg_clicks_active_day"
        ] = (
            total / active_days
            if active_days > 0
            else 0
        )


    if "vle_engagement_trend_ratio" in row:

        row[
            "vle_engagement_trend_ratio"
        ] = (
            last14 + 1
        ) / (
            first14 + 1
        )


    return (
        row,
        errors
    )
