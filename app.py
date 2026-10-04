
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# SOLVEXA SafeProbe — Real Dataset Diagnosis
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "solvexa_demo_model.joblib"
DATA_PATH = BASE_DIR / "solvexa_heldout_examples.csv"
FULL_DATA_PATH = BASE_DIR / "solvexa_features.csv"

st.set_page_config(
    page_title="SOLVEXA | SafeProbe",
    page_icon="🔬",
    layout="wide",
)

# Clean, judge-friendly styling
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #10263d, #164765);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0;
        color: white;
        font-size: 2.5rem;
    }
    .hero p {
        color: #d5e8f5;
        margin: 0.6rem 0 0;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>SOLVEXA · SafeProbe</h1>
        <p>Evidence-guided bearing diagnosis using real sensor recordings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Select a sensor recording, run the classifier, and then reveal "
    "the reference label to evaluate the prediction."
)


@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_examples(path, modified_time):
    return pd.read_csv(path)


# Check required files
if (
    not MODEL_PATH.exists()
    or not DATA_PATH.exists()
    or not FULL_DATA_PATH.exists()
):
    st.error(
        "Required files are missing. Keep these files in the same "
        "folder as app.py:"
    )
    st.code(
    "solvexa_demo_model.joblib\n"
    "solvexa_heldout_examples.csv\n"
    "solvexa_features.csv"
)
    st.stop()

try:
    bundle = load_model(
        str(MODEL_PATH), MODEL_PATH.stat().st_mtime
    )
    examples = load_examples(
        str(DATA_PATH), DATA_PATH.stat().st_mtime
    )
    full_df = pd.read_csv(FULL_DATA_PATH)
except Exception as exc:
    st.error(f"Could not load the model or dataset: {exc}")
    st.info(
        "Check that the project's Python environment has compatible "
        "joblib, pandas, and scikit-learn versions."
    )
    st.stop()


model = bundle["model"]
feature_cols = bundle["feature_cols"]
# Prepare the K001 reference-bearing recordings
reference_df = full_df[full_df["bearing"] == "K001"].copy()

if reference_df.empty:
    st.error("No K001 reference-bearing recordings found.")
    st.stop()

required_columns = feature_cols + ["filename", "condition", "bearing"]
missing_columns = [
    col for col in required_columns if col not in examples.columns
]

if missing_columns:
    st.error(f"The examples CSV is missing columns: {missing_columns}")
    st.stop()

if examples.empty:
    st.error("The held-out examples file contains no recordings.")
    st.stop()


st.divider()
st.header("🔬 Real Dataset Diagnosis")

st.caption(
    f"Evaluation set: {len(examples)} recordings | "
    f"Held-out operating condition: "
    f"{bundle.get('held_out_condition', 'Not specified')}"
)

# Select a recording without asking the judge to enter sensor values
def format_recording(index):
    row = faulty_examples.iloc[index]
    return f"{row['filename']}  ·  {row['condition']} · KA04"

# Show only damaged-bearing recordings in the demo dropdown
faulty_examples = examples[
    examples["bearing"].astype(str) == "KA04"
].reset_index(drop=True)

if faulty_examples.empty:
    st.error("No KA04 damaged-bearing recordings were found.")
    st.stop()


# Reset the previous diagnosis and return to the first recording
if st.button("🔄 Reset diagnosis"):
    st.session_state.pop("diagnosis_result", None)
    st.session_state.pop("reveal_reference", None)
    st.session_state["recording_selector"] = 0
    st.rerun()

selected_index = st.selectbox(
    "Select a faulty sensor recording",
    options=list(range(len(faulty_examples))),
    format_func=format_recording,
    key="recording_selector",
)


selected_row = faulty_examples.iloc[selected_index]



left, right = st.columns(2)
left.metric(
    "Recording",
    f"{selected_index + 1} of {len(faulty_examples)}"
)
right.metric("Operating condition", str(selected_row["condition"]))

st.write("### Run the bearing classifier")
st.write(
    "The model receives the extracted sensor features from the selected "
    "recording. The reference label is not passed to the classifier."
)

if st.button(
    "🚀 Run Diagnosis",
    type="primary",
    use_container_width=True,
):
    try:
        # Only sensor feature columns enter the model.
        X_one = faulty_examples.loc[[selected_index], feature_cols].copy()

        prediction = str(model.predict(X_one)[0])

        probabilities = None
        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_one)[0]
            classes = model.classes_
            confidence = float(max(probabilities))

        st.session_state["diagnosis_result"] = {
            "index": selected_index,
            "prediction": prediction,
            "probabilities": (
                probabilities.tolist()
                if probabilities is not None
                else None
            ),
            "classes": (
                [str(label) for label in classes]
                if probabilities is not None
                else None
            ),
            "confidence": confidence,
        }
        st.session_state["reveal_reference"] = False

    except Exception as exc:
        st.error(f"Diagnosis failed: {exc}")


result = st.session_state.get("diagnosis_result")

# Display the diagnosis report for the selected recording
result = st.session_state.get("diagnosis_result")

if result is not None and result["index"] == selected_index:
    prediction = result["prediction"]
    confidence = result["confidence"]

    st.divider()
    st.header("Diagnosis Report")

    if prediction == "K001":
        st.success("Detected condition: Reference bearing (K001)")
        st.write(
            "The model classifies this recording as the dataset's "
            "reference-bearing condition. This is a reference class, "
            "not proof that the machine has no possible faults."
        )
    elif prediction == "KA04":
        st.warning("Detected condition: Damaged-bearing class (KA04)")
        st.write(
            "The model classifies this recording as the KA04 "
            "bearing-damage class. The exact physical defect and "
            "severity are not determined by this classifier."
        )
    else:
        st.info(f"Predicted dataset class: {prediction}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted bearing label", prediction)

    with col2:
        if confidence is not None:
            st.metric("Model confidence", f"{confidence:.1%}")
        else:
            st.write("Confidence score unavailable.")

    st.caption(
        "Confidence is a model probability, not a guarantee of "
        "diagnostic correctness."
    )

    if result["probabilities"] is not None:
        st.subheader("Class probabilities")

        probability_table = pd.DataFrame({
            "Bearing label": result["classes"],
            "Predicted probability": [
                f"{p:.1%}" for p in result["probabilities"]
            ],
        })

        st.dataframe(
            probability_table,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Sensor evidence")

    signal_groups = {
        "Vibration": "vibration_",
        "Phase current 1": "current1_",
        "Phase current 2": "current2_",
        "Force": "force_",
        "Speed": "speed_",
        "Torque": "torque_",
    }

    evidence = []

    for signal, prefix in signal_groups.items():
        for stat in ["rms", "std", "kurtosis"]:
            feature = prefix + stat

            if feature in feature_cols:
                evidence.append({
                    "Signal": signal,
                    "Statistic": stat.upper(),
                    "Value": float(selected_row[feature]),
                })

    if evidence:
        st.dataframe(
            pd.DataFrame(evidence),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Sensor statistics describe the selected recording. They are "
        "not calibrated fault thresholds and do not independently prove "
        "the cause of a fault."
    )
    st.subheader("📊 Comparison with K001 Reference Baseline")

    st.write(
        "The table compares this recording with K001 reference recordings "
        "from the same operating condition. Differences are investigation "
        "clues, not fault-severity scores."
    )

    selected_condition = str(selected_row["condition"])
    baseline = reference_df[
        reference_df["condition"].astype(str) == selected_condition
    ].copy()
    comparison_all_df = pd.DataFrame()
    comparison_rows = []
    comparison_df = pd.DataFrame()

    if len(baseline) < 5:
        st.info(
            f"Only {len(baseline)} matching K001 reference recordings "
            f"were found for {selected_condition}. At least 5 are "
            "required for this comparison."
        )
    else:
        for feature in feature_cols:
            reference_values = pd.to_numeric(
                baseline[feature], errors="coerce"
            ).dropna()
            selected_value = pd.to_numeric(
                pd.Series([selected_row[feature]]), errors="coerce"
            ).iloc[0]

            if reference_values.empty or pd.isna(selected_value):
                continue

            reference_median = float(reference_values.median())
            q1 = float(reference_values.quantile(0.25))
            q3 = float(reference_values.quantile(0.75))
            reference_iqr = q3 - q1
            robust_scale = reference_iqr / 1.349

            if robust_scale <= 1e-12:
                robust_scale = float(reference_values.std())
            if pd.isna(robust_scale) or robust_scale < 0:
                continue

            # A minimum scale prevents near-constant reference features
            # from producing enormous, misleading standardized scores.
            reference_magnitude = max(
                abs(reference_median),
                float(reference_values.abs().median()),
                1e-12,
            )
            scale_floor = 0.05 * reference_magnitude
            scale_used = max(float(robust_scale), scale_floor, 1e-12)
            difference = float(selected_value) - reference_median
            deviation = difference / scale_used

            comparison_rows.append({
                "Feature": feature,
                "Selected value": float(selected_value),
                "K001 reference median": reference_median,
                "Difference from median": difference,
                "Reference IQR": float(reference_iqr),
                "Adjusted deviation": float(deviation),
                "Absolute deviation": abs(float(deviation)),
            })

        
        if comparison_rows:
    # Keep every valid feature for recommendation scoring.
            comparison_all_df = pd.DataFrame(comparison_rows).sort_values(
               "Absolute deviation", ascending=False
        )

    # Show only the top 10 in the comparison table.
            comparison_df = comparison_all_df.head(10).copy()


            display_df = comparison_df[[
                "Feature", "Selected value", "K001 reference median",
                "Difference from median", "Reference IQR", "Adjusted deviation"
            ]].copy()
            display_df["Adjusted deviation"] = display_df["Adjusted deviation"].map(
                lambda value: f"{value:+.2f}"
            )
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info(
                "No reliable numeric comparisons could be calculated for "
                "this recording and its matching reference set."
            )

    st.caption(
        "Adjusted deviation uses a minimum scale when the reference spread "
        "is very small. It helps rank unusual features but is not a calibrated "
        "fault threshold, physical-defect identifier, or severity estimate. "
        "Reference IQR is shown to make baseline variability visible."
    )

    st.subheader("What do these results mean?")
    if not comparison_df.empty:
        top_features = comparison_df.head(5)
        st.write(
            "These are the largest differences relative to the same-condition "
            "K001 reference set. Read them as measurements to investigate, "
            "not proof of a particular physical defect."
        )
        for _, item in top_features.iterrows():
            feature = str(item["Feature"])
            selected_value = float(item["Selected value"])
            reference_value = float(item["K001 reference median"])
            difference = float(item["Difference from median"])
            direction = "higher" if difference > 0 else "lower" if difference < 0 else "approximately equal"
            st.markdown(
                f"- **{feature.replace('_', ' ').title()}** is {direction} "
                f"than the K001 reference median ({selected_value:.4g} vs "
                f"{reference_value:.4g}; difference {difference:+.4g})."
            )
    else:
        st.info(
            "A feature-based explanation is unavailable because there are "
            "not enough matching reference recordings or valid numeric features."
        )
    
    
    st.subheader("Recommended next diagnostic test")

    if not comparison_all_df.empty:
        family_prefixes = {
            "Vibration": ("vibration_",),
            "Motor current": ("current1_", "current2_"),
            "Force/load": ("force_",),
            "Operating condition": ("speed_", "torque_"),
        }

        # Score every sensor family using ALL valid feature comparisons,
        # not only the top 10 features shown in the display table.
        # Cap each feature's contribution so a tiny reference spread cannot
        # let one extreme standardized value dominate the entire family.
        # This is a transparent prototype heuristic, not a validated threshold.
        MAX_FEATURE_CONTRIBUTION = 5.0
        family_scores = {}
        family_feature_counts = {}
        for family, prefixes in family_prefixes.items():
            deviations = [
                min(float(row["Absolute deviation"]), MAX_FEATURE_CONTRIBUTION)
                for _, row in comparison_all_df.iterrows()
                if str(row["Feature"]).startswith(prefixes)
            ]

            family_feature_counts[family] = len(deviations)
            family_scores[family] = (
                sum(deviations) / len(deviations)
                if deviations else 0.0
            )

        # Sort families from strongest to weakest.
        ranked_families = sorted(
            family_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_family, top_score = ranked_families[0]
        second_family, second_score = ranked_families[1]

        # Compare the leading score with the runner-up.
        lead = top_score - second_score
        lead_percent = (
            (lead / top_score) * 100
            if top_score > 0 else 0.0
        )

        st.markdown("### Sensor-family comparison")

        score_table = pd.DataFrame(
            [
                {
                    "Rank": rank,
                    "Sensor family": family,
                    "Mean capped deviation (max 5)": round(score, 2),
                    "Valid features scored": family_feature_counts[family],
                }
                for rank, (family, score) in enumerate(
                    ranked_families, start=1
                )
            ]
        )

        st.dataframe(
            score_table,
            hide_index=True,
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        col1.metric("Top priority", top_family, f"{top_score:.2f}")
        col2.metric(
            "Runner-up",
            second_family,
            f"{second_score:.2f}",
        )

        # A relative gap below 15% is treated as close for this prototype.
        if top_score <= 0:
            st.info(
                "No meaningful feature deviations were available. "
                "Repeat a multi-sensor recording under controlled conditions."
            )
        elif lead_percent < 15:
            st.warning(
                f"**Close scores:** {top_family} leads {second_family} "
                f"by only {lead:.2f} ({lead_percent:.1f}%). "
                "Consider investigating both sensor families."
            )
        else:
            st.success(
                f"**Leading priority:** {top_family} is ahead of "
                f"{second_family} by {lead:.2f} ({lead_percent:.1f}%)."
            )

        # Give the main recommended test.
        recommendations = {
            "Vibration": (
                "Vibration spectrum and envelope analysis",
                "Repeat the vibration measurement at the same speed and "
                "load, then inspect bearing-related frequency patterns.",
            ),
            "Motor current": (
                "Motor current signature analysis",
                "Repeat the phase-current measurement under the same "
                "operating condition and compare its spectrum with a baseline.",
            ),
            "Force/load": (
                "Force/load and sensor-mounting verification",
                "Check sensor mounting, load consistency and sensor setup, "
                "then repeat the measurement.",
            ),
            "Operating condition": (
                "Operating speed and torque verification",
                "Repeat the recording at a controlled speed and load "
                "before interpreting differences in other signals.",
            ),
        }

        test_name, test_action = recommendations[top_family]
        st.markdown(f"**Suggested test: {test_name}.** {test_action}")

        if lead_percent < 15 and top_score > 0:
            second_test, second_action = recommendations[second_family]
            st.markdown(
                f"**Also investigate: {second_test}.** {second_action}"
            )

        st.caption(
            "The 15% lead threshold is a prototype setting, not a validated "
            "industrial threshold. Scores rank statistical differences; "
            "they do not confirm a physical fault or its severity."
        )

    else:
        st.info(
            "A recommendation could not be calculated. Collect a repeat "
            "multi-sensor recording and compare it with a matching K001 "
            "reference set."
        )




    
    st.subheader("Verify the diagnosis")

    if st.button("Reveal Actual Dataset Label"):
        st.session_state["reveal_reference"] = True

    if st.session_state.get("reveal_reference", False):
        actual_label = str(selected_row["bearing"])
        st.metric("Actual dataset label", actual_label)

        if prediction == actual_label:
            st.success("Prediction matches the dataset label.")
        else:
            st.error("Prediction differs from the dataset label.")

st.divider()

with st.expander("About the dataset, labels & model"):
    st.subheader("1. What do K001 and KA04 mean?")

    st.markdown(
        """
        - **K001:** the reference-bearing label in the dataset.
        - **KA04:** the bearing-damage label used in this prototype.

        These labels identify the two classes the model was trained to
        distinguish. The model does not identify every possible motor
        fault or independently establish the physical defect's severity.
        """
    )

    st.subheader("2. What sensor data are used?")

    st.markdown(
        """
        The dataset provides six signals used by this prototype:

        - Vibration
        - Phase current 1
        - Phase current 2
        - Force
        - Speed
        - Torque

        Each signal contributes nine statistical features, giving
        **54 features per recording**. These include mean, standard
        deviation, RMS, minimum, maximum, peak, crest factor, kurtosis
        and skewness.
        """
    )

    st.subheader("3. Why only two classes?")

    st.write(
        "This proof of concept focuses on binary bearing classification "
        "before expanding to additional bearing conditions or other "
        "motor faults. It is not a general-purpose motor fault detector."
    )

    st.subheader("4. Model evaluation")

    st.write(
        "The model was trained on 116 recordings and evaluated on 40 "
        "recordings from the held-out operating condition N15_M07_F10. "
        "The reported accuracy and balanced accuracy for this split "
        "were both 100%. These recordings are from the same dataset, "
        "so this does not establish performance on independent motors."
    )

    st.subheader("5. Limitations")

    st.write(
        "This is a research prototype, not a certified industrial "
        "safety system. "
        "Real-machine decisions require independent verification."
    )

