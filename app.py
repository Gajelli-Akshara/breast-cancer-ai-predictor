
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix)
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Breast Cancer AI Predictor",
    page_icon="🧬",
    layout="wide")

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")


data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names
feature_means = X.mean(axis=0)

st.title("🧬 Breast Cancer AI Predictor")

st.write(
    "A machine-learning prototype for classifying "
    "breast tumor measurements.")

st.info("Educational research prototype — not a clinical diagnostic tool.")

st.header("🔬 Tumor Feature Analysis")

st.write("The fields below contain the average values from the "
    "Breast Cancer Wisconsin Diagnostic dataset. "
    "You can edit them before prediction."
)

user_inputs = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):

    default_value = f"{feature_means[i]:.4f}"

    with col1 if i % 2 == 0 else col2:

        user_inputs[feature] = st.text_input(
            feature,
            value=default_value,
            key=f"feature_{i}"
        )


if st.button("🔍 Predict", type="primary"):

    try:

        input_values = []

        for feature in feature_names:

            value = float(user_inputs[feature])
            input_values.append(value)

        input_data = np.array([input_values])
        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)[0]

        probabilities = model.predict_proba(scaled_input)[0]

        confidence = float(probabilities[prediction]) * 100

        st.markdown("---")

        st.subheader("Diagnostic Result")

        if prediction == 0:

            st.error(
                f"⚠️ **Result: Malignant** "
                f"(Model confidence: {confidence:.2f}%)"
            )

        else:

            st.success(
                f"✅ **Result: Benign** "
                f"(Model confidence: {confidence:.2f}%)"
            )

    except ValueError:

        st.error(
            "Please make sure all measurement fields contain "
            "valid numerical values."
        )

    except Exception as e:

        st.error("An error occurred while making the prediction.")

        st.code(str(e))

st.markdown("---")

st.header("📊 Model Performance")

st.write(
    "The following evaluation uses a separate 80% training set "
    "and 20% testing set from the 569-sample dataset."
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

evaluation_scaler = StandardScaler()

X_train_scaled = evaluation_scaler.fit_transform(X_train)

X_test_scaled = evaluation_scaler.transform(X_test)


evaluation_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

evaluation_model.fit(
    X_train_scaled,
    y_train
)

y_pred = evaluation_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )


with col3:

    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


with col4:

    st.metric(
        "F1-Score",
        f"{f1 * 100:.2f}%"
    )



st.subheader(" Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)


st.subheader(" Dataset Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.metric(
        "Total Samples",
        X.shape[0]
    )


with info_col2:

    st.metric(
        "Features",
        X.shape[1]
    )


with info_col3:

    st.metric(
        "Test Samples",
        X_test.shape[0]
    )


st.caption(
    "Dataset: Breast Cancer Wisconsin Diagnostic dataset "
    "(569 samples and 30 numerical features)."
)


st.markdown("---")

st.warning(
    "⚠️ This application is an educational research prototype "
    "and is not intended for clinical diagnosis or medical decision-making."
)
