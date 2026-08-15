import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DATABASE
# =========================================================

DATABASE_NAME = "loan_predictions.db"


def create_database():

    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            loan_id INTEGER,
            no_of_dependents INTEGER,
            education TEXT,
            self_employed TEXT,
            income_annum REAL,
            loan_amount REAL,
            loan_term INTEGER,
            cibil_score INTEGER,
            residential_assets_value REAL,
            commercial_assets_value REAL,
            luxury_assets_value REAL,
            bank_asset_value REAL,
            prediction TEXT,
            approved_probability REAL,
            rejected_probability REAL,
            prediction_time TEXT
        )
    """)

    connection.commit()
    connection.close()


create_database()


# =========================================================
# SAVE PREDICTION
# =========================================================

def save_prediction(
    loan_id,
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
    prediction,
    approved_probability,
    rejected_probability
):

    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            loan_id,
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value,
            prediction,
            approved_probability,
            rejected_probability,
            prediction_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        loan_id,
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        prediction,
        approved_probability,
        rejected_probability,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


# =========================================================
# GET HISTORY
# =========================================================

def get_prediction_history():

    connection = sqlite3.connect(DATABASE_NAME)

    history = pd.read_sql_query(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """,
        connection
    )

    connection.close()

    return history


# =========================================================
# CLEAR HISTORY
# =========================================================

def clear_prediction_history():

    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("DELETE FROM predictions")

    connection.commit()
    connection.close()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load("loan_approval_predictor.pkl")


model = load_model()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model):

    preprocessor = model.named_steps["preprocessor"]
    decision_tree = model.named_steps["model"]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    importances = (
        decision_tree.feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    return importance_df.sort_values(
        by="Importance",
        ascending=False
    )


# =========================================================
# CLEAN FEATURE NAMES
# =========================================================

def clean_feature_name(feature):

    replacements = {

        "num__loan_id": "Loan ID",

        "num__no_of_dependents":
            "Number of Dependents",

        "num__income_annum":
            "Annual Income",

        "num__loan_amount":
            "Loan Amount",

        "num__loan_term":
            "Loan Term",

        "num__cibil_score":
            "CIBIL Score",

        "num__residential_assets_value":
            "Residential Assets",

        "num__commercial_assets_value":
            "Commercial Assets",

        "num__luxury_assets_value":
            "Luxury Assets",

        "num__bank_asset_value":
            "Bank Asset Value",

        "cat__education_Graduate":
            "Education - Graduate",

        "cat__education_Not Graduate":
            "Education - Not Graduate",

        "cat__self_employed_No":
            "Self Employed - No",

        "cat__self_employed_Yes":
            "Self Employed - Yes"
    }

    return replacements.get(
        feature,
        feature
    )


# =========================================================
# HEADER
# =========================================================

st.title("🏦 Loan Approval Predictor")

st.write(
    "Intelligent loan approval prediction powered by Machine Learning"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🏦 About the Project")

    st.write(
        """
        This application predicts whether a loan
        application is likely to be Approved or
        Rejected using a trained Decision Tree
        Classifier.
        """
    )

    st.divider()

    st.subheader("🤖 Machine Learning Model")

    st.write("Decision Tree Classifier")

    st.subheader("⚙️ Preprocessing")

    st.write("• RobustScaler")
    st.write("• OneHotEncoder")

    st.subheader("📊 Prediction Classes")

    st.write("0 → Approved")
    st.write("1 → Rejected")

    st.divider()

    st.info(
        "Please enter all required information "
        "before making a prediction."
    )


# =========================================================
# APPLICANT INFORMATION
# =========================================================

st.header("👤 Applicant Information")

col1, col2, col3 = st.columns(3)


with col1:

    loan_id = st.number_input(
        "Loan ID *",
        min_value=1,
        value=None,
        placeholder="Enter Loan ID",
        step=1
    )


with col2:

    no_of_dependents = st.number_input(
        "Number of Dependents *",
        min_value=0,
        value=None,
        placeholder="Enter number of dependents",
        step=1
    )


with col3:

    education = st.selectbox(
        "Education *",
        [
            "Select education",
            "Graduate",
            "Not Graduate"
        ]
    )


col1, col2, col3 = st.columns(3)


with col1:

    self_employed = st.selectbox(
        "Self Employed *",
        [
            "Select option",
            "No",
            "Yes"
        ]
    )


with col2:

    cibil_score = st.number_input(
        "CIBIL Score *",
        min_value=1,
        max_value=900,
        value=None,
        placeholder="Enter CIBIL score",
        step=1
    )


with col3:

    loan_term = st.number_input(
        "Loan Term *",
        min_value=1,
        value=None,
        placeholder="Enter loan term",
        step=1
    )


# =========================================================
# FINANCIAL INFORMATION
# =========================================================

st.header("💰 Financial Information")

col1, col2, col3 = st.columns(3)


with col1:

    income_annum = st.number_input(
        "Annual Income *",
        min_value=1,
        value=None,
        placeholder="Enter annual income",
        step=10000
    )


with col2:

    loan_amount = st.number_input(
        "Loan Amount *",
        min_value=1,
        value=None,
        placeholder="Enter loan amount",
        step=10000
    )


with col3:

    if (
        income_annum is not None
        and loan_amount is not None
        and income_annum > 0
    ):

        loan_income_ratio = (
            loan_amount / income_annum
        )

        st.metric(
            "Loan / Income Ratio",
            f"{loan_income_ratio:.2f}"
        )

    else:

        st.metric(
            "Loan / Income Ratio",
            "—"
        )


# =========================================================
# ASSET INFORMATION
# =========================================================

st.header("🏠 Asset Information")

col1, col2 = st.columns(2)


with col1:

    residential_assets_value = st.number_input(
        "Residential Assets Value *",
        min_value=0,
        value=None,
        placeholder="Enter residential assets value",
        step=10000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value *",
        min_value=0,
        value=None,
        placeholder="Enter commercial assets value",
        step=10000
    )


with col2:

    luxury_assets_value = st.number_input(
        "Luxury Assets Value *",
        min_value=0,
        value=None,
        placeholder="Enter luxury assets value",
        step=10000
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value *",
        min_value=0,
        value=None,
        placeholder="Enter bank asset value",
        step=10000
    )


# =========================================================
# TOTAL ASSETS
# =========================================================

if all(
    value is not None
    for value in [
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]
):

    total_assets = (
        residential_assets_value
        + commercial_assets_value
        + luxury_assets_value
        + bank_asset_value
    )

else:

    total_assets = None


col1, col2, col3 = st.columns(3)


with col2:

    if total_assets is not None:

        st.metric(
            "🏠 Total Asset Value",
            f"₹ {total_assets:,.0f}"
        )

    else:

        st.metric(
            "🏠 Total Asset Value",
            "—"
        )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Loan Approval",
    use_container_width=True,
    type="primary"
)


# =========================================================
# VALIDATION + PREDICTION
# =========================================================

if predict_button:

    missing_fields = []


    # -----------------------------------------------------
    # CHECK NUMERICAL FIELDS
    # -----------------------------------------------------

    if loan_id is None:
        missing_fields.append("Loan ID")

    if no_of_dependents is None:
        missing_fields.append(
            "Number of Dependents"
        )

    if cibil_score is None:
        missing_fields.append(
            "CIBIL Score"
        )

    if loan_term is None:
        missing_fields.append(
            "Loan Term"
        )

    if income_annum is None:
        missing_fields.append(
            "Annual Income"
        )

    if loan_amount is None:
        missing_fields.append(
            "Loan Amount"
        )

    if residential_assets_value is None:
        missing_fields.append(
            "Residential Assets Value"
        )

    if commercial_assets_value is None:
        missing_fields.append(
            "Commercial Assets Value"
        )

    if luxury_assets_value is None:
        missing_fields.append(
            "Luxury Assets Value"
        )

    if bank_asset_value is None:
        missing_fields.append(
            "Bank Asset Value"
        )


    # -----------------------------------------------------
    # CHECK CATEGORICAL FIELDS
    # -----------------------------------------------------

    if education == "Select education":

        missing_fields.append(
            "Education"
        )


    if self_employed == "Select option":

        missing_fields.append(
            "Self Employed"
        )


    # =====================================================
    # SHOW VALIDATION ERROR
    # =====================================================

    if missing_fields:

        st.error(
            "⚠️ Please complete all required fields "
            "before making a prediction."
        )

        st.warning(
            "Missing: "
            + ", ".join(missing_fields)
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    else:

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame({

            "loan_id": [loan_id],

            "no_of_dependents": [
                no_of_dependents
            ],

            "education": [
                education
            ],

            "self_employed": [
                self_employed
            ],

            "income_annum": [
                income_annum
            ],

            "loan_amount": [
                loan_amount
            ],

            "loan_term": [
                loan_term
            ],

            "cibil_score": [
                cibil_score
            ],

            "residential_assets_value": [
                residential_assets_value
            ],

            "commercial_assets_value": [
                commercial_assets_value
            ],

            "luxury_assets_value": [
                luxury_assets_value
            ],

            "bank_asset_value": [
                bank_asset_value
            ]
        })


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        probabilities = model.predict_proba(
            input_data
        )[0]


        approved_probability = (
            probabilities[0] * 100
        )


        rejected_probability = (
            probabilities[1] * 100
        )


        if prediction == 0:

            prediction_label = "Approved"

            confidence = approved_probability

        else:

            prediction_label = "Rejected"

            confidence = rejected_probability


        # -------------------------------------------------
        # SAVE TO DATABASE
        # -------------------------------------------------

        save_prediction(

            loan_id,
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value,
            prediction_label,
            approved_probability,
            rejected_probability
        )


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.header("📊 Prediction Result")


        if prediction == 0:

            st.success(
                "## ✅ LOAN APPROVED"
            )

            st.write(
                f"**Model Confidence: "
                f"{approved_probability:.2f}%**"
            )

        else:

            st.error(
                "## ❌ LOAN REJECTED"
            )

            st.write(
                f"**Model Confidence: "
                f"{rejected_probability:.2f}%**"
            )


        # =================================================
        # PROBABILITY
        # =================================================

        st.subheader(
            "📈 Prediction Probability"
        )

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "✅ Approved",
                f"{approved_probability:.2f}%"
            )

            st.progress(
                int(approved_probability)
            )


        with col2:

            st.metric(
                "❌ Rejected",
                f"{rejected_probability:.2f}%"
            )

            st.progress(
                int(rejected_probability)
            )


        # =================================================
        # APPLICATION SUMMARY
        # =================================================

        st.subheader(
            "📋 Application Summary"
        )

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "CIBIL Score",
                cibil_score
            )


        with col2:

            st.metric(
                "Loan Amount",
                f"₹ {loan_amount:,.0f}"
            )


        with col3:

            st.metric(
                "Annual Income",
                f"₹ {income_annum:,.0f}"
            )


        with col4:

            st.metric(
                "Total Assets",
                f"₹ {total_assets:,.0f}"
            )


        # =================================================
        # MODEL INSIGHTS
        # =================================================

        st.divider()

        st.header(
            "🧠 Model Insights"
        )


        importance_df = (
            get_feature_importance(model)
        )


        importance_df["Feature"] = (
            importance_df["Feature"]
            .apply(clean_feature_name)
        )


        top_features = (
            importance_df
            .head(10)
            .copy()
        )


        st.subheader(
            "Top 10 Important Features"
        )


        st.bar_chart(
            top_features
            .set_index("Feature")
            ["Importance"]
        )


        # =================================================
        # FEATURE DETAILS
        # =================================================

        with st.expander(
            "🔍 View Feature Importance Details"
        ):

            display_df = (
                top_features.copy()
            )


            display_df["Importance"] = (
                display_df["Importance"]
                * 100
            ).round(2)


            display_df = (
                display_df.rename(
                    columns={
                        "Importance":
                        "Importance (%)"
                    }
                )
            )


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


        # =================================================
        # SUBMITTED APPLICATION
        # =================================================

        with st.expander(
            "📄 View Submitted Application"
        ):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


        # =================================================
        # DOWNLOAD REPORT
        # =================================================

        st.subheader(
            "📥 Download Prediction Report"
        )


        report_data = (
            input_data.copy()
        )


        report_data["Prediction"] = (
            prediction_label
        )


        report_data[
            "Approved Probability (%)"
        ] = round(
            approved_probability,
            2
        )


        report_data[
            "Rejected Probability (%)"
        ] = round(
            rejected_probability,
            2
        )


        report_data[
            "Prediction Time"
        ] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        csv_data = (
            report_data
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            label="📥 Download Report",
            data=csv_data,
            file_name="loan_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

st.divider()

st.header(
    "📜 Prediction History"
)


history_df = (
    get_prediction_history()
)


if history_df.empty:

    st.info(
        "No prediction history available yet."
    )

else:

    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    total_predictions = len(
        history_df
    )


    approved_count = (
        history_df["prediction"]
        .eq("Approved")
        .sum()
    )


    rejected_count = (
        history_df["prediction"]
        .eq("Rejected")
        .sum()
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📊 Total Predictions",
            total_predictions
        )


    with col2:

        st.metric(
            "✅ Approved",
            approved_count
        )


    with col3:

        st.metric(
            "❌ Rejected",
            rejected_count
        )


    # -----------------------------------------------------
    # HISTORY TABLE
    # -----------------------------------------------------

    st.subheader(
        "Recent Applications"
    )


    history_display = (
        history_df.copy()
    )


    history_display = (
        history_display.rename(
            columns={
                "id": "Record ID",
                "loan_id": "Loan ID",
                "no_of_dependents":
                    "Dependents",
                "education":
                    "Education",
                "self_employed":
                    "Self Employed",
                "income_annum":
                    "Annual Income",
                "loan_amount":
                    "Loan Amount",
                "loan_term":
                    "Loan Term",
                "cibil_score":
                    "CIBIL Score",
                "prediction":
                    "Prediction",
                "approved_probability":
                    "Approved %",
                "rejected_probability":
                    "Rejected %",
                "prediction_time":
                    "Prediction Time"
            }
        )
    )


    history_display[
        "Annual Income"
    ] = history_display[
        "Annual Income"
    ].apply(
        lambda x:
        f"₹ {x:,.0f}"
    )


    history_display[
        "Loan Amount"
    ] = history_display[
        "Loan Amount"
    ].apply(
        lambda x:
        f"₹ {x:,.0f}"
    )


    history_display[
        "Approved %"
    ] = history_display[
        "Approved %"
    ].apply(
        lambda x:
        f"{x:.2f}%"
    )


    history_display[
        "Rejected %"
    ] = history_display[
        "Rejected %"
    ].apply(
        lambda x:
        f"{x:.2f}%"
    )


    st.dataframe(
        history_display,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # DOWNLOAD HISTORY
    # -----------------------------------------------------

    history_csv = (
        history_df
        .to_csv(index=False)
        .encode("utf-8")
    )


    st.download_button(
        label="📥 Download Prediction History",
        data=history_csv,
        file_name="loan_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )


    # -----------------------------------------------------
    # CLEAR HISTORY
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "⚠️ Database Management"
    )


    if st.button(
        "🗑️ Clear All Prediction History"
    ):

        clear_prediction_history()

        st.success(
            "Prediction history cleared successfully."
        )

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🏦 Loan Approval Prediction System | "
    "Python • Scikit-learn • Streamlit • SQLite"
)