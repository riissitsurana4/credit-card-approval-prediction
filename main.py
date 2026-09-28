import streamlit as st

import Logistic_regression as logistic_model
from offline_train_model import load_model as load_offline_model
import random_forest as random_forest_model


MODEL_OPTIONS = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
}


@st.cache_data
def load_data(model_name):
    return MODEL_OPTIONS[model_name].prepare_data()


@st.cache_resource
def load_model(model_name):
    return load_offline_model(model_name)


def category_options(data, column):
    return sorted(str(value) for value in data[column].dropna().unique())


def applicant_form(data):
    st.subheader("Applicant details")

    first_column, second_column = st.columns(2)
    with first_column:
        gender = st.selectbox(
            "Gender", category_options(data, "CODE_GENDER"), index=None,
            placeholder="Select gender",
        )
        owns_car = st.selectbox(
            "Owns a car?", category_options(data, "FLAG_OWN_CAR"), index=None,
            placeholder="Select an option",
        )
        owns_realty = st.selectbox(
            "Owns property?", category_options(data, "FLAG_OWN_REALTY"), index=None,
            placeholder="Select an option",
        )
        children = st.number_input(
            "Number of children", min_value=0, step=1, value=None
        )
        income = st.number_input(
            "Annual income", min_value=0.0, step=1000.0, value=None
        )
        age = st.number_input("Age", min_value=0.0, step=1.0, value=None)

    with second_column:
        income_type = st.selectbox(
            "Income type", category_options(data, "NAME_INCOME_TYPE"), index=None,
            placeholder="Select income type",
        )
        education = st.selectbox(
            "Education", category_options(data, "NAME_EDUCATION_TYPE"), index=None,
            placeholder="Select education",
        )
        family_status = st.selectbox(
            "Family status", category_options(data, "NAME_FAMILY_STATUS"), index=None,
            placeholder="Select family status",
        )
        housing = st.selectbox(
            "Housing type", category_options(data, "NAME_HOUSING_TYPE"), index=None,
            placeholder="Select housing type",
        )
        occupation = st.selectbox(
            "Occupation", category_options(data, "OCCUPATION_TYPE"), index=None,
            placeholder="Select occupation",
        )
        family_members = st.number_input(
            "Number of family members", min_value=0.0, step=1.0, value=None
        )
        years_employed = st.number_input(
            "Years employed", min_value=0.0, step=1.0, value=None
        )

    st.subheader("Model")
    model_name = st.selectbox(
        "Choose the model to use",
        list(MODEL_OPTIONS),
        index=None,
        placeholder="Select a model",
        help="The selected model will be loaded from its offline training artifact.",
    )
    submitted = st.form_submit_button("Predict credit status", type="primary")

    if not submitted:
        return None

    required_choices = {
        "Gender": gender,
        "Owns a car": owns_car,
        "Owns property": owns_realty,
        "Income type": income_type,
        "Education": education,
        "Family status": family_status,
        "Housing type": housing,
        "Occupation": occupation,
        "Model": model_name,
        "Number of children": children,
        "Annual income": income,
        "Age": age,
        "Number of family members": family_members,
        "Years employed": years_employed,
    }
    missing_choices = [
        label for label, value in required_choices.items() if value is None
    ]
    if missing_choices:
        st.warning("Please make a selection for: " + ", ".join(missing_choices))
        return None

    common_values = {
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": owns_car,
        "FLAG_OWN_REALTY": owns_realty,
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "OCCUPATION_TYPE": occupation,
        "CNT_FAM_MEMBERS": family_members,
    }

    if model_name == "Logistic Regression":
        common_values.update(
            {
                "AGE": age,
                "YEARS_EMPLOYED": years_employed,
            }
        )
    else:
        common_values.update(
            {
                "ID": 0,
                "DAYS_BIRTH": -round(age * 365.25),
                "DAYS_EMPLOYED": -round(years_employed * 365.25),
                "AGE_YEARS": round(age, 1),
                "EMPLOYMENT_YEARS": round(years_employed, 1),
                "EMPLOYMENT_IS_UNKNOWN": 0,
                "OCCUPATION_IS_UNKNOWN": int(occupation == "Unknown"),
                "FLAG_MOBIL": category_options(data, "FLAG_MOBIL")[0],
                "FLAG_WORK_PHONE": category_options(data, "FLAG_WORK_PHONE")[0],
                "FLAG_PHONE": category_options(data, "FLAG_PHONE")[0],
                "FLAG_EMAIL": category_options(data, "FLAG_EMAIL")[0],
                "CREDIT_RECORD_COUNT": 0,
                "CREDIT_MONTH_MIN": 0,
                "CREDIT_MONTH_MAX": 0,
                "CREDIT_MONTH_RANGE": 0,
                "INCOME_PER_FAMILY_MEMBER": (
                    income / family_members if family_members else income
                ),
            }
        )

    return model_name, common_values


def main():
    st.set_page_config(page_title="Credit Approval Predictor", page_icon=":chart_with_upwards_trend:")
    st.title("Credit Approval Predictor")
    st.write("Enter an applicant's details and choose a model to estimate credit status.")

    data = load_data("Random Forest")

    with st.form("applicant_form"):
        result = applicant_form(data)

    if result is None:
        return

    model_name, applicant = result
    st.session_state["selected_model"] = model_name
    model = load_model(model_name)
    prediction, probability = MODEL_OPTIONS[model_name].predict_applicant(
        model, applicant
    )

    if prediction == 1:
        st.success(f"Good credit (probability: {probability:.2%})")
    else:
        st.error(f"Bad credit (probability of good credit: {probability:.2%})")


if __name__ == "__main__":
    main()
