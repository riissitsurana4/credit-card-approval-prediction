from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent
APPLICATION_PATH = DATA_DIR / "application_record.csv"
CREDIT_PATH = DATA_DIR / "credit_record.csv"

CATEGORICAL_COLUMNS = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

NUMERIC_COLUMNS = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "CNT_FAM_MEMBERS",
    "AGE",
    "YEARS_EMPLOYED",
]


def classify_status(status):
    if pd.isna(status):
        return 1
    status = str(status).strip()
    if status in {"2", "3", "4", "5"}:
        return 0
    return 1


def prepare_data():
    application = pd.read_csv(APPLICATION_PATH)
    credit = pd.read_csv(CREDIT_PATH)

    credit["TARGET"] = credit["STATUS"].apply(classify_status)
    applicant_target = credit.groupby("ID")["TARGET"].min().reset_index()

    data = application.merge(applicant_target, on="ID", how="inner")
    data["OCCUPATION_TYPE"] = data["OCCUPATION_TYPE"].fillna("Unknown")
    data = data.drop_duplicates(subset="ID")

    data["AGE"] = (-data["DAYS_BIRTH"] / 365.25).round(1)
    data["YEARS_EMPLOYED"] = data["DAYS_EMPLOYED"].apply(
        lambda x: 0 if x == 365243 else round(-x / 365.25, 1)
    )
    data = data.drop(
        columns=[
            "ID",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "FLAG_MOBIL",
            "FLAG_WORK_PHONE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        ]
    )
    data = data.fillna({"OCCUPATION_TYPE": "Unknown"})
    return data


def build_model(data):
    x = data.drop(columns=["TARGET"])
    y = data["TARGET"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scaler", StandardScaler())]), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x, y)
    return model


def predict_applicant(model, applicant):
    applicant_df = pd.DataFrame([applicant])
    prediction = model.predict(applicant_df)[0]
    probability = model.predict_proba(applicant_df)[0][1]
    return int(prediction), float(probability)


def choose_option(prompt, options):
    print(f"\n{prompt}")
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    while True:
        try:
            choice = int(input("Enter your choice: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Please enter a valid number.")


def get_positive_number(prompt, number_type=float):
    while True:
        try:
            value = number_type(input(prompt))
            if value >= 0:
                return value
            print("Please enter a value greater than or equal to 0.")
        except ValueError:
            print("Please enter a valid number.")


def get_applicant_from_cli(data):
    print("\n======================================")
    print("     CREDIT CARD APPROVAL PREDICTOR")
    print("======================================\n")
    print("Enter applicant details:\n")

    applicant = {
        "CODE_GENDER": choose_option("Gender:", ["M", "F"]),
        "FLAG_OWN_CAR": choose_option("Owns a car?", ["Y", "N"]),
        "FLAG_OWN_REALTY": choose_option("Owns property?", ["Y", "N"]),
        "CNT_CHILDREN": get_positive_number("Number of children: ", int),
        "AMT_INCOME_TOTAL": get_positive_number("Annual income: ", float),
        "NAME_INCOME_TYPE": choose_option("Income type:", sorted(data["NAME_INCOME_TYPE"].unique())),
        "NAME_EDUCATION_TYPE": choose_option("Education:", sorted(data["NAME_EDUCATION_TYPE"].unique())),
        "NAME_FAMILY_STATUS": choose_option("Family status:", sorted(data["NAME_FAMILY_STATUS"].unique())),
        "NAME_HOUSING_TYPE": choose_option("Housing type:", sorted(data["NAME_HOUSING_TYPE"].unique())),
        "OCCUPATION_TYPE": choose_option("Occupation:", sorted(data["OCCUPATION_TYPE"].unique())),
        "CNT_FAM_MEMBERS": get_positive_number("Number of family members: ", float),
        "AGE": get_positive_number("Age: ", float),
        "YEARS_EMPLOYED": get_positive_number("Years employed: ", float),
    }
    return applicant


def run_cli():
    data = prepare_data()
    model = build_model(data)
    applicant = get_applicant_from_cli(data)
    prediction, probability = predict_applicant(model, applicant)

    print("\n======================================")
    print("     LOGISTIC REGRESSION RESULT")
    print("======================================\n")
    print(f"Prediction: {'GOOD CREDIT' if prediction == 1 else 'BAD CREDIT'}")
    print(f"Good Credit Probability: {probability:.2%}")


if __name__ == "__main__":
    run_cli()
