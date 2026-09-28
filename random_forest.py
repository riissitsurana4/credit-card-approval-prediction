from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path(__file__).resolve().parent
APPLICATION_PATH = DATA_DIR / "application_record.csv"
CREDIT_PATH = DATA_DIR / "credit_record.csv"
BAD_STATUSES = {"2", "3", "4", "5"}
DECISION_THRESHOLD = 0.75


def classify_status(status):
    return 0 if str(status).strip() in BAD_STATUSES else 1


def prepare_data():
    application = pd.read_csv(APPLICATION_PATH)
    credit = pd.read_csv(CREDIT_PATH)

    credit["TARGET"] = credit["STATUS"].apply(classify_status)
    applicant_target = credit.groupby("ID", as_index=False)["TARGET"].min()
    credit_summary = (
        credit.groupby("ID")["MONTHS_BALANCE"]
        .agg(
            CREDIT_RECORD_COUNT="count",
            CREDIT_MONTH_MIN="min",
            CREDIT_MONTH_MAX="max",
        )
        .reset_index()
    )
    credit_summary["CREDIT_MONTH_RANGE"] = (
        credit_summary["CREDIT_MONTH_MAX"] - credit_summary["CREDIT_MONTH_MIN"]
    )

    data = application.merge(applicant_target, on="ID", how="inner")
    data = data.merge(credit_summary, on="ID", how="left")
    data = data.drop_duplicates(subset="ID").copy()
    data["AGE_YEARS"] = (-data["DAYS_BIRTH"] / 365.25).round(1)
    data["EMPLOYMENT_YEARS"] = data["DAYS_EMPLOYED"].apply(
        lambda value: 0 if value == 365243 else round(-value / 365.25, 1)
    )
    data["EMPLOYMENT_IS_UNKNOWN"] = (data["DAYS_EMPLOYED"] == 365243).astype(int)
    data["OCCUPATION_IS_UNKNOWN"] = data["OCCUPATION_TYPE"].isna().astype(int)
    data["OCCUPATION_TYPE"] = data["OCCUPATION_TYPE"].fillna("Unknown")
    data["INCOME_PER_FAMILY_MEMBER"] = (
        data["AMT_INCOME_TOTAL"] / data["CNT_FAM_MEMBERS"].replace(0, pd.NA)
    )
    return data


def feature_columns(data):
    x = data.drop(columns=["TARGET"])
    categorical = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in x.columns if column not in categorical]
    return numeric, categorical


def build_model(data):
    x = data.drop(columns=["TARGET"])
    y = data["TARGET"].astype(int)
    numeric, categorical = feature_columns(data)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def predict_applicant(model, applicant):
    row = pd.DataFrame([applicant])
    probability = float(model.predict_proba(row)[0, 1])
    return int(probability >= DECISION_THRESHOLD), probability


def choose_option(prompt, options):
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")
    while True:
        try:
            choice = int(input("Enter your choice: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(options)}.")


def get_nonnegative(prompt, value_type=float):
    while True:
        try:
            value = value_type(input(prompt))
            if value >= 0:
                return value
        except ValueError:
            pass
        print("Please enter a valid non-negative number.")


def get_applicant_from_cli(data):
    print("\n======================================")
    print("       RANDOM FOREST PREDICTION")
    print("======================================")
    applicant = {
        "ID": get_nonnegative("Applicant ID: ", int),
        "CODE_GENDER": choose_option("Gender:", sorted(data["CODE_GENDER"].dropna().unique())),
        "FLAG_OWN_CAR": choose_option("Owns a car?", sorted(data["FLAG_OWN_CAR"].dropna().unique())),
        "FLAG_OWN_REALTY": choose_option("Owns property?", sorted(data["FLAG_OWN_REALTY"].dropna().unique())),
        "CNT_CHILDREN": get_nonnegative("Number of children: ", int),
        "AMT_INCOME_TOTAL": get_nonnegative("Annual income: "),
        "NAME_INCOME_TYPE": choose_option("Income type:", sorted(data["NAME_INCOME_TYPE"].dropna().unique())),
        "NAME_EDUCATION_TYPE": choose_option("Education:", sorted(data["NAME_EDUCATION_TYPE"].dropna().unique())),
        "NAME_FAMILY_STATUS": choose_option("Family status:", sorted(data["NAME_FAMILY_STATUS"].dropna().unique())),
        "NAME_HOUSING_TYPE": choose_option("Housing type:", sorted(data["NAME_HOUSING_TYPE"].dropna().unique())),
        "DAYS_BIRTH": -round(get_nonnegative("Age in years: ") * 365.25),
        "DAYS_EMPLOYED": -round(get_nonnegative("Years employed: ") * 365.25),
        "FLAG_MOBIL": choose_option("Mobile phone available?", sorted(data["FLAG_MOBIL"].dropna().unique())),
        "FLAG_WORK_PHONE": choose_option("Work phone available?", sorted(data["FLAG_WORK_PHONE"].dropna().unique())),
        "FLAG_PHONE": choose_option("Phone available?", sorted(data["FLAG_PHONE"].dropna().unique())),
        "FLAG_EMAIL": choose_option("Email available?", sorted(data["FLAG_EMAIL"].dropna().unique())),
        "OCCUPATION_TYPE": choose_option("Occupation:", sorted(data["OCCUPATION_TYPE"].fillna("Unknown").unique())),
        "CNT_FAM_MEMBERS": get_nonnegative("Number of family members: "),
        "CREDIT_RECORD_COUNT": 0,
        "CREDIT_MONTH_MIN": 0,
        "CREDIT_MONTH_MAX": 0,
        "CREDIT_MONTH_RANGE": 0,
    }
    applicant["AGE_YEARS"] = round(-applicant["DAYS_BIRTH"] / 365.25, 1)
    applicant["EMPLOYMENT_YEARS"] = round(-applicant["DAYS_EMPLOYED"] / 365.25, 1)
    applicant["EMPLOYMENT_IS_UNKNOWN"] = int(applicant["DAYS_EMPLOYED"] == 365243)
    applicant["OCCUPATION_IS_UNKNOWN"] = int(applicant["OCCUPATION_TYPE"] == "Unknown")
    applicant["INCOME_PER_FAMILY_MEMBER"] = (
        applicant["AMT_INCOME_TOTAL"] / applicant["CNT_FAM_MEMBERS"]
        if applicant["CNT_FAM_MEMBERS"]
        else applicant["AMT_INCOME_TOTAL"]
    )
    return applicant


def run_cli():
    data = prepare_data()
    model = build_model(data)
    applicant = get_applicant_from_cli(data)
    prediction, probability = predict_applicant(model, applicant)
    print("\n======================================")
    print("       RANDOM FOREST PREDICTION")
    print("======================================\n")
    print(f"Prediction: {'GOOD CREDIT' if prediction == 1 else 'BAD CREDIT'}")
    print(f"Probability of Good Credit: {probability:.2%}")


if __name__ == "__main__":
    run_cli()
