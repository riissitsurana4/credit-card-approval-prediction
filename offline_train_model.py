from pathlib import Path

import joblib

import Logistic_regression as logistic_model
import random_forest as random_forest_model


MODEL_MODULES = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
}
MODEL_FILES = {
    model_name: Path(__file__).with_name(
        f"{model_name.lower().replace(' ', '_')}.joblib"
    )
    for model_name in MODEL_MODULES
}


def train_and_save_models():
    for model_name, model_module in MODEL_MODULES.items():
        data = model_module.prepare_data()
        model = model_module.build_model(data)
        joblib.dump(model, MODEL_FILES[model_name], compress=3)


def load_model(model_name):
    try:
        model_path = MODEL_FILES[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported model: {model_name}") from error

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Run offline_train_model.py before starting the app."
        )
    return joblib.load(model_path)


if __name__ == "__main__":
    train_and_save_models()
