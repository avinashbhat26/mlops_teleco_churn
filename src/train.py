from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .config import load_params, PROJECT_ROOT


def main():
    params = load_params()

    # MLflow setup
    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    target_col = params["data"]["target_col"]
    train_path = PROJECT_ROOT / params["data"]["processed_train_path"]

    df_train = pd.read_csv(train_path)
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    preprocessor = joblib.load(PROJECT_ROOT / "models" / "preprocessor.pkl")
    X_train_prepared = preprocessor.transform(X_train)

    with mlflow.start_run():
        rf_params = params["model"]
        model = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            random_state=rf_params["random_state"],
        )
        model.fit(X_train_prepared, y_train)

        y_pred = model.predict(X_train_prepared)
        acc = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        mlflow.log_params(rf_params)
        mlflow.log_metric("train_accuracy", acc)
        mlflow.log_metric("train_f1", f1)

        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / "model.pkl")

        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
