import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import load_params, PROJECT_ROOT


def main():
    params = load_params()
    target_col = params["data"]["target_col"]

    test_path = PROJECT_ROOT / params["data"]["processed_test_path"]
    df_test = pd.read_csv(test_path)

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    preprocessor = joblib.load(PROJECT_ROOT / "models" / "preprocessor.pkl")
    model = joblib.load(PROJECT_ROOT / "models" / "model.pkl")

    X_test_prepared = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_prepared)
    y_proba = model.predict_proba(X_test_prepared)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc,
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
