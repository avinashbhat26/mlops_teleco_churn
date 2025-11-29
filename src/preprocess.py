from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_params, PROJECT_ROOT


def build_preprocessor(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def main():
    params = load_params()
    target_col = params["data"]["target_col"]

    train_path = PROJECT_ROOT / params["data"]["processed_train_path"]
    df_train = pd.read_csv(train_path)

    preprocessor = build_preprocessor(df_train, target_col)

    X_train = df_train.drop(columns=[target_col])
    preprocessor.fit(X_train)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")


if __name__ == "__main__":
    main()
