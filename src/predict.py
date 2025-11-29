from typing import Dict

import joblib
import pandas as pd

from .config import PROJECT_ROOT, load_params


class ChurnModelService:
    def __init__(self):
        # Load preprocessor and model
        self.preprocessor = joblib.load(PROJECT_ROOT / "models" / "preprocessor.pkl")
        self.model = joblib.load(PROJECT_ROOT / "models" / "model.pkl")

        # Figure out all feature columns used during training
        params = load_params()
        self.target_col = params["data"]["target_col"]
        train_path = PROJECT_ROOT / params["data"]["processed_train_path"]
        train_df = pd.read_csv(train_path)

        # All columns except target are features
        self.feature_cols = [c for c in train_df.columns if c != self.target_col]

    def _build_feature_df(self, data: Dict) -> pd.DataFrame:
        """
        Build a single-row DataFrame with ALL training features.
        Any features not provided in the request are filled with None,
        and the preprocessor's imputers will handle them.
        """
        # Start with all features = None
        row = {col: None for col in self.feature_cols}
        # Override with provided values
        row.update(data)
        # Create DataFrame with correct column order
        return pd.DataFrame([row], columns=self.feature_cols)

    def predict_single(self, data: Dict) -> Dict:
        df = self._build_feature_df(data)
        X_prepared = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X_prepared)[0, 1]
        pred = int(self.model.predict(X_prepared)[0])

        return {
            "prediction": pred,
            "churn_probability": float(proba),
        }
