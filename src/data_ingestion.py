from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import load_params, PROJECT_ROOT


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def split_and_save(df: pd.DataFrame, params: dict) -> None:
    target_col = params["data"]["target_col"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)


def main():
    params = load_params()
    raw_path = PROJECT_ROOT / params["data"]["raw_path"]

    df = load_raw_data(raw_path)

    # Drop ID column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Map target to 0/1 if Yes/No
    target_col = params["data"]["target_col"]
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"No": 0, "Yes": 1})

    split_and_save(df, params)


if __name__ == "__main__":
    main()
