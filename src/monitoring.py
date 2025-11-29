import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from .config import load_params, PROJECT_ROOT


def main():
    params = load_params()

    train_path = PROJECT_ROOT / params["data"]["processed_train_path"]
    test_path = PROJECT_ROOT / params["data"]["processed_test_path"]

    # Load processed train/test
    ref = pd.read_csv(train_path)
    curr = pd.read_csv(test_path)

    # Build a simple data drift report
    report = Report(metrics=[DataDriftPreset()])

    # Evidently 0.4.25 API
    report.run(reference_data=ref, current_data=curr)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # save_html expects a string path
    out_path = (reports_dir / "evidently_drift.html").as_posix()
    report.save_html(out_path)


if __name__ == "__main__":
    main()
