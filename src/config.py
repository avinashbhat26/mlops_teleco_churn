from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_params(path: str = "params.yaml") -> dict:
    params_path = PROJECT_ROOT / path
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
