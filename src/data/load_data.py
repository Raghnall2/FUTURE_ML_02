import sys
from pathlib import Path
import yaml
import pandas as pd

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration dictionary from YAML file."""
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict = None) -> pd.DataFrame:
    """Load raw dataset from path defined in config.yaml."""
    if config is None:
        config = load_config()

    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    return pd.read_csv(raw_path)


def load_processed_data(config: dict = None) -> pd.DataFrame:
    """Load processed dataset from path defined in config.yaml."""
    if config is None:
        config = load_config()

    processed_path = PROJECT_ROOT / config["data"]["processed_path"]
    return pd.read_csv(processed_path)


def save_processed_data(df: pd.DataFrame, config: dict = None) -> None:
    """Save processed dataframe to destination path."""
    if config is None:
        config = load_config()

    processed_path = PROJECT_ROOT / config["data"]["processed_path"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Raw data successfully loaded! Shape: {df.shape}")
