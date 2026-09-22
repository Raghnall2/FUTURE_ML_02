import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Add project root to sys.path FIRST before importing internal project packages
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Now imports from src will resolve cleanly
from src.data.load_data import load_processed_data, load_config


def build_tfidf_vectorizer(corpus: pd.Series, config: dict) -> TfidfVectorizer:
    """Build and fit a TF-IDF vectorizer using config parameters."""
    tfidf_params = config.get("features", {}).get("tfidf", {})
    vectorizer = TfidfVectorizer(
        max_features=tfidf_params.get("max_features", 10000),
        ngram_range=tuple(tfidf_params.get("ngram_range", (1, 2))),
        min_df=tfidf_params.get("min_df", 2),
        max_df=tfidf_params.get("max_df", 0.95),
        sublinear_tf=tfidf_params.get("sublinear_tf", True),
        norm=tfidf_params.get("norm", "l2"),
        use_idf=tfidf_params.get("use_idf", True),
    )
    vectorizer.fit(corpus)
    return vectorizer


def transform_text(vectorizer: TfidfVectorizer, texts: pd.Series):
    """Transform text using a fitted TF-IDF vectorizer."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = None) -> None:
    """Save fitted vectorizer to disk."""
    if path is None:
        config = load_config()
        path = config["models"]["artifacts"]["vectorizer_save_path"]
    save_path = PROJECT_ROOT / path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, save_path)
    print(f"Vectorizer saved to {save_path}")


def load_vectorizer(path: str = None) -> TfidfVectorizer:
    """Load fitted vectorizer from disk."""
    if path is None:
        config = load_config()
        path = config["models"]["artifacts"]["vectorizer_save_path"]
    load_path = PROJECT_ROOT / path
    return joblib.load(load_path)


if __name__ == "__main__":
    cfg = load_config()
    df = load_processed_data(cfg)
    
    print("Building TF-IDF Vectorizer...")
    vectorizer = build_tfidf_vectorizer(df["cleaned_text"], cfg)
    X = transform_text(vectorizer, df["cleaned_text"])
    print(f"Feature matrix shape: {X.shape}")
    
    save_vectorizer(vectorizer)