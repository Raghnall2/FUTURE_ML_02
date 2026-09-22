import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_processed_data, load_config
from src.features.build_features import build_tfidf_vectorizer, transform_text, save_vectorizer, load_vectorizer


def train_priority_classifier(X_train, y_train, config: dict = None):
    """Train classification model using configuration parameters."""
    if config is None:
        config = load_config()

    model_type = config.get("models", {}).get("active_model", "linear_svc").lower()

    if model_type == "linear_svc":
        svc_params = config.get("models", {}).get("linear_svc", {})
        model = LinearSVC(
            C=svc_params.get("C", 1.0),
            loss=svc_params.get("loss", "squared_hinge"),
            max_iter=svc_params.get("max_iter", 2000),
            class_weight=svc_params.get("class_weight", "balanced"),
            random_state=svc_params.get("random_state", 42),
        )
    elif model_type == "logistic_regression":
        lr_params = config.get("models", {}).get("logistic_regression", {})
        model = LogisticRegression(
            C=lr_params.get("C", 1.0),
            penalty=lr_params.get("penalty", "l2"),
            solver=lr_params.get("solver", "lbfgs"),
            max_iter=lr_params.get("max_iter", 1000),
            class_weight=lr_params.get("class_weight", "balanced"),
            random_state=lr_params.get("random_state", 42),
        )
    else:
        svc_params = config.get("models", {}).get("linear_svc", {})
        model = LinearSVC(
            C=svc_params.get("C", 1.0),
            loss=svc_params.get("loss", "squared_hinge"),
            max_iter=svc_params.get("max_iter", 2000),
            class_weight=svc_params.get("class_weight", "balanced"),
            random_state=svc_params.get("random_state", 42),
        )

    model.fit(X_train, y_train)
    return model


def save_encoder(encoder: LabelEncoder, path: str = None) -> None:
    """Save label encoder artifact to disk."""
    if path is None:
        config = load_config()
        path = config.get("models", {}).get("artifacts", {}).get(
            "encoder_save_path", "models/topic_encoder.pkl"
        )

    save_path = PROJECT_ROOT / path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, save_path)
    print(f"Encoder saved to {save_path}")


def load_encoder(path: str = None) -> LabelEncoder:
    """Load label encoder artifact from disk."""
    if path is None:
        config = load_config()
        path = config.get("models", {}).get("artifacts", {}).get(
            "encoder_save_path", "models/topic_encoder.pkl"
        )

    load_path = PROJECT_ROOT / path
    return joblib.load(load_path)


def save_model(model, path: str = None) -> None:
    """Save trained model artifact to disk."""
    if path is None:
        config = load_config()
        path = config.get("models", {}).get("artifacts", {}).get(
            "model_save_path", "models/ticket_classifier.pkl"
        )

    save_path = PROJECT_ROOT / path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")


def load_model(path: str = None):
    """Load trained model artifact from disk."""
    if path is None:
        config = load_config()
        path = config.get("models", {}).get("artifacts", {}).get(
            "model_save_path", "models/ticket_classifier.pkl"
        )

    load_path = PROJECT_ROOT / path
    return joblib.load(load_path)


def save_pipeline(pipeline, path: str = None) -> None:
    """Save sklearn Pipeline artifact to disk."""
    if path is None:
        config = load_config()
        path = config.get("models", {}).get("artifacts", {}).get(
            "pipeline_save_path", "models/ticket_pipeline.pkl"
        )

    save_path = PROJECT_ROOT / path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"Pipeline saved to {save_path}")


def load_pipeline(path: str = None):
    """Load sklearn Pipeline artifact from disk."""
    if path is None:
        config = load_config()
        path = config.get("models", {}).get("artifacts", {}).get(
            "pipeline_save_path", "models/ticket_pipeline.pkl"
        )

    load_path = PROJECT_ROOT / path
    return joblib.load(load_path)


def predict_ticket(texts, config: dict = None):
    """Predict category labels for input texts using saved artifacts."""
    if config is None:
        config = load_config()

    if isinstance(texts, str):
        texts = [texts]

    vectorizer = load_vectorizer()
    model = load_model()
    encoder = load_encoder()

    features = vectorizer.transform(texts)
    encoded_preds = model.predict(features)
    return encoder.inverse_transform(encoded_preds)


if __name__ == "__main__":
    cfg = load_config()
    df = load_processed_data(cfg)

    target_col = cfg["data"]["target_column"]
    X_raw = df["cleaned_text"]
    y = df[target_col]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=cfg["data"].get("test_size", 0.2),
        random_state=cfg["data"].get("random_state", 42),
        stratify=y,
    )
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    save_encoder(encoder)

    # Vectorize
    vectorizer = build_tfidf_vectorizer(X_train_raw, cfg)
    X_train = transform_text(vectorizer, X_train_raw)
    X_test = transform_text(vectorizer, X_test_raw)

    # Train
    model = train_priority_classifier(X_train, y_train_encoded, cfg)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test_encoded, y_pred, target_names=encoder.classes_))

    # Save artifacts
    save_vectorizer(vectorizer)
    save_model(model)
