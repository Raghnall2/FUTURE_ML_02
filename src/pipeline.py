import sys
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config, load_processed_data
from src.data.clean_text import clean_single_text
from src.models.predict import save_pipeline, load_pipeline, save_encoder, load_encoder


class TicketClassificationPipeline:
    """End-to-end inference pipeline for raw ticket text classification."""

    def __init__(self, pipeline=None, encoder=None, config=None):
        self.config = config or load_config()
        self.pipeline = pipeline
        self.encoder = encoder

    def load(self):
        """Load trained pipeline and encoder artifacts from disk."""
        self.pipeline = load_pipeline()
        self.encoder = load_encoder()
        return self

    def predict(self, texts):
        """Predict category label for raw ticket text(s)."""
        if self.pipeline is None or self.encoder is None:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        # Clean raw text inputs before passing to vectorizer
        cleaned_texts = [clean_single_text(t) for t in texts]
        encoded_preds = self.pipeline.predict(cleaned_texts)
        return self.encoder.inverse_transform(encoded_preds)


def create_classifier(config: dict):
    """Instantiate classifier based on configuration."""
    model_type = config.get("models", {}).get("active_model", "linear_svc").lower()

    if model_type == "logistic_regression":
        lr_params = config.get("models", {}).get("logistic_regression", {})
        return LogisticRegression(
            C=lr_params.get("C", 1.0),
            penalty=lr_params.get("penalty", "l2"),
            solver=lr_params.get("solver", "lbfgs"),
            max_iter=lr_params.get("max_iter", 1000),
            class_weight=lr_params.get("class_weight", "balanced"),
            random_state=lr_params.get("random_state", 42),
        )
    else:
        svc_params = config.get("models", {}).get("linear_svc", {})
        return LinearSVC(
            C=svc_params.get("C", 1.0),
            loss=svc_params.get("loss", "squared_hinge"),
            max_iter=svc_params.get("max_iter", 2000),
            class_weight=svc_params.get("class_weight", "balanced"),
            random_state=svc_params.get("random_state", 42),
        )


def train_and_save_pipeline(config: dict = None):
    """Train full scikit-learn Pipeline and persist artifacts."""
    if config is None:
        config = load_config()

    print("Loading processed dataset...")
    df = load_processed_data(config)

    target_col = config["data"].get("target_column", "Topic_group")
    text_col = "cleaned_text" if "cleaned_text" in df.columns else config["data"].get("text_column", "Document")

    X = df[text_col]
    y = df[target_col]

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Train / Test split
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)
    stratify = y_encoded if config["data"].get("stratify", True) else None

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # TF-IDF Vectorizer
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

    # Classifier
    clf = create_classifier(config)

    # Build Pipeline
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", clf),
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train_enc)

    # Save artifacts
    save_pipeline(pipeline)
    save_encoder(encoder)

    # Evaluate
    preds = pipeline.predict(X_test)
    print("\nPipeline Evaluation Report:")
    print(classification_report(y_test_enc, preds, target_names=encoder.classes_))

    return pipeline, encoder


if __name__ == "__main__":
    print("Training end-to-end pipeline...")
    pipe, enc = train_and_save_pipeline()

    # Test sample inference on raw text
    inference = TicketClassificationPipeline(pipeline=pipe, encoder=enc)
    sample_tickets = [
        "Need access to the finance department shared drive folder",
        "My laptop screen is flickering and power cable is broken",
        "Inquiry regarding employee health insurance and annual leave policy",
    ]

    print("\nInference Test on Raw Tickets:")
    for ticket in sample_tickets:
        pred = inference.predict(ticket)[0]
        print(f"Ticket: '{ticket}' -> Predicted: [{pred}]")
