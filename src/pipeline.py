import sys
from pathlib import Path
from typing import List, Union
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config, load_processed_data
from src.data.clean_text import clean_single_text
from src.features.build_features import build_tfidf_vectorizer
from src.models.predict import train_priority_classifier, save_encoder, load_encoder


class TicketClassificationPipeline:
    """End-to-end inference pipeline for raw support ticket text."""

    def __init__(self, pipeline: Pipeline = None, encoder: LabelEncoder = None, config: dict = None):
        self.config = config or load_config()
        self.pipeline = pipeline
        self.encoder = encoder

    @classmethod
    def load(cls, config: dict = None) -> "TicketClassificationPipeline":
        """Load trained pipeline and encoder artifacts from disk."""
        if config is None:
            config = load_config()

        pipe_path = PROJECT_ROOT / config.get("models", {}).get("artifacts", {}).get(
            "pipeline_save_path", "models/ticket_pipeline.pkl"
        )
        encoder_path = PROJECT_ROOT / config.get("models", {}).get("artifacts", {}).get(
            "encoder_save_path", "models/topic_encoder.pkl"
        )

        pipeline = joblib.load(pipe_path)
        encoder = joblib.load(encoder_path)
        return cls(pipeline=pipeline, encoder=encoder, config=config)

    def predict(self, texts: Union[str, List[str]]) -> List[str]:
        """Accept raw strings and return predicted category labels."""
        if isinstance(texts, str):
            texts = [texts]

        # 1. Clean raw text input
        cleaned = [clean_single_text(t) for t in texts]

        # 2. Predict through sklearn pipeline (TF-IDF + LinearSVC)
        encoded_preds = self.pipeline.predict(cleaned)

        # 3. Inverse transform back to human-readable categories
        return self.encoder.inverse_transform(encoded_preds).tolist()


def train_and_save_pipeline(config: dict = None):
    """Train unified TF-IDF + Classifier pipeline and save artifacts."""
    if config is None:
        config = load_config()

    df = load_processed_data(config)
    target_col = config["data"]["target_column"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df[target_col],
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["data"].get("random_state", 42),
        stratify=df[target_col],
    )

    # Fit label encoder
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    save_encoder(encoder)

    # Build vectorizer & classifier
    vectorizer = build_tfidf_vectorizer(X_train, config)
    classifier = train_priority_classifier(
        vectorizer.transform(X_train), y_train_enc, config
    )

    # Combine into unified Pipeline
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])

    # Save pipeline
    pipe_path = PROJECT_ROOT / config["models"]["artifacts"]["pipeline_save_path"]
    pipe_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, pipe_path)
    print(f"Pipeline saved to {pipe_path}")

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
