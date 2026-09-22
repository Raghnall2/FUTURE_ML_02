import sys
from pathlib import Path
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_raw_data, load_config, save_processed_data, load_processed_data

# Ensure required NLTK datasets are downloaded
for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Initialize resources once (for speed)
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_single_text(text: str) -> str:
    """Clean, tokenize, remove stopwords, and lemmatize a single string."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs & emails
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    text=re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

    # 4. Tokenize
    tokens = word_tokenize(text)

    # 5. Remove stopwords & short tokens, and lemmatize
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]

    # Return as cleaned text string
    return " ".join(cleaned_tokens)


def preprocess_data(save: bool = False) -> pd.DataFrame:
    """Load raw dataset and clean the 'Document' column."""
    df = load_raw_data()
    print(f"Loaded {len(df)} rows. Cleaning text...")
    
    df["cleaned_text"] = df["Document"].apply(clean_single_text)
    
    if save:
        save_processed_data(df)

    return df


if __name__ == "__main__":
    df = preprocess_data(save=True)
    print("\nCleaned Sample:")
    print(df[["Document", "cleaned_text"]].head())
