import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Load NLP resources
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text):
    """
    Preprocess employee feedback and return
    every stage of preprocessing.
    """

    # 1. Validate input
    if text is None or not isinstance(text, str):
        return False, "Invalid text input."

    if not text.strip():
        return False, "Text is empty."

    original_text = text

    # 2. Convert to lowercase
    lowercase_text = text.lower()

    # 3. Remove URLs
    noise_filtered_text = re.sub(
        r"http\S+|www\S+",
        "",
        lowercase_text
    )

    # 4. Remove special characters and punctuation
    cleaned_text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        noise_filtered_text
    )

    # 5. Remove repeated spaces
    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text
    ).strip()

    # Check if anything remains
    if not cleaned_text:
        return False, "No valid text remains after cleaning."

    # 6. Tokenization
    tokens = word_tokenize(cleaned_text)

    # 7. Stop-word removal
    filtered_tokens = [
        word for word in tokens
        if word not in STOP_WORDS
    ]

    # 8. Lemmatization
    lemmatized_tokens = [
        LEMMATIZER.lemmatize(word)
        for word in filtered_tokens
    ]

    # 9. Final processed text
    processed_text = " ".join(lemmatized_tokens)

    return True, {
        "original": original_text,
        "lowercase": lowercase_text,
        "noise_filtered": noise_filtered_text,
        "cleaned": cleaned_text,
        "tokens": tokens,
        "without_stopwords": filtered_tokens,
        "lemmatized": lemmatized_tokens,
        "processed_text": processed_text
    }