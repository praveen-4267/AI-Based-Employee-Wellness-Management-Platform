from nltk.sentiment import SentimentIntensityAnalyzer


# Create VADER analyzer
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """
    Analyze text using VADER sentiment analysis.
    """

    # Validate input
    if text is None or not isinstance(text, str):
        return False, "Invalid text input."

    if not text.strip():
        return False, "Text is empty."

    # Calculate VADER sentiment scores
    scores = analyzer.polarity_scores(text)

    # Get individual scores
    positive = scores["pos"]
    negative = scores["neg"]
    neutral = scores["neu"]
    compound = scores["compound"]

    # Classify sentiment using compound score
    if compound >= 0.05:
        sentiment = "Positive"

    elif compound <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return True, {
        "text": text,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "compound": compound,
        "sentiment": sentiment
    }      