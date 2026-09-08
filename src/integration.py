from src.ingestion import read_direct_text
from src.preprocessing import preprocess_text
from src.sentiment import analyze_sentiment


def process_single_text(text):
    """
    Run one employee feedback through the complete pipeline.
    """

    # Step 1: Validate input
    valid, result = read_direct_text(text)

    if not valid:
        return False, result

    # Step 2: Preprocess
    valid, preprocessing_result = preprocess_text(text)

    if not valid:
        return False, preprocessing_result

    processed_text = preprocessing_result["processed_text"]

    # Step 3: Sentiment analysis
    valid, sentiment_result = analyze_sentiment(processed_text)

    if not valid:
        return False, sentiment_result

    # Step 4: Combine results
    final_result = {
        "input_text": text,
        "processed_text": processed_text,
        "sentiment": sentiment_result["sentiment"],
        "positive_score": sentiment_result["positive"],
        "negative_score": sentiment_result["negative"],
        "neutral_score": sentiment_result["neutral"],
        "compound_score": sentiment_result["compound"]
    }

    return True, final_result