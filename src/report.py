from src.preprocessing import preprocess_text
from src.sentiment import analyze_sentiment


def generate_sentiment_report(feedback_list):
    """
    Generate a sentiment report for multiple employee feedback records.
    """

    report = []

    for index, feedback in enumerate(feedback_list, start=1):

        # Preprocess the feedback
        valid, processed_result = preprocess_text(feedback)

        if not valid:
            continue

        processed_text = processed_result["processed_text"]

        # Analyze sentiment
        valid, sentiment_result = analyze_sentiment(processed_text)

        if not valid:
            continue

        # Store the results
        record = {
            "sample_number": index,
            "input_text": feedback,
            "processed_text": processed_text,
            "sentiment": sentiment_result["sentiment"],
            "positive_score": sentiment_result["positive"],
            "negative_score": sentiment_result["negative"],
            "neutral_score": sentiment_result["neutral"],
            "compound_score": sentiment_result["compound"]
        }

        report.append(record)

    return report