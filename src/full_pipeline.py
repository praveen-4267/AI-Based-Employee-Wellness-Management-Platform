
"""
Task 7: Full AI Employee Wellness Pipeline

Pipeline:
Input
  ↓
Milestone 1 - Ingestion
  ↓
Milestone 1 - Preprocessing
  ↓
Milestone 1 - VADER Sentiment
  ↓
Milestone 2 - BERT / DistilBERT Emotion
  ↓
Milestone 2 - Multi-label Emotion
  ↓
Confidence
  ↓
Final Result
"""

import os
import json
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.ingestion import read_direct_text
from src.preprocessing import preprocess_text
from src.sentiment import analyze_sentiment


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BERT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "models",
    "bert_emotion"
)

DISTILBERT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "models",
    "distilbert_emotion"
)

MULTILABEL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "models",
    "multilabel_emotion"
)

LABEL_MAPPING_PATH = os.path.join(
    MULTILABEL_MODEL_PATH,
    "label_mapping.json"
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Load label mapping
# ---------------------------------------------------------

def load_label_mapping():
    """Load emotion labels from the trained model."""

    if not os.path.exists(LABEL_MAPPING_PATH):
        raise FileNotFoundError(
            f"Label mapping not found: {LABEL_MAPPING_PATH}"
        )

    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    # Support the label mapping format used by our project
    if "label_to_id" in mapping:
        return {
            int(value): key
            for key, value in mapping["label_to_id"].items()
        }

    if "id2label" in mapping:
        return {
            int(key): value
            for key, value in mapping["id2label"].items()
        }

    raise KeyError(
        "label_mapping.json does not contain "
        "'label_to_id' or 'id2label'."
    )

# ---------------------------------------------------------
# Load single-label BERT / DistilBERT
# ---------------------------------------------------------

def load_single_label_model(model_type="bert"):
    """
    Load the trained BERT or DistilBERT model.

    model_type:
        "bert"
        "distilbert"
    """

    if model_type.lower() == "bert":
        model_path = BERT_MODEL_PATH

    elif model_type.lower() == "distilbert":
        model_path = DISTILBERT_MODEL_PATH

    else:
        raise ValueError(
            "model_type must be 'bert' or 'distilbert'"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ---------------------------------------------------------
# Load multi-label model
# ---------------------------------------------------------

def load_multilabel_model():
    """Load the trained multi-label BERT model."""

    if not os.path.exists(MULTILABEL_MODEL_PATH):
        raise FileNotFoundError(
            f"Multi-label model not found: {MULTILABEL_MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MULTILABEL_MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MULTILABEL_MODEL_PATH
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ---------------------------------------------------------
# BERT / DistilBERT prediction
# ---------------------------------------------------------

def predict_single_emotion(
    text,
    tokenizer,
    model,
    label_mapping
):
    """
    Predict one primary emotion using BERT or DistilBERT.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_id = torch.argmax(probabilities).item()

    primary_emotion = label_mapping.get(
        predicted_id,
        model.config.id2label.get(
            predicted_id,
            str(predicted_id)
        )
    )

    primary_confidence = float(
        probabilities[predicted_id].item()
    )

    all_probabilities = {}

    for index, probability in enumerate(probabilities):
        label = label_mapping.get(
            index,
            model.config.id2label.get(
                index,
                str(index)
            )
        )

        all_probabilities[label] = float(
            probability.item()
        )

    return {
        "primary_emotion": primary_emotion,
        "primary_confidence": primary_confidence,
        "emotion_probabilities": all_probabilities
    }


# ---------------------------------------------------------
# Multi-label prediction
# ---------------------------------------------------------

def predict_multilabel_emotions(
    text,
    tokenizer,
    model,
    label_mapping,
    threshold=0.5
):
    """
    Predict one or more applicable emotions.

    Sigmoid is used because this is a multi-label model.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(
        outputs.logits
    )[0]

    all_probabilities = {}

    predicted_emotions = []

    for index, probability in enumerate(probabilities):

        label = label_mapping.get(
            index,
            model.config.id2label.get(
                index,
                str(index)
            )
        )

        score = float(probability.item())

        all_probabilities[label] = score

        if score >= threshold:
            predicted_emotions.append(label)

    # Make sure there is always one emotion
    # even when no probability reaches the threshold.
    if not predicted_emotions:

        highest_index = torch.argmax(
            probabilities
        ).item()

        highest_label = label_mapping.get(
            highest_index,
            model.config.id2label.get(
                highest_index,
                str(highest_index)
            )
        )

        predicted_emotions.append(highest_label)

    return {
        "predicted_emotions": predicted_emotions,
        "probabilities": all_probabilities,
        "threshold": threshold
    }


# ---------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------


def process_employee_feedback(
    text,
    model_type="bert",
    threshold=0.5
):
    """
    Run the complete Milestone 1 + Milestone 2 pipeline.
    """

    # 1. INGESTION
    valid, ingested_text = read_direct_text(text)

    if not valid:
        raise ValueError(
            f"Input text is invalid: {ingested_text}"
        )

    # 2. PREPROCESSING
    preprocess_valid, preprocessing_result = preprocess_text(
        ingested_text
    )

    if not preprocess_valid:
        raise ValueError(
            f"Preprocessing failed: {preprocessing_result}"
        )

    processed_text = preprocessing_result["processed_text"]

    if not processed_text.strip():
        raise ValueError(
            "Text became empty after preprocessing."
        )

    # 3. VADER SENTIMENT
    sentiment_valid, sentiment_result = analyze_sentiment(
        processed_text
    )

    if not sentiment_valid:
        raise ValueError(
            f"Sentiment analysis failed: {sentiment_result}"
        )

    # 4. LABEL MAPPING
    label_mapping = load_label_mapping()

    # 5. BERT / DISTILBERT
    single_tokenizer, single_model = (
        load_single_label_model(model_type)
    )

    single_prediction = predict_single_emotion(
        processed_text,
        single_tokenizer,
        single_model,
        label_mapping
    )

    # 6. MULTI-LABEL EMOTION
    multi_tokenizer, multi_model = (
        load_multilabel_model()
    )

    multi_prediction = predict_multilabel_emotions(
        processed_text,
        multi_tokenizer,
        multi_model,
        label_mapping,
        threshold
    )

    # 7. FINAL RESULT
    result = {
        "input_text": ingested_text,
        "processed_text": processed_text,
        "sentiment": sentiment_result,
        "model": model_type,
        "primary_emotion":
            single_prediction["primary_emotion"],
        "primary_confidence":
            single_prediction["primary_confidence"],
        "emotion_probabilities":
            single_prediction["emotion_probabilities"],
        "multi_label_emotions":
            multi_prediction["predicted_emotions"],
        "multi_label_probabilities":
            multi_prediction["probabilities"],
        "confidence_threshold":
            multi_prediction["threshold"]
    }

    return result


def print_result(result):
    """Display the complete pipeline result."""

    print("\n" + "=" * 60)
    print("AI EMPLOYEE WELLNESS ANALYSIS")
    print("=" * 60)

    print("\nInput Text:")
    print(result["input_text"])

    print("\nProcessed Text:")
    print(result["processed_text"])

    print("\nVADER Sentiment:")
    print(result["sentiment"])

    print("\nModel:")
    print(result["model"])

    print("\nPrimary Emotion:")
    print(result["primary_emotion"])

    print("\nPrimary Confidence:")
    print(
        f"{result['primary_confidence'] * 100:.2f}%"
    )

    print("\nSingle-label Emotion Probabilities:")

    for emotion, probability in sorted(
        result["emotion_probabilities"].items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(
            f"  {emotion}: "
            f"{probability * 100:.2f}%"
        )

    print("\nMulti-label Emotions:")

    for emotion in result["multi_label_emotions"]:
        probability = result[
            "multi_label_probabilities"
        ].get(emotion, 0)

        print(
            f"  {emotion}: "
            f"{probability * 100:.2f}%"
        )

    print("\nConfidence Threshold:")
    print(
        f"{result['confidence_threshold'] * 100:.0f}%"
    )

    print("\n" + "=" * 60)
if __name__ == "__main__":

    print("AI Employee Wellness Management Platform")
    print("Milestone 1 + Milestone 2 Full Pipeline")
    print(f"Running on: {DEVICE}")

    test_text = (
        "I am very happy with my team, "
        "but I am also nervous about the upcoming deadline."
    )

    try:
        result = process_employee_feedback(
            test_text,
            model_type="bert",
            threshold=0.5
        )

        print_result(result)

    except Exception as error:
        print("\nPipeline Error:")
        print(error)