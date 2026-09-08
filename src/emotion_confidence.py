import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Saved multi-label model
MODEL_DIR = Path(
    "src/models/multilabel_emotion"
)

# Default threshold
DEFAULT_THRESHOLD = 0.50

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise"
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)
    model.eval()

    return tokenizer, model


def load_threshold():

    mapping_file = (
        MODEL_DIR / "label_mapping.json"
    )

    if mapping_file.exists():

        with open(
            mapping_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return float(
            data.get(
                "confidence_threshold",
                DEFAULT_THRESHOLD
            )
        )

    return DEFAULT_THRESHOLD


def predict_emotions(
    text,
    tokenizer,
    model,
    threshold
):

    if text is None:
        return False, "Text is empty."

    if not isinstance(text, str):
        return False, "Input must be text."

    if not text.strip():
        return False, "Text is empty."

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    encoding = {
        key: value.to(device)
        for key, value in encoding.items()
    }

    with torch.no_grad():

        outputs = model(**encoding)

        probabilities = torch.sigmoid(
            outputs.logits
        )[0]

    emotion_probabilities = {}

    for index, emotion in enumerate(EMOTIONS):

        emotion_probabilities[emotion] = (
            probabilities[index].item()
        )

    predicted_emotions = [
        emotion
        for emotion, probability
        in emotion_probabilities.items()
        if probability >= threshold
    ]

    # Primary emotion = highest probability
    primary_emotion = max(
        emotion_probabilities,
        key=emotion_probabilities.get
    )

    primary_confidence = (
        emotion_probabilities[primary_emotion]
    )

    # If nothing crosses the threshold,
    # still return the strongest emotion.
    if not predicted_emotions:

        predicted_emotions = [
            primary_emotion
        ]

    result = {
        "input_text": text,
        "predicted_emotions": predicted_emotions,
        "primary_emotion": primary_emotion,
        "primary_confidence":
            primary_confidence,
        "threshold": threshold,
        "probabilities":
            emotion_probabilities
    }

    return True, result


def display_result(result):

    print("\n" + "=" * 60)
    print("EMOTION CONFIDENCE RESULT")
    print("=" * 60)

    print("\nInput:")
    print(result["input_text"])

    print("\nEmotion probabilities:")

    for emotion, probability in (
        result["probabilities"].items()
    ):

        print(
            f"  {emotion:10s}: "
            f"{probability * 100:.2f}%"
        )

    print("\nConfidence threshold:")
    print(
        f"{result['threshold'] * 100:.0f}%"
    )

    print("\nPredicted emotions:")

    print(
        ", ".join(
            result["predicted_emotions"]
        )
    )

    print("\nPrimary emotion:")
    print(result["primary_emotion"])

    print("\nPrimary confidence:")
    print(
        f"{result['primary_confidence'] * 100:.2f}%"
    )


def main():

    print("=" * 60)
    print("Task 4 - Emotion Confidence Scoring")
    print("=" * 60)

    print("\nDevice:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("\nLoading trained multi-label model...")

    tokenizer, model = load_model()

    threshold = load_threshold()

    print(
        "Confidence threshold:",
        f"{threshold * 100:.0f}%"
    )

    test_samples = [

        "I am extremely happy about my new job.",

        "I am scared and worried about losing my job.",

        "I am angry because my manager treated me unfairly.",

        "I am sad about the situation.",

        "I am shocked and surprised by the result.",

        "I am excited about the opportunity, but I am also nervous.",

        "I am happy with my team but afraid of the upcoming deadline."

    ]

    for text in test_samples:

        valid, result = predict_emotions(
            text,
            tokenizer,
            model,
            threshold
        )

        if valid:

            display_result(result)

        else:

            print(
                "\nInvalid input:",
                result
            )

    print("\n" + "=" * 60)
    print("TASK 4 CONFIDENCE TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()