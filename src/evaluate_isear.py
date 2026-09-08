from pathlib import Path

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

ISEAR_FILE = Path(
    "data/milestone2/isear/raw/"
    "ISEAR_dataset_complete.csv"
)

MODEL_DIR = Path(
    "src/models/bert_emotion"
)

OUTPUT_DIR = Path(
    "data/milestone2/isear"
)


# --------------------------------------------------
# Project emotion labels
# --------------------------------------------------

PROJECT_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise"
]


# ISEAR contains these seven emotions.
# Surprise is not part of ISEAR.
ISEAR_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "shame",
    "guilt"
]


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Load ISEAR
# --------------------------------------------------

def load_isear():

    print("\nLoading ISEAR dataset...")

    if not ISEAR_FILE.exists():

        raise FileNotFoundError(
            f"ISEAR file not found: {ISEAR_FILE}"
        )

    df = pd.read_csv(
        ISEAR_FILE
    )

    print(
        "Total ISEAR samples:",
        len(df)
    )

    print(
        "\nColumns:",
        list(df.columns)
    )

    return df


# --------------------------------------------------
# Prepare held-out benchmark
# --------------------------------------------------

def prepare_benchmark(df):

    print(
        "\nPreparing held-out ISEAR benchmark..."
    )

    # Rename columns if necessary.
    # The downloaded dataset normally contains
    # emotion and content.
    column_names = {
        column.lower(): column
        for column in df.columns
    }

    emotion_column = column_names.get(
        "emotion"
    )

    text_column = column_names.get(
        "content"
    )

    if emotion_column is None:

        raise ValueError(
            "Could not find the 'emotion' column."
        )

    if text_column is None:

        raise ValueError(
            "Could not find the 'content' column."
        )

    benchmark = df[
        [emotion_column, text_column]
    ].copy()

    benchmark.columns = [
        "emotion",
        "text"
    ]

    benchmark["emotion"] = (
        benchmark["emotion"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    benchmark["text"] = (
        benchmark["text"]
        .astype(str)
        .str.strip()
    )

    # Keep only the five emotions that overlap
    # with the project model.
    benchmark = benchmark[
        benchmark["emotion"].isin(
            [
                "anger",
                "disgust",
                "fear",
                "joy",
                "sadness"
            ]
        )
    ]

    # Remove empty text.
    benchmark = benchmark[
        benchmark["text"] != ""
    ]

    # Remove duplicate text.
    benchmark = benchmark.drop_duplicates(
        subset=["text"]
    )

    # Use a fixed held-out sample.
    benchmark = benchmark.sample(
        n=min(1000, len(benchmark)),
        random_state=42
    ).reset_index(drop=True)

    print(
        "Held-out benchmark samples:",
        len(benchmark)
    )

    print(
        "\nEmotion distribution:"
    )

    print(
        benchmark["emotion"]
        .value_counts()
        .sort_index()
    )

    return benchmark


# --------------------------------------------------
# Load BERT
# --------------------------------------------------

def load_bert():

    print(
        "\nLoading trained BERT model..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)
    model.eval()

    print(
        "BERT loaded successfully."
    )

    return tokenizer, model


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict(
    text,
    tokenizer,
    model
):

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

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    predicted_id = torch.argmax(
        probabilities
    ).item()

    predicted_label = (
        model.config.id2label[
            predicted_id
        ].lower()
    )

    confidence = (
        probabilities[
            predicted_id
        ].item()
    )

    return predicted_label, confidence


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

def evaluate(
    benchmark,
    tokenizer,
    model
):

    print(
        "\nGenerating BERT predictions..."
    )

    actual = []
    predicted = []
    confidences = []

    for index, row in benchmark.iterrows():

        prediction, confidence = predict(
            row["text"],
            tokenizer,
            model
        )

        actual.append(
            row["emotion"]
        )

        predicted.append(
            prediction
        )

        confidences.append(
            confidence
        )

        if (index + 1) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(benchmark)}"
            )

    benchmark["predicted_emotion"] = (
        predicted
    )

    benchmark["confidence"] = (
        confidences
    )

    benchmark["correct"] = (
        benchmark["emotion"]
        == benchmark["predicted_emotion"]
    )

    return benchmark


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_metrics(
    benchmark
):

    actual = benchmark[
        "emotion"
    ]

    predicted = benchmark[
        "predicted_emotion"
    ]

    labels = [
        "anger",
        "disgust",
        "fear",
        "joy",
        "sadness"
    ]

    accuracy = accuracy_score(
        actual,
        predicted
    )

    precision = precision_score(
        actual,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1
    }


# --------------------------------------------------
# Emotion-wise analysis
# --------------------------------------------------

def emotion_wise_results(
    benchmark
):

    rows = []

    for emotion in [
        "anger",
        "disgust",
        "fear",
        "joy",
        "sadness"
    ]:

        subset = benchmark[
            benchmark["emotion"]
            == emotion
        ]

        correct = subset[
            "correct"
        ].sum()

        total = len(subset)

        accuracy = (
            correct / total
            if total > 0
            else 0
        )

        average_confidence = (
            subset["confidence"].mean()
            if total > 0
            else 0
        )

        rows.append({
            "emotion": emotion,
            "samples": total,
            "correct": int(correct),
            "incorrect": int(
                total - correct
            ),
            "accuracy": accuracy,
            "average_confidence":
                average_confidence
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("TASK 6 - ISEAR BENCHMARK")
    print("=" * 60)

    print(
        "\nDevice:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "\nISEAR emotion categories:"
    )

    print(
        ", ".join(ISEAR_EMOTIONS)
    )

    print(
        "\nProject emotion categories:"
    )

    print(
        ", ".join(PROJECT_EMOTIONS)
    )

    print(
        "\nNote: Surprise is not represented "
        "in ISEAR."
    )

    df = load_isear()

    benchmark = prepare_benchmark(
        df
    )

    tokenizer, model = load_bert()

    benchmark = evaluate(
        benchmark,
        tokenizer,
        model
    )

    metrics = calculate_metrics(
        benchmark
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "OVERALL ISEAR RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nAccuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        "\nEmotion-wise results:"
    )

    emotion_results = (
        emotion_wise_results(
            benchmark
        )
    )

    print(
        emotion_results.to_string(
            index=False
        )
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            benchmark["emotion"],
            benchmark["predicted_emotion"],
            labels=[
                "anger",
                "disgust",
                "fear",
                "joy",
                "sadness"
            ],
            target_names=[
                "anger",
                "disgust",
                "fear",
                "joy",
                "sadness"
            ],
            digits=4,
            zero_division=0
        )
    )

    incorrect = benchmark[
        benchmark["correct"] == False
    ]

    print(
        "\nIncorrect predictions:",
        len(incorrect)
    )

    print(
        "\nAverage confidence:",
        f"{benchmark['confidence'].mean() * 100:.2f}%"
    )

    # Save benchmark results
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    benchmark_file = (
        OUTPUT_DIR /
        "isear_benchmark_predictions.csv"
    )

    metrics_file = (
        OUTPUT_DIR /
        "isear_metrics.csv"
    )

    benchmark.to_csv(
        benchmark_file,
        index=False
    )

    metrics_df = pd.DataFrame([
        {
            "metric": "accuracy",
            "value": metrics["accuracy"]
        },
        {
            "metric": "precision_macro",
            "value": metrics["precision"]
        },
        {
            "metric": "recall_macro",
            "value": metrics["recall"]
        },
        {
            "metric": "macro_f1",
            "value": metrics["macro_f1"]
        }
    ])

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    print(
        "\nSaved prediction results to:"
    )

    print(
        benchmark_file
    )

    print(
        "\nSaved metrics to:"
    )

    print(
        metrics_file
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "TASK 6 ISEAR BENCHMARK COMPLETED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()