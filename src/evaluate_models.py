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
    classification_report,
    confusion_matrix
)


TEST_FILE = Path(
    "data/milestone2/processed/test.csv"
)

BERT_DIR = Path(
    "src/models/bert_emotion"
)

DISTILBERT_DIR = Path(
    "src/models/distilbert_emotion"
)


EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise"
]


LABEL_TO_ID = {
    emotion: index
    for index, emotion in enumerate(EMOTIONS)
}


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model(model_dir):

    print(f"\nLoading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    )

    model.to(device)
    model.eval()

    return tokenizer, model


def predict_dataset(
    dataframe,
    tokenizer,
    model
):

    predictions = []

    print(
        "\nGenerating predictions for",
        len(dataframe),
        "test samples..."
    )

    for index, text in enumerate(
        dataframe["text"]
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

            predicted_id = torch.argmax(
                outputs.logits,
                dim=1
            ).item()

        predictions.append(
            model.config.id2label.get(
                predicted_id,
                EMOTIONS[predicted_id]
            ).lower()
        )

        if (index + 1) % 200 == 0:

            print(
                f"Processed {index + 1}/"
                f"{len(dataframe)}"
            )

    return predictions


def calculate_metrics(
    actual,
    predicted
):

    accuracy = accuracy_score(
        actual,
        predicted
    )

    precision = precision_score(
        actual,
        predicted,
        labels=EMOTIONS,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        labels=EMOTIONS,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=EMOTIONS,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1
    }


def evaluate_model(
    model_name,
    model_dir,
    test_df
):

    print("\n" + "=" * 60)
    print(f"{model_name} EVALUATION")
    print("=" * 60)

    tokenizer, model = load_model(
        model_dir
    )

    predictions = predict_dataset(
        test_df,
        tokenizer,
        model
    )

    actual = [
        label.lower()
        for label in test_df["label"]
    ]

    metrics = calculate_metrics(
        actual,
        predictions
    )

    print("\nOverall metrics:")

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"Macro F1 : {metrics['macro_f1']:.4f}"
    )

    print("\nDetailed classification report:")

    print(
        classification_report(
            actual,
            predictions,
            labels=EMOTIONS,
            target_names=EMOTIONS,
            digits=4,
            zero_division=0
        )
    )

    print("Confusion matrix:")

    matrix = confusion_matrix(
        actual,
        predictions,
        labels=EMOTIONS
    )

    print(
        pd.DataFrame(
            matrix,
            index=EMOTIONS,
            columns=EMOTIONS
        )
    )

    return metrics, predictions


def save_results(
    bert_metrics,
    distilbert_metrics
):

    results = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "Macro F1"
        ],

        "BERT": [
            bert_metrics["accuracy"],
            bert_metrics["precision"],
            bert_metrics["recall"],
            bert_metrics["macro_f1"]
        ],

        "DistilBERT": [
            distilbert_metrics["accuracy"],
            distilbert_metrics["precision"],
            distilbert_metrics["recall"],
            distilbert_metrics["macro_f1"]
        ]
    })

    output_file = Path(
        "data/milestone2/processed/"
        "model_evaluation.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        "\nResults saved to:",
        output_file
    )


def main():

    print("=" * 60)
    print("TASK 5 - BERT vs DISTILBERT EVALUATION")
    print("=" * 60)

    print("\nDevice:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        "\nTest samples:",
        len(test_df)
    )

    print("\nTest emotion distribution:")

    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    bert_metrics, bert_predictions = (
        evaluate_model(
            "BERT",
            BERT_DIR,
            test_df
        )
    )

    distilbert_metrics, distilbert_predictions = (
        evaluate_model(
            "DistilBERT",
            DISTILBERT_DIR,
            test_df
        )
    )

    print("\n" + "=" * 60)
    print("BERT vs DISTILBERT COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "Macro F1"
        ],

        "BERT": [
            bert_metrics["accuracy"],
            bert_metrics["precision"],
            bert_metrics["recall"],
            bert_metrics["macro_f1"]
        ],

        "DistilBERT": [
            distilbert_metrics["accuracy"],
            distilbert_metrics["precision"],
            distilbert_metrics["recall"],
            distilbert_metrics["macro_f1"]
        ]

    })

    print(
        comparison.to_string(
            index=False
        )
    )

    save_results(
        bert_metrics,
        distilbert_metrics
    )

    print("\n" + "=" * 60)
    print("TASK 5 EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()