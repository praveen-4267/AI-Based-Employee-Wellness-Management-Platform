import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


RAW_FILE = Path("data/milestone2/raw/en-annotated.tsv")
PROCESSED_DIR = Path("data/milestone2/processed")


# XED label numbers
EMOTION_LABELS = {
    1: "anger",
    2: "anticipation",
    3: "disgust",
    4: "fear",
    5: "joy",
    6: "sadness",
    7: "surprise",
    8: "trust"
}


# Emotions required by the project
REQUIRED_EMOTIONS = {
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise"
}


def read_xed_dataset():
    """Read the original XED TSV dataset."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    rows = []

    with open(RAW_FILE, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                continue

            text = parts[0].strip()
            label_string = parts[1].strip()

            if not text or not label_string:
                continue

            try:
                labels = [
                    int(label)
                    for label in label_string.split(",")
                ]
            except ValueError:
                continue

            emotion_names = [
                EMOTION_LABELS[label]
                for label in labels
                if label in EMOTION_LABELS
            ]

            rows.append({
                "text": text,
                "labels": emotion_names
            })

    return pd.DataFrame(rows)


def create_single_label_dataset(df):
    """
    Create a single-label dataset for BERT
    and DistilBERT classification.
    """

    rows = []

    for _, row in df.iterrows():

        labels = row["labels"]

        # Keep only examples containing exactly
        # one of the six required emotions.
        if len(labels) == 1 and labels[0] in REQUIRED_EMOTIONS:

            rows.append({
                "text": row["text"],
                "label": labels[0]
            })

    result = pd.DataFrame(rows)

    result = result.drop_duplicates(
        subset=["text"]
    )

    return result


def create_multi_label_dataset(df):
    """
    Create a true multi-label dataset.

    Examples containing only the six required
    emotions are retained.
    """

    rows = []

    for _, row in df.iterrows():

        labels = row["labels"]

        if not labels:
            continue

        # All labels must belong to our six
        # project emotions.
        if all(label in REQUIRED_EMOTIONS for label in labels):

            rows.append({
                "text": row["text"],
                "labels": "|".join(sorted(set(labels)))
            })

    result = pd.DataFrame(rows)

    result = result.drop_duplicates(
        subset=["text"]
    )

    return result


def create_train_validation_test(df):
    """Create stratified train, validation and test sets."""

    train_df, temporary_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=42,
        stratify=temporary_df["label"]
    )

    return train_df, validation_df, test_df


def save_datasets(
    train_df,
    validation_df,
    test_df,
    multi_label_df
):
    """Save all processed datasets."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    multi_label_df.to_csv(
        PROCESSED_DIR / "multilabel.csv",
        index=False
    )


def main():

    print("=" * 50)
    print("XED Emotion Dataset Preparation")
    print("=" * 50)

    print("\nReading dataset...")

    df = read_xed_dataset()

    print("Total annotated rows:", len(df))

    print("\nCreating single-label dataset...")

    single_label_df = create_single_label_dataset(df)

    print(
        "Single-label samples:",
        len(single_label_df)
    )

    print("\nSingle-label emotion distribution:")

    print(
        single_label_df["label"]
        .value_counts()
        .sort_index()
    )

    if single_label_df.empty:
        print("\nERROR: No single-label data found.")
        return

    print("\nCreating train/validation/test split...")

    train_df, validation_df, test_df = (
        create_train_validation_test(
            single_label_df
        )
    )

    print("Training samples:", len(train_df))
    print("Validation samples:", len(validation_df))
    print("Test samples:", len(test_df))

    print("\nCreating multi-label dataset...")

    multi_label_df = create_multi_label_dataset(df)

    print(
        "Multi-label samples:",
        len(multi_label_df)
    )

    save_datasets(
        train_df,
        validation_df,
        test_df,
        multi_label_df
    )

    print("\nDatasets saved successfully!")

    print("\nOutput files:")

    print("data/milestone2/processed/train.csv")
    print("data/milestone2/processed/validation.csv")
    print("data/milestone2/processed/test.csv")
    print("data/milestone2/processed/multilabel.csv")

    print("\n" + "=" * 50)
    print("Dataset preparation completed")
    print("=" * 50)


if __name__ == "__main__":
    main()  