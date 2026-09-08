import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "google-bert/bert-base-uncased"

DATA_FILE = Path(
    "data/milestone2/processed/multilabel.csv"
)

MODEL_DIR = Path(
    "src/models/multilabel_emotion"
)

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5

# Configurable confidence threshold
CONFIDENCE_THRESHOLD = 0.50

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

ID_TO_LABEL = {
    index: emotion
    for emotion, index in LABEL_TO_ID.items()
}


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("BERT Multi-Label Emotion Classification")
print("=" * 60)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


class MultiLabelEmotionDataset(Dataset):

    def __init__(self, dataframe, tokenizer):

        self.texts = dataframe["text"].tolist()
        self.labels = []

        for label_string in dataframe["labels"]:

            label_vector = [
                0.0
                for _ in EMOTIONS
            ]

            labels = label_string.split("|")

            for label in labels:

                if label in LABEL_TO_ID:

                    label_vector[
                        LABEL_TO_ID[label]
                    ] = 1.0

            self.labels.append(label_vector)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.float
        )

        return item


print("\nLoading multi-label dataset...")

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print(
    "Multi-label samples:",
    len(df)
)


print("\nPreparing dataset...")

dataset = MultiLabelEmotionDataset(
    df,
    None
)


print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")


dataset = MultiLabelEmotionDataset(
    df,
    tokenizer
)


# Split dataset into training and validation data
total_size = len(dataset)

validation_size = int(
    total_size * 0.20
)

training_size = (
    total_size - validation_size
)

train_dataset, validation_dataset = (
    torch.utils.data.random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(42)
    )
)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nLoading pretrained BERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTIONS),
    problem_type="multi_label_classification",
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID
)

model.to(device)

print(
    "Multi-label BERT loaded successfully."
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


def calculate_accuracy(logits, labels):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= CONFIDENCE_THRESHOLD
    ).float()

    correct = (
        predictions == labels
    ).all(dim=1).sum().item()

    total = labels.size(0)

    return correct, total


def train_one_epoch():

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for step, batch in enumerate(
        train_loader
    ):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        batch_correct, batch_total = (
            calculate_accuracy(
                outputs.logits,
                batch["labels"]
            )
        )

        correct += batch_correct
        total += batch_total

        if (step + 1) % 100 == 0:

            print(
                f"Step {step + 1}/"
                f"{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    return (
        total_loss / len(train_loader),
        correct / total
    )


def evaluate():

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in validation_loader:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            total_loss += outputs.loss.item()

            batch_correct, batch_total = (
                calculate_accuracy(
                    outputs.logits,
                    batch["labels"]
                )
            )

            correct += batch_correct
            total += batch_total

    return (
        total_loss / len(validation_loader),
        correct / total
    )


print("\nStarting multi-label fine-tuning...")

best_validation_accuracy = 0.0

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )
    print("=" * 60)

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    validation_loss, validation_accuracy = (
        evaluate()
    )

    print(
        "\nTraining Loss:",
        round(train_loss, 4)
    )

    print(
        "Training Exact-Match Accuracy:",
        round(train_accuracy, 4)
    )

    print(
        "Validation Loss:",
        round(validation_loss, 4)
    )

    print(
        "Validation Exact-Match Accuracy:",
        round(validation_accuracy, 4)
    )

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save_pretrained(
            MODEL_DIR
        )

        tokenizer.save_pretrained(
            MODEL_DIR
        )

        with open(
            MODEL_DIR / "label_mapping.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "label_to_id": LABEL_TO_ID,
                    "id_to_label": ID_TO_LABEL,
                    "confidence_threshold":
                        CONFIDENCE_THRESHOLD
                },
                file,
                indent=4
            )

        print(
            "\nBest multi-label model saved!"
        )


def predict_emotions(text):

    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
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

    results = []

    for index, probability in enumerate(
        probabilities
    ):

        results.append({
            "emotion":
                ID_TO_LABEL[index],

            "confidence":
                probability.item()
        })

    predicted_emotions = [
        result["emotion"]
        for result in results
        if result["confidence"]
        >= CONFIDENCE_THRESHOLD
    ]

    # Always provide the strongest emotion
    # if no emotion crosses the threshold.
    if not predicted_emotions:

        strongest = max(
            results,
            key=lambda item:
                item["confidence"]
        )

        predicted_emotions = [
            strongest["emotion"]
        ]

    return results, predicted_emotions


sample_texts = [

    "I am extremely happy and excited about my new job.",

    "I am scared and worried about losing my job.",

    "I am angry because my manager treated me unfairly.",

    "I feel sad and disappointed about my work.",

    "I am shocked and surprised by the unexpected result.",

    "I feel disgusted and angry about this terrible situation.",

    "I am excited about the opportunity, but I am also nervous.",

    "I am happy with my team but afraid of the upcoming deadline."
]


print("\n" + "=" * 60)
print("MULTI-LABEL SAMPLE PREDICTIONS")
print("=" * 60)

for text in sample_texts:

    results, predicted_emotions = (
        predict_emotions(text)
    )

    print("\nText:", text)

    print(
        "Predicted emotions:",
        ", ".join(predicted_emotions)
    )

    print("Emotion probabilities:")

    for result in results:

        print(
            f"  {result['emotion']:10s}: "
            f"{result['confidence'] * 100:.2f}%"
        )


print("\n" + "=" * 60)
print("MULTI-LABEL TRAINING COMPLETED")
print("=" * 60)

print(
    "\nConfidence threshold:",
    CONFIDENCE_THRESHOLD
)

print(
    "Best validation exact-match accuracy:",
    f"{best_validation_accuracy * 100:.2f}%"
)

print(
    "\nSaved model location:",
    MODEL_DIR
)