import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_NAME = "google-bert/bert-base-uncased"

DATA_DIR = Path("data/milestone2/processed")
MODEL_DIR = Path("src/models/bert_emotion")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5

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


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("BERT Emotion Classification Training")
print("=" * 60)

print("\nDevice:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. DATASET CLASS
# ============================================================

class EmotionDataset(Dataset):

    def __init__(self, dataframe, tokenizer):

        self.texts = dataframe["text"].tolist()

        self.labels = [
            LABEL_TO_ID[label]
            for label in dataframe["label"]
        ]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
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
            label,
            dtype=torch.long
        )

        return item


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\nLoading datasets...")

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}"
    )

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found: {VALIDATION_FILE}"
    )


train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print("Training samples:", len(train_df))
print("Validation samples:", len(validation_df))


# ============================================================
# 5. LOAD TOKENIZER
# ============================================================

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")


# ============================================================
# 6. CREATE DATASETS
# ============================================================

train_dataset = EmotionDataset(
    train_df,
    tokenizer
)

validation_dataset = EmotionDataset(
    validation_df,
    tokenizer
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


# ============================================================
# 7. LOAD PRETRAINED BERT
# ============================================================

print("\nLoading pretrained BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTIONS),
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID
)

model.to(device)

print("BERT model loaded successfully.")


# ============================================================
# 8. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 9. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        correct += (
            predictions == batch["labels"]
        ).sum().item()

        total += batch["labels"].size(0)

        if (step + 1) % 100 == 0:

            print(
                f"Step {step + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# 10. VALIDATION FUNCTION
# ============================================================

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

            loss = outputs.loss

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            correct += (
                predictions == batch["labels"]
            ).sum().item()

            total += batch["labels"].size(0)

    average_loss = total_loss / len(
        validation_loader
    )

    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# 11. TRAIN BERT
# ============================================================

print("\nStarting BERT fine-tuning...")

best_validation_accuracy = 0.0

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    train_loss, train_accuracy = train_one_epoch()

    validation_loss, validation_accuracy = evaluate()

    print("\nTraining Loss:",
          round(train_loss, 4))

    print("Training Accuracy:",
          round(train_accuracy, 4))

    print("Validation Loss:",
          round(validation_loss, 4))

    print("Validation Accuracy:",
          round(validation_accuracy, 4))

    # Save the best model
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        with open(
            MODEL_DIR / "label_mapping.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "label_to_id": LABEL_TO_ID,
                    "id_to_label": ID_TO_LABEL
                },
                file,
                indent=4
            )

        print("\nBest BERT model saved!")


# ============================================================
# 12. SAMPLE PREDICTION
# ============================================================

def predict_emotion(text):

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

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    predicted_id = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0, predicted_id
    ].item()

    return (
        ID_TO_LABEL[predicted_id],
        confidence
    )


# ============================================================
# 13. TEST SAMPLE
# ============================================================

sample_texts = [
    "I am extremely happy and excited about my new job.",
    "I feel angry because my manager treated me unfairly.",
    "I am scared about losing my job.",
    "I am very sad and disappointed.",
    "I am surprised by the unexpected result.",
    "I feel disgusted by this terrible situation."
]

print("\n" + "=" * 60)
print("BERT SAMPLE PREDICTIONS")
print("=" * 60)

for text in sample_texts:

    emotion, confidence = predict_emotion(text)

    print("\nText:", text)
    print("Predicted emotion:", emotion)
    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )


print("\n" + "=" * 60)
print("BERT TRAINING COMPLETED")
print("=" * 60)

print(
    "\nBest validation accuracy:",
    f"{best_validation_accuracy * 100:.2f}%"
)

print(
    "\nSaved model location:",
    MODEL_DIR
)