import pandas as pd
from pathlib import Path


def validate_text(text):
    """Check whether text is valid and non-empty."""

    if text is None:
        return False, "Text is empty."

    if not isinstance(text, str):
        return False, "Input must be text."

    if not text.strip():
        return False, "Text is empty or contains only spaces."

    return True, "Valid text."


def read_direct_text(text):
    """Read and validate text entered directly by the user."""

    valid, message = validate_text(text)

    if valid:
        return True, text

    return False, message


def read_txt_file(file_path):
    """Read text from a .txt file."""

    path = Path(file_path)

    if path.suffix.lower() != ".txt":
        return False, "Invalid file format. Only .txt files are supported."

    if not path.exists():
        return False, "File not found."

    text = path.read_text(encoding="utf-8")

    valid, message = validate_text(text)

    if valid:
        return True, text

    return False, message


def read_csv_file(file_path):
    """Read employee feedback from a .csv file."""

    path = Path(file_path)

    if path.suffix.lower() != ".csv":
        return False, "Invalid file format. Only .csv files are supported."

    if not path.exists():
        return False, "File not found."

    try:
        df = pd.read_csv(path)

    except Exception as e:
        return False, f"Unable to read CSV file: {e}"

    if "feedback" not in df.columns:
        return False, "CSV must contain a 'feedback' column."

    feedback = df["feedback"].dropna().astype(str)

    feedback = feedback[feedback.str.strip() != ""]

    if feedback.empty:
        return False, "CSV contains no valid feedback text."

    return True, feedback.tolist()