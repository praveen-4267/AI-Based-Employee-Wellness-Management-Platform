from src.ingestion import (
    read_direct_text,
    read_txt_file,
    read_csv_file
)


print("========== TASK 1: TEXT INGESTION TEST ==========")


# 1. Direct text test
print("\n1. DIRECT TEXT")

text = "I am happy with my work environment."

valid, result = read_direct_text(text)

print("Status:", "VALID" if valid else "INVALID")
print("Result:", result)


# 2. TXT file test
print("\n2. TXT FILE")

valid, result = read_txt_file("data/sample.txt")

print("Status:", "VALID" if valid else "INVALID")

if valid:
    print("Text:")
    print(result)
else:
    print("Error:", result)


# 3. CSV file test
print("\n3. CSV FILE")

valid, result = read_csv_file("data/sample.csv")

print("Status:", "VALID" if valid else "INVALID")

if valid:
    print("Feedback records:")
    for feedback in result:
        print("-", feedback)
else:
    print("Error:", result)
    # 4. Empty text test
print("\n4. EMPTY TEXT")

valid, result = read_direct_text("")

print("Status:", "VALID" if valid else "INVALID")
print("Result:", result)


# 5. Spaces-only text test
print("\n5. SPACES ONLY")

valid, result = read_direct_text("     ")

print("Status:", "VALID" if valid else "INVALID")
print("Result:", result)


# 6. Invalid file format test
print("\n6. INVALID FILE FORMAT")

valid, result = read_txt_file("data/sample.csv")

print("Status:", "VALID" if valid else "INVALID")
print("Result:", result)


# 7. Missing file test
print("\n7. MISSING FILE")

valid, result = read_txt_file("data/not_found.txt")

print("Status:", "VALID" if valid else "INVALID")
print("Result:", result)
# 8. CSV without feedback column
print("\n8. CSV WITHOUT FEEDBACK COLUMN")

valid, result = read_csv_file("data/invalid.csv")

print("Status:", "VALID" if valid else "INVALID")
print("Result:", result)
    # ==========================================
# TASK 2 - PREPROCESSING TEST
# ==========================================

from src.preprocessing import preprocess_text


print("\n========== TASK 2: PREPROCESSING TEST ==========")

test_text = "I am VERY happy!!!   with my work @company 😊"

valid, result = preprocess_text(test_text)

if valid:

    print("\nOriginal Text:")
    print(result["original"])

    print("\nLowercase:")
    print(result["lowercase"])

    print("\nNoise Filtered:")
    print(result["noise_filtered"])

    print("\nCleaned Text:")
    print(result["cleaned"])

    print("\nTokens:")
    print(result["tokens"])

    print("\nAfter Stop-word Removal:")
    print(result["without_stopwords"])

    print("\nAfter Lemmatization:")
    print(result["lemmatized"])

    print("\nFinal Processed Text:")
    print(result["processed_text"])

else:

    print("\nStatus: INVALID")
    print("Error:", result)
# ==========================================
# TASK 2 - PREPROCESSING VALIDATION TESTS
# ==========================================

print("\n==========================================")
print("TASK 2 - VALIDATION TESTS")
print("==========================================")


def run_test(test_number, test_name, text, expected=None):
    """
    Run a preprocessing test and compare
    the actual result with the expected result.
    """

    valid, result = preprocess_text(text)

    print(f"\nTEST {test_number} - {test_name}")
    print("Input:", repr(text))

    # Test for invalid/empty input
    if expected is None:

        if not valid:
            print("Expected: Invalid input")
            print("Actual:", result)
            print("Status: PASS")
        else:
            print("Expected: Invalid input")
            print("Actual: Valid input")
            print("Status: FAIL")

        return

    # Test for valid input
    if valid:

        actual = result["processed_text"]

        print("Expected:", expected)
        print("Actual:", actual)

        if actual == expected:
            print("Status: PASS")
        else:
            print("Status: FAIL")

    else:

        print("Expected:", expected)
        print("Actual: Processing failed")
        print("Error:", result)
        print("Status: FAIL")


# Test 1 - Tokenization
run_test(
    1,
    "TOKENIZATION",
    "I am happy with my job.",
    "happy job"
)


# Test 2 - Stop-word removal
run_test(
    2,
    "STOP-WORD REMOVAL",
    "I am very happy with my job.",
    "happy job"
)


# Test 3 - Lemmatization
run_test(
    3,
    "LEMMATIZATION",
    "Employees are working on different projects.",
    "employee working different project"
)


# Test 4 - Noise filtering
run_test(
    4,
    "NOISE FILTERING",
    "I am happy with my job. Visit https://example.com",
    "happy job visit"
)


# Test 5 - Special characters
run_test(
    5,
    "SPECIAL CHARACTERS",
    "I am happy @work #employee $100 😊",
    "happy work employee"
)


# Test 6 - Punctuation
run_test(
    6,
    "PUNCTUATION",
    "I am happy!!! Is my work good?",
    "happy work good"
)


# Test 7 - Empty text
run_test(
    7,
    "EMPTY TEXT",
    "",
    None
)


# Test 8 - Repeated spaces
run_test(
    8,
    "REPEATED SPACES",
    "I     am      very     happy.",
    "happy"
)


# Test 9 - Short text
run_test(
    9,
    "SHORT TEXT",
    "Happy",
    "happy"
)


# Test 10 - Long text
long_text = """
I am working in the organization and I generally enjoy my work.
My team is supportive and the working environment is good.
However, sometimes the workload becomes high and I feel stressed
because of long working hours and multiple deadlines.
"""

run_test(
    10,
    "LONG TEXT",
    long_text,
    "working organization generally enjoy work team supportive working environment good however sometimes workload becomes high feel stressed long working hour multiple deadline"
)
# ==========================================
# TASK 3 - VADER SENTIMENT TEST
# ==========================================

from src.sentiment import analyze_sentiment


print("\n==========================================")
print("TASK 3 - VADER SENTIMENT TEST")
print("==========================================")


test_text = "I really enjoy my work and my team is excellent."

valid, result = analyze_sentiment(test_text)


if valid:

    print("\nInput Text:")
    print(result["text"])

    print("\nPositive Score:")
    print(result["positive"])

    print("\nNegative Score:")
    print(result["negative"])

    print("\nNeutral Score:")
    print(result["neutral"])

    print("\nCompound Score:")
    print(result["compound"])

    print("\nSentiment:")
    print(result["sentiment"])

else:

    print("\nStatus: INVALID")
    print("Error:", result)
    # ==========================================
# TASK 3 - VADER VALIDATION TESTS
# ==========================================

print("\n==========================================")
print("TASK 3 - VADER VALIDATION TESTS")
print("==========================================")


def run_sentiment_test(test_number, test_name, text, expected_sentiment):

    valid, result = analyze_sentiment(text)

    print(f"\nTEST {test_number} - {test_name}")
    print("Input:", text)

    if valid:

        print("Positive Score:", result["positive"])
        print("Negative Score:", result["negative"])
        print("Neutral Score:", result["neutral"])
        print("Compound Score:", result["compound"])
        print("Expected:", expected_sentiment)
        print("Actual:", result["sentiment"])

        if result["sentiment"] == expected_sentiment:
            print("Status: PASS")
        else:
            print("Status: FAIL")

    else:

        print("Error:", result)
        print("Status: FAIL")


# Test 1 - Positive sentiment
run_sentiment_test(
    1,
    "POSITIVE SENTIMENT",
    "I love my job and I am very happy with my team.",
    "Positive"
)


# Test 2 - Negative sentiment
run_sentiment_test(
    2,
    "NEGATIVE SENTIMENT",
    "I hate my job and I feel extremely stressed and unhappy.",
    "Negative"
)


# Test 3 - Neutral sentiment
run_sentiment_test(
    3,
    "NEUTRAL SENTIMENT",
    "I work in the software department.",
    "Neutral"
)


# Test 4 - Strong positive sentiment
run_sentiment_test(
    4,
    "STRONG POSITIVE",
    "This is an excellent and wonderful workplace.",
    "Positive"
)


# Test 5 - Strong negative sentiment
run_sentiment_test(
    5,
    "STRONG NEGATIVE",
    "This is a terrible and horrible work experience.",
    "Negative"
)


# Test 6 - Simple neutral statement
run_sentiment_test(
    6,
    "SIMPLE NEUTRAL",
    "My working hours are from nine to six.",
    "Neutral"
)
# ==========================================
# TASK 4 - INITIAL SENTIMENT REPORT
# ==========================================

from src.report import generate_sentiment_report


print("\n==========================================")
print("TASK 4 - INITIAL SENTIMENT REPORT")
print("==========================================")


# Read feedback from sample CSV
valid, feedback_list = read_csv_file("data/sample.csv")


if valid:

    report = generate_sentiment_report(feedback_list)

    print("\nNumber of analyzed samples:", len(report))

    for record in report:

        print("\n------------------------------------------")
        print("Sample:", record["sample_number"])

        print("\nInput Text:")
        print(record["input_text"])

        print("\nProcessed Text:")
        print(record["processed_text"])

        print("\nSentiment:")
        print(record["sentiment"])

        print("\nPositive Score:")
        print(record["positive_score"])

        print("\nNegative Score:")
        print(record["negative_score"])

        print("\nNeutral Score:")
        print(record["neutral_score"])

        print("\nCompound Score:")
        print(record["compound_score"])

else:

    print("Error reading CSV:", feedback_list)
    # ==========================================
# TASK 5 - COMPLETE PIPELINE INTEGRATION
# ==========================================

from src.integration import process_single_text


print("\n==========================================")
print("TASK 5 - COMPLETE PIPELINE TEST")
print("==========================================")


integration_tests = [
    "I am very happy with my job and my team.",
    "I am stressed and unhappy with my workload.",
    "My working hours are from nine to six."
]


for number, text in enumerate(integration_tests, start=1):

    print(f"\nTEST {number}")
    print("Input:", text)

    valid, result = process_single_text(text)

    if valid:

        print("Processed:", result["processed_text"])
        print("Sentiment:", result["sentiment"])
        print("Positive:", result["positive_score"])
        print("Negative:", result["negative_score"])
        print("Neutral:", result["neutral_score"])
        print("Compound:", result["compound_score"])
        print("Status: PASS")

    else:

        print("Error:", result)
        print("Status: FAIL")
