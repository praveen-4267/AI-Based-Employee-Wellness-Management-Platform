import csv
import os

from full_pipeline import process_employee_feedback


# ---------------------------------------------------------
# Edge cases required for Milestone 2 Task 8
# ---------------------------------------------------------

TEST_CASES = [
    (
        "Positive",
        "I am very happy and satisfied with my work today."
    ),

    (
        "Negative",
        "I am extremely frustrated and unhappy with my workload."
    ),

    (
        "Neutral",
        "The meeting is scheduled for tomorrow at 10 AM."
    ),

    (
        "Mixed emotions",
        "I am excited about the promotion but nervous about the new responsibilities."
    ),

    (
        "Very short",
        "Happy."
    ),

    (
        "Long text",
        (
            "I have been working on this project for several weeks and "
            "although there have been many challenges, I am learning new "
            "skills, communicating with my colleagues, solving problems, "
            "meeting deadlines, and trying to maintain a healthy balance "
            "between my professional responsibilities and personal life."
        )
    ),

    (
        "Informal language",
        "OMG this deadline is crazy lol, I'm so stressed rn."
    ),

    (
        "Emoji",
        "I got the promotion today! 😊🎉 I am really happy!"
    ),

    (
        "Ambiguous",
        "Everything is fine, I guess."
    ),

    (
        "Empty",
        ""
    ),

    (
        "Invalid input",
        12345
    )
]


def run_edge_case_tests():

    results = []

    print("\n")
    print("=" * 70)
    print("TASK 8 - EDGE CASE TESTING")
    print("=" * 70)

    for case_name, text in TEST_CASES:

        print("\n" + "-" * 70)
        print(f"Test Case: {case_name}")
        print(f"Input: {repr(text)}")

        try:

            result = process_employee_feedback(
                text,
                model_type="bert",
                threshold=0.5
            )

            print("Status: PASS")
            print(
                f"Primary Emotion: "
                f"{result['primary_emotion']}"
            )

            print(
                f"Primary Confidence: "
                f"{result['primary_confidence'] * 100:.2f}%"
            )

            print(
                f"VADER Sentiment: "
                f"{result['sentiment']}"
            )

            print(
                f"Multi-label Emotions: "
                f"{result['multi_label_emotions']}"
            )

            results.append({
                "test_case": case_name,
                "input": str(text),
                "status": "PASS",
                "primary_emotion":
                    result["primary_emotion"],
                "primary_confidence":
                    result["primary_confidence"],
                "sentiment":
                    result["sentiment"],
                "multi_label_emotions":
                    ", ".join(
                        result["multi_label_emotions"]
                    ),
                "error": ""
            })

        except Exception as error:

            print("Status: HANDLED")
            print(f"Error: {error}")

            results.append({
                "test_case": case_name,
                "input": str(text),
                "status": "HANDLED",
                "primary_emotion": "",
                "primary_confidence": "",
                "sentiment": "",
                "multi_label_emotions": "",
                "error": str(error)
            })

    return results


def save_results(results):

    output_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
        "data",
        "milestone2",
        "processed"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "edge_case_results.csv"
    )

    fieldnames = [
        "test_case",
        "input",
        "status",
        "primary_emotion",
        "primary_confidence",
        "sentiment",
        "multi_label_emotions",
        "error"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n")
    print("=" * 70)
    print("RESULT FILE SAVED")
    print("=" * 70)
    print(output_file)


if __name__ == "__main__":

    test_results = run_edge_case_tests()

    save_results(test_results)

    print("\n")
    print("=" * 70)
    print("TASK 8 TESTING COMPLETE")
    print("=" * 70)