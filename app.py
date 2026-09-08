from flask import Flask, render_template, request

from src.full_pipeline import process_employee_feedback


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None
    feedback = ""

    if request.method == "POST":

        feedback = request.form.get("feedback", "").strip()

        model_type = request.form.get("model", "bert")

        try:

            if not feedback:
                raise ValueError("Please enter employee feedback.")

            result = process_employee_feedback(
                feedback,
                model_type=model_type,
                threshold=0.5
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error,
        feedback=feedback
    )


if __name__ == "__main__":
    app.run(debug=True)