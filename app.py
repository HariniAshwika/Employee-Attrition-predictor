from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model, columns, and encoders
with open("attrition_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get form data and convert to float
            input_values = [float(request.form.get(f'feature_{i}')) for i in range(len(feature_columns))]
            input_array = np.array([input_values])

            # Predict
            result = model.predict(input_array)[0]
            if result == 1:
                prediction = "⚠️ Yes - This employee is likely to leave."
            else:
                prediction = "✅ No - This employee is likely to stay."
            return render_template("index.html", prediction=prediction)
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)