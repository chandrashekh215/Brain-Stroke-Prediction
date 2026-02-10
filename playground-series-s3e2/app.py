from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.joblib")

# Load the model
stroke_model = joblib.load(model_path)

# Helper function for prediction
def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    encoded_cols, numeric_cols = stroke_model["encoded_cols"], stroke_model["numeric_cols"]
    preprocessor = stroke_model["preprocessor"]
    input_df[encoded_cols] = preprocessor.transform(input_df)
    X = input_df[numeric_cols + encoded_cols]
    prediction = stroke_model['model'].predict(X)
    return prediction


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form data
        gender = request.form["gender"].lower()
        age = int(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        ever_married = request.form["ever_married"].lower()
        work_type = request.form["work_type"]
        residence_type = request.form["residence_type"]
        avg_glucose_level = float(request.form["avg_glucose_level"])
        bmi = float(request.form["bmi"])
        smoking_status = request.form["smoking_status"].lower()

        # Map work type
        work_type_mapping = {
            "Government job": "Govt_job",
            "Children": "children",
            "Never Worked": "Never_worked",
            "Private": "Private",
        }

        single_input = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type_mapping.get(work_type, work_type),
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,
        }

        # Predict and redirect to result page
        prediction = predict_input(single_input)
        result = "Likely" if prediction[0] == 1 else "Not Likely"
        return render_template("result.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Flask app is starting...")
    print("="*60)
    print("📱 Open your browser and visit:")
    print("   → http://127.0.0.1:5000")
    print("   → http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
