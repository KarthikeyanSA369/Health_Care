from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models and scalers
with open("model/diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)

with open("model/diabetes_scaler.pkl", "rb") as f:
    diabetes_scaler = pickle.load(f)

with open("model/heart_model.pkl", "rb") as f:
    heart_model = pickle.load(f)

with open("model/heart_scaler.pkl", "rb") as f:
    heart_scaler = pickle.load(f)

with open("model/obesity_model.pkl", "rb") as f:
    obesity_model = pickle.load(f)

with open("model/obesity_scaler.pkl", "rb") as f:
    obesity_scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/hello")
def hello():
    return "WORKING"


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/obesity")
def obesity():
    return render_template("obesity.html")


# ---------------- DIABETES ----------------
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]

        values = [float(request.form[f]) for f in features]
        data = np.array(values).reshape(1, -1)

        data_scaled = diabetes_scaler.transform(data)

        pred = diabetes_model.predict(data_scaled)[0]
        prob = diabetes_model.predict_proba(data_scaled)[0][1] * 100

        result = "Diabetic" if pred == 1 else "Non-Diabetic"

        return render_template(
            "result.html",
            prediction=f"{result} ({prob:.2f}% confidence)"
        )

    except Exception as e:
        return render_template(
            "error.html",
            error_message=f"Error in Diabetes Check: {str(e)}"
        )


# ---------------- HEART ----------------
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        features = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ]

        values = [float(request.form[f]) for f in features]
        data = np.array(values).reshape(1, -1)

        data_scaled = heart_scaler.transform(data)

        pred = heart_model.predict(data_scaled)[0]
        prob = heart_model.predict_proba(data_scaled)[0][1] * 100

        result = "No Heart Disease" if pred == 1 else "Heart Disease"

        return render_template(
            "result.html",
            prediction=f"{result} ({prob:.2f}% confidence)"
        )

    except Exception as e:
        return render_template(
            "error.html",
            error_message=f"Error in Heart Check: {str(e)}"
        )


# ---------------- OBESITY ----------------
@app.route("/predict_obesity", methods=["POST"])
def predict_obesity():
    try:
        features = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

        values = [float(request.form[f]) for f in features]
        data = np.array(values).reshape(1, -1)

        data_scaled = obesity_scaler.transform(data)

        pred = obesity_model.predict(data_scaled)[0]
        prob = max(obesity_model.predict_proba(data_scaled)[0]) * 100

        classes = ["Obese", "Not Obese"]
        result = classes[pred] if pred < len(classes) else f"Class {pred}"

        return render_template(
            "result.html",
            prediction=f"{result} ({prob:.2f}% confidence)"
        )

    except Exception as e:
        return render_template(
            "error.html",
            error_message=f"Error in Obesity Check: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)

print(app.url_map)