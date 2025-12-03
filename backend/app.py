from flask import Flask, request, render_template
import numpy as np
import pickle

from flask_cors import CORS

# -------------------------------------------------------
# LOAD MODEL + ENCODERS + SCALER
# -------------------------------------------------------
with open("diet_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
label_encoders = model_data["label_encoders"]
target_encoder = model_data["target_encoder"]
feature_names = model_data["feature_names"]

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------------------------
# PREDICT PAGE
# -------------------------------------------------------
@app.route("/predict_form")
def predict_form():
    return render_template("frontend/prediction.html")


# -------------------------------------------------------
# PROCESS PREDICTION
# -------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -----------------------------
        # GET FORM DATA
        # -----------------------------
        user_name = request.form.get("name")
        user_state = request.form.get("state")
        weight = float(request.form.get("weight"))
        bp_top = float(request.form.get("bp_top"))
        bp_bottom = float(request.form.get("bp_bottom"))
        glucose = float(request.form.get("glucose"))
        heartrate = float(request.form.get("heartrate"))
        stamina = request.form.get("stamina")
        disease = request.form.get("disease")
        severity = request.form.get("severity")
        daily_calories = float(request.form.get("daily_calories"))

        # -----------------------------
        # PREPROCESSING SAME AS NOTEBOOK
        # -----------------------------

        # BP conversion: (top + bottom) / 2
        bp_value = (bp_top + bp_bottom) / 2

        # -----------------------------
        # BUILD INPUT DATAFRAME ORDERED AS FEATURE NAMES
        # -----------------------------
        input_dict = {
            "Weight": weight,
            "BP": bp_value,
            "Glucose": glucose,
            "HeartRate": heartrate,
            "Stamina": stamina,
            "Disease": disease,
            "Severity": severity,
            "DailyCalories": daily_calories,
        }

        # -----------------------------
        # LABEL ENCODING FOR CATEGORICAL COLUMNS
        # -----------------------------
        for col in label_encoders:
            input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]

        # Convert to numpy array following feature order
        input_vector = np.array([[input_dict[col] for col in feature_names]])

        # -----------------------------
        # SCALING
        # -----------------------------
        input_scaled = scaler.transform(input_vector)

        # -----------------------------
        # PREDICT
        # -----------------------------
        pred_encoded = model.predict(input_scaled)[0]
        predicted_diet = target_encoder.inverse_transform([pred_encoded])[0]

        # -----------------------------
        # FOOD SUGGESTIONS BASED ON STATE & DIET

        from food_map import FOOD_MAP   # Import your food map file

        food_list = FOOD_MAP.get(user_state, {}).get(predicted_diet, [])

        return render_template(
            "result.html",
            name=user_name,
            state=user_state,
            diet=predicted_diet,
            foods=food_list
        )

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
