from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os


app = Flask(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/random_forest_balanced.pkl"

FEATURES = [
    "founded_at",
    "first_funding_at",
    "funding_rounds",
    "funding_total_usd",
    "first_milestone_at",
    "milestones",
    "relationships",
    "lat",
    "lng",
    "founded_at_year",
    "first_funding_at_year",
    "first_milestone_at_year",

    "category_code_reduced_Other",
    "category_code_reduced_advertising",
    "category_code_reduced_biotech",
    "category_code_reduced_consulting",
    "category_code_reduced_ecommerce",
    "category_code_reduced_enterprise",
    "category_code_reduced_games_video",
    "category_code_reduced_mobile",
    "category_code_reduced_other",
    "category_code_reduced_software",
    "category_code_reduced_web",

    "country_code_reduced_AUS",
    "country_code_reduced_CAN",
    "country_code_reduced_DEU",
    "country_code_reduced_ESP",
    "country_code_reduced_FRA",
    "country_code_reduced_GBR",
    "country_code_reduced_IND",
    "country_code_reduced_ISR",
    "country_code_reduced_NLD",
    "country_code_reduced_Other",
    "country_code_reduced_USA"
]


# --------------------------------------------------
# Class labels
# --------------------------------------------------

CLASS_LABELS = {
    0: "Closed / Failed",
    1: "Operating",
    2: "Acquired",
    3: "Successful / Active"
}


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = None


def load_model():
    global model

    try:
        if not os.path.exists(MODEL_PATH):
            print(
                f"Model file not found: {MODEL_PATH}"
            )
            return False

        model = joblib.load(MODEL_PATH)

        # Get feature names directly from trained model
        FEATURE_NAMES = list(model.feature_names_in_)

        # Build feature schema for frontend
        model_metadata = {
            "feature_schema": {
                "feature_names": FEATURE_NAMES
            },
            "model_info": {
                "model_name": "Random Forest - Balanced Classes",
                "n_features": len(FEATURE_NAMES),
                "n_classes": len(model.classes_),
                "classes": [int(x) for x in model.classes_]
            }
        }

        print("Model loaded successfully!")
        print("Model:", MODEL_PATH)
        print("Features:", len(FEATURE_NAMES))

        print("Model loaded successfully!")
        
        print(
            f"Model: {MODEL_PATH}"
        )

        return True

    except Exception as e:
        print(
            f"Error loading model: {e}"
        )
        return False


# --------------------------------------------------
# Prepare input
# --------------------------------------------------

def prepare_input(data):
    """
    Convert incoming form/JSON data into
    the exact 34-feature DataFrame expected
    by the trained Random Forest.
    """

    if isinstance(data, dict):
        df = pd.DataFrame([data])

    elif isinstance(data, list):
        df = pd.DataFrame(data)

    else:
        raise ValueError(
            "Input must be a dictionary or list."
        )

    # Add missing features
    for feature in FEATURES:

        if feature not in df.columns:
            df[feature] = 0

    # Keep only training features
    df = df[FEATURES].copy()

    # Convert values to numeric
    for column in FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Replace invalid/missing values
    df = df.fillna(0)

    return df


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():

    if model is None:
        return render_template(
            "error.html",
            error="Model is not loaded."
        )

    return render_template(
        "index.html",
        features=FEATURES
    )


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------------
        # Get request data
        # ------------------------------------------

        if request.is_json:

            data = request.get_json()

        else:

            data = request.form.to_dict()

        # ------------------------------------------
        # Prepare features
        # ------------------------------------------

        input_df = prepare_input(data)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = model.predict(
            input_df
        )[0]

        prediction = int(prediction)

        # ------------------------------------------
        # Probabilities
        # ------------------------------------------

        probabilities = {}

        if hasattr(model, "predict_proba"):

            prediction_proba = model.predict_proba(
                input_df
            )[0]

            for class_value, probability in zip(
                model.classes_,
                prediction_proba
            ):

                class_value = int(class_value)

                probabilities[
                    f"class_{class_value}"
                ] = float(probability)

            confidence = float(
                max(prediction_proba)
            )

        else:

            confidence = 0.0

        # ------------------------------------------
        # Response
        # ------------------------------------------

        response = {
            "prediction": prediction,

            "prediction_label": CLASS_LABELS.get(
                prediction,
                f"Class {prediction}"
            ),

            "probabilities": probabilities,

            "confidence": confidence,

            "model_type":
                "Random Forest - Balanced Classes"
        }

        # JSON response
        if request.is_json:

            return jsonify(response)

        # Web page response
        return render_template(
            "result.html",
            result=response,
            input_data=data
        )

    except Exception as e:

        print(
            f"Prediction error: {e}"
        )

        return render_template(
            "error.html",
            error=f"Prediction failed: {str(e)}"
        ), 500


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy" if model is not None else "model_not_loaded",
        "model": MODEL_PATH,
        "features": len(FEATURES)
    })


# --------------------------------------------------
# Model information
# --------------------------------------------------

@app.route("/api/info")
def api_info():
    """Professional API documentation page."""
    return render_template("api_info.html")


# --------------------------------------------------
# Error handlers
# --------------------------------------------------

@app.errorhandler(404)
def not_found(error):

    return render_template(
        "error.html",
        error="Page not found"
    ), 404


@app.errorhandler(500)
def internal_error(error):

    return render_template(
        "error.html",
        error="Internal server error"
    ), 500


# --------------------------------------------------
# Start application
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("STARTUP ACQUISITION ML APPLICATION")
    print("=" * 60)

    if load_model():

        print("Starting Flask application...")
        print(
            "Prediction endpoint: /predict"
        )

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            use_reloader =False
        )

    else:

        print(
            "ERROR: Could not load the trained model."
        )