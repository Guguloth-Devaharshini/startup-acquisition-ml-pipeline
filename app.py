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
# Global model variables
# --------------------------------------------------

model = None
model_metadata = None


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model():
    global model
    global model_metadata

    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found: {MODEL_PATH}")
            return False

        model = joblib.load(MODEL_PATH)

        feature_names = list(model.feature_names_in_)

        model_metadata = {
            "feature_schema": {
                "feature_names": feature_names
            },
            "model_info": {
                "model_name": "Random Forest - Balanced Classes",
                "n_features": len(feature_names),
                "n_classes": len(model.classes_),
                "classes": [int(x) for x in model.classes_],
                "class_labels": {
                    str(int(cls)): CLASS_LABELS.get(
                        int(cls),
                        f"Class {int(cls)}"
                    )
                    for cls in model.classes_
                }
            }
        }

        print("=" * 60)
        print("MODEL LOADED SUCCESSFULLY")
        print("=" * 60)
        print("Model:", MODEL_PATH)
        print("Features:", len(feature_names))
        print("Classes:", len(model.classes_))
        print(
            "Class labels:",
            model_metadata["model_info"]["class_labels"]
        )

        return True

    except Exception as e:
        print(f"Error loading model: {e}")
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

    if df.empty:
        raise ValueError(
            "No startup information was provided."
        )

    # Treat empty strings as missing values
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # --------------------------------------------------
    # Convert friendly UI selections into model features
    # --------------------------------------------------

    if "industry" in df.columns:

        industry = str(
            df["industry"].iloc[0]
        ).lower()

        industry_feature = (
            f"category_code_reduced_{industry}"
        )

        if industry_feature in FEATURES:
            df[industry_feature] = 1

    if "country" in df.columns:

        country = str(
            df["country"].iloc[0]
        )

        country_feature = (
            f"country_code_reduced_{country}"
        )

        if country_feature in FEATURES:
            df[country_feature] = 1

    # --------------------------------------------------
    # Required numeric startup inputs
    # --------------------------------------------------

    important_fields = [
        "funding_rounds",
        "funding_total_usd",
        "milestones",
        "relationships",
        "lat",
        "lng",
        "founded_at_year",
        "first_funding_at_year",
        "first_milestone_at_year"
    ]

    # --------------------------------------------------
    # Check required fields
    # --------------------------------------------------

    missing_fields = []

    for field in important_fields:

        if field not in df.columns:

            missing_fields.append(field)

        elif df[field].isna().all():

            missing_fields.append(field)

    if missing_fields:

        raise ValueError(
            "Please complete the required startup information: "
            + ", ".join(missing_fields)
        )

    # --------------------------------------------------
    # Add missing model features as 0
    # --------------------------------------------------

    for feature in FEATURES:

        if feature not in df.columns:

            df[feature] = 0

    # --------------------------------------------------
    # Keep exact training feature order
    # --------------------------------------------------

    df = df[FEATURES].copy()

    # --------------------------------------------------
    # Convert all features to numeric
    # --------------------------------------------------

    for column in FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------
    # Check required fields after conversion
    # --------------------------------------------------

    invalid_fields = []

    for field in important_fields:

        if df[field].isna().any():

            invalid_fields.append(field)

    if invalid_fields:

        raise ValueError(
            "Invalid or missing values in: "
            + ", ".join(invalid_fields)
        )

    # --------------------------------------------------
    # Fill remaining optional encoded features
    # --------------------------------------------------

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
        features=FEATURES,
        metadata=model_metadata
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

        prediction = model.predict(input_df)[0]

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

        # ------------------------------------------
        # JSON response
        # ------------------------------------------

        if request.is_json:
            return jsonify(response)

        # ------------------------------------------
        # Web page response
        # ------------------------------------------

        return render_template(
            "result.html",
            result=response,
            input_data=data
        )

    except Exception as e:

        print(f"Prediction error: {e}")

        # JSON errors should also return JSON
        if request.is_json:
            return jsonify({
                "error": str(e)
            }), 400

        return render_template(
            "error.html",
            error=f"Prediction failed: {str(e)}"
        ), 400


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": (
            "healthy"
            if model is not None
            else "model_not_loaded"
        ),

        "model": MODEL_PATH,

        "features": len(FEATURES)
    })


# --------------------------------------------------
# Model information / API documentation
# --------------------------------------------------

@app.route("/api/info", methods=["GET"])
def api_info():

    return render_template(
        "api_info.html",
        metadata=model_metadata,
        features=FEATURES
    )


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
        print("Prediction endpoint: /predict")
        print("Health endpoint: /health")
        print("API documentation: /api/info")

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            use_reloader=False
        )

    else:

        print(
            "ERROR: Could not load the trained model."
        )