import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = "fe_outcomes2.csv"
MODEL_PATH = "models/random_forest_balanced.pkl"


# --------------------------------------------------
# Features
# --------------------------------------------------

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
# Main training function
# --------------------------------------------------

def main():

    # Load dataset
    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-16"
    )

    print("Original dataset:", df.shape)


    # --------------------------------------------------
    # Create target classes
    # --------------------------------------------------

    # Original status values:
    # 0.0, 0.333333, 0.666667, 1.0
    #
    # Convert them into:
    # 0, 1, 2, 3

    df["status_class"] = (
        df["status"] * 3
    ).round().astype(int)


    # --------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------

    df = df.drop_duplicates()

    print(
        "After removing duplicates:",
        df.shape
    )


    # --------------------------------------------------
    # Select features
    # --------------------------------------------------

    features = [
        col
        for col in FEATURES
        if col in df.columns
    ]

    X = df[features].copy()
    y = df["status_class"].copy()

    print(
        "Number of features:",
        len(features)
    )


    # --------------------------------------------------
    # Train / Test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        "Training rows:",
        len(X_train)
    )

    print(
        "Testing rows:",
        len(X_test)
    )


    # --------------------------------------------------
    # Random Forest with class weighting
    # --------------------------------------------------

    print(
        "Training Random Forest with class weights..."
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    predictions = model.predict(X_test)


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    print("\nModel Performance")
    print("-----------------")

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Macro F1:",
        round(macro_f1, 4)
    )


    print("\nClassification Report")
    print("---------------------")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        "\nModel saved to:",
        MODEL_PATH
    )


# --------------------------------------------------
# Run program
# --------------------------------------------------

if __name__ == "__main__":
    main()