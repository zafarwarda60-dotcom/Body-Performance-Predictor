"""Clean, evaluate, and save body-performance regression and classification models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "bodyPerformance.csv"
MODEL_DIR = ROOT / "models"

# API-safe names mapped from the source CSV's original column headers.
COLUMN_MAP = {
    "gripForce": "grip_force",
    "sit and bend forward_cm": "sit_bend_forward",
    "sit-ups counts": "sit_ups_count",
    "broad jump_cm": "broad_jump_cm",
    "body fat_%": "body_fat",
}
REGRESSION_FEATURES = [
    "age", "gender", "height_cm", "weight_kg", "diastolic", "systolic",
    "grip_force", "sit_bend_forward", "sit_ups_count", "broad_jump_cm",
]
CLASSIFICATION_FEATURES = REGRESSION_FEATURES + ["body_fat"]
TARGET_REGRESSION = "body_fat"
TARGET_CLASSIFICATION = "class"
CATEGORICAL_FEATURES = ["gender"]
RANDOM_STATE = 42


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop incomplete/duplicate records and impossible blood-pressure rows."""
    before = len(frame)
    frame = frame.rename(columns=COLUMN_MAP).copy()
    frame = frame.dropna(subset=REGRESSION_FEATURES + ["body_fat", "class"])
    complete = len(frame)
    frame = frame.drop_duplicates()
    unique = len(frame)
    frame = frame[(frame["diastolic"] > 0) & (frame["systolic"] > frame["diastolic"])]
    return frame.reset_index(drop=True), {
        "input_rows": before,
        "missing_rows_removed": before - complete,
        "duplicate_rows_removed": complete - unique,
        "invalid_blood_pressure_rows_removed": unique - len(frame),
        "clean_rows": len(frame),
    }


def make_pipeline(estimator, features: list[str]) -> Pipeline:
    categorical = [name for name in features if name in CATEGORICAL_FEATURES]
    numeric = [name for name in features if name not in categorical]
    prep = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    data, cleaning = clean_data(pd.read_csv(DATA))
    MODEL_DIR.mkdir(exist_ok=True)

    x_train, x_test, y_train, y_regression_test = train_test_split(
        data[REGRESSION_FEATURES], data[TARGET_REGRESSION],
        test_size=0.2, random_state=RANDOM_STATE,
    )
    regressor = make_pipeline(
        RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        REGRESSION_FEATURES,
    )
    regressor.fit(x_train, y_train)
    regression_prediction = regressor.predict(x_test)
    regression_cv = cross_val_score(
        make_pipeline(RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1), REGRESSION_FEATURES),
        data[REGRESSION_FEATURES], data[TARGET_REGRESSION], cv=5, scoring="r2", n_jobs=1,
    )
    joblib.dump(regressor, MODEL_DIR / "body_fat_regressor.joblib", compress=3)

    x_train, x_test, y_train, y_classification_test = train_test_split(
        data[CLASSIFICATION_FEATURES], data[TARGET_CLASSIFICATION],
        test_size=0.2, random_state=RANDOM_STATE, stratify=data[TARGET_CLASSIFICATION],
    )
    classifier = make_pipeline(
        RandomForestClassifier(n_estimators=250, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        CLASSIFICATION_FEATURES,
    )
    classifier.fit(x_train, y_train)
    classification_prediction = classifier.predict(x_test)
    labels = sorted(data[TARGET_CLASSIFICATION].unique().tolist())
    classification_cv = cross_val_score(
        make_pipeline(RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1), CLASSIFICATION_FEATURES),
        data[CLASSIFICATION_FEATURES], data[TARGET_CLASSIFICATION], cv=5, scoring="accuracy", n_jobs=1,
    )
    joblib.dump(classifier, MODEL_DIR / "performance_classifier.joblib", compress=3)

    # Use source feature names, including one-hot encoded gender indicators.
    transformed_names = regressor.named_steps["preprocess"].get_feature_names_out().tolist()
    importance = regressor.named_steps["model"].feature_importances_
    feature_importance = sorted(
        [{"feature": name.split("__", 1)[-1], "importance": round(float(value), 4)}
         for name, value in zip(transformed_names, importance)],
        key=lambda item: item["importance"], reverse=True,
    )
    results = {
        "cleaning": cleaning,
        "features": {
            "regression": REGRESSION_FEATURES,
            "classification": CLASSIFICATION_FEATURES,
        },
        "regression": {
            "r2": round(float(r2_score(y_regression_test, regression_prediction)), 4),
            "mae": round(float(mean_absolute_error(y_regression_test, regression_prediction)), 2),
            "cross_validation_r2_mean": round(float(regression_cv.mean()), 4),
            "cross_validation_r2_std": round(float(regression_cv.std()), 4),
        },
        "classification": {
            "accuracy": round(float(accuracy_score(y_classification_test, classification_prediction)), 4),
            "cross_validation_accuracy_mean": round(float(classification_cv.mean()), 4),
            "cross_validation_accuracy_std": round(float(classification_cv.std()), 4),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_classification_test, classification_prediction, labels=labels).tolist(),
        },
        "regression_feature_importance": feature_importance,
        "holdout": {"test_size": 0.2, "random_state": RANDOM_STATE},
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
