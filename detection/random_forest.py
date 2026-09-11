"""Trains a RandomForest classifier to detect obfuscated JavaScript files
based on hand-crafted lexical/statistical heuristics, and saves it (bundled
with its scaler) to detection/models/random_forest/<model_name>.pkl.
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from detection.features import (
        FEATURE_COLUMNS, PACKAGE_DIR, extract_features_for_prediction, extract_features_with_filenames,
        resolve_train_dirs, resolve_test_dirs,
    )
except ImportError:
    # Allows running this file directly (e.g. `python random_forest.py` from
    # inside detection/), where "detection" isn't an importable package.
    from features import (
        FEATURE_COLUMNS, PACKAGE_DIR, extract_features_for_prediction, extract_features_with_filenames,
        resolve_train_dirs, resolve_test_dirs,
    )

# --- Configuration ---
DEFAULT_DATASET = "dataset_v1"
DEFAULT_MODEL_NAME = "random_forest_model"

MODELS_DIR = os.path.join(PACKAGE_DIR, "models", "random_forest")


def remove_outliers(data, labels):
    """Removes outliers using IsolationForest, keeping labels aligned."""
    iso_forest = IsolationForest(random_state=42)
    predictions = iso_forest.fit_predict(data)
    mask = predictions == 1
    return data[mask], labels[mask]


def build_dataframe(obfuscated_dir, nonobfuscated_dir):
    obfuscated, _, _ = extract_features_with_filenames(obfuscated_dir, label=1)
    non_obfuscated, _, _ = extract_features_with_filenames(nonobfuscated_dir, label=0)

    all_features = obfuscated + non_obfuscated
    return pd.DataFrame(all_features, columns=FEATURE_COLUMNS + ['label'])


def train_model(dataset, model_name):
    """Trains on data/train/<dataset>/ and saves the fitted model + scaler,
    bundled together, to models/random_forest/<model_name>.pkl."""
    train_dirs = resolve_train_dirs(dataset)
    df_train = build_dataframe(train_dirs["obfuscated"], train_dirs["non_obfuscated"])
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']

    X_train = X_train.fillna(X_train.mean())
    X_train, y_train = remove_outliers(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_scaled, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump({"model": clf, "scaler": scaler}, model_path)

    return model_path


def evaluate_model(model_path, test_dataset):
    """Scores a saved model against data/test/<test_dataset>/, returning
    accuracy as the fraction of files correctly classified (0-1)."""
    bundle = joblib.load(model_path)
    clf, scaler = bundle["model"], bundle["scaler"]

    test_dirs = resolve_test_dirs(test_dataset)
    df_test = build_dataframe(test_dirs["obfuscated"], test_dirs["non_obfuscated"])
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']

    X_test_scaled = scaler.transform(X_test)
    predictions = clf.predict(X_test_scaled)
    return accuracy_score(y_test, predictions)


def predict_directory(model_path, directory):
    """Predicts every .js file in a directory using a saved model. Returns
    {filename: 1|0} (1 = obfuscated, 0 = non_obfuscated)."""
    bundle = joblib.load(model_path)
    clf, scaler = bundle["model"], bundle["scaler"]

    feature_rows, filenames, forced_labels = extract_features_for_prediction(directory)
    if not filenames:
        return {}

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(X)
    predictions = clf.predict(X_scaled)
    results = (forced if forced is not None else int(pred) for forced, pred in zip(forced_labels, predictions))
    return dict(zip(filenames, results))


def main():
    model_path = train_model(DEFAULT_DATASET, DEFAULT_MODEL_NAME)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
