import csv
import os
import random

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "eye_training_data.csv"
MODEL_FILE = "contextra_eye_model.joblib"


def create_sample_data():
    """
    Creates clearly labelled SYNTHETIC data to test the ML pipeline.
    This is not real-world drowsiness training data.
    """
    random.seed(42)
    rows = []

    for _ in range(500):
        # Synthetic eye aspect ratio (EAR) values.
        ear_open = random.uniform(0.23, 0.38)
        ear_closed = random.uniform(0.08, 0.20)

        rows.append([ear_open, 0])     # 0 = synthetic closed? corrected below
        rows.append([ear_closed, 1])

    # Label convention: 0 = closed, 1 = open.
    # Replace labels to match the generated EAR values.
    corrected = []
    for ear, label in rows:
        if ear >= 0.23:
            corrected.append([ear, 1])
        else:
            corrected.append([ear, 0])

    with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ear", "label"])
        writer.writerows(corrected)

    print("Created synthetic pipeline-test dataset:", DATA_FILE)


def main():
    if not os.path.exists(DATA_FILE):
        create_sample_data()

    data = np.genfromtxt(DATA_FILE, delimiter=",", skip_header=1)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("CSV must have columns: ear,label")

    X = data[:, [0]]
    y = data[:, 1].astype(int)

    if len(set(y)) < 2:
        raise ValueError("Training data must contain both labels: 0 and 1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== CONTEXTRA EYE-STATE ML EXPERIMENT ===")
    print("Training examples:", len(X_train))
    print("Test examples:", len(X_test))
    print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["closed", "open"],
        zero_division=0
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    joblib.dump(model, MODEL_FILE)
    print("\nSaved model to:", MODEL_FILE)
    print(
        "\nIMPORTANT: Synthetic data only. "
        "This result does not measure real-world eye-state or drowsiness accuracy."
    )


if __name__ == "__main__":
    main()
