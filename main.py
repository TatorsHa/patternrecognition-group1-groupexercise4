from pathlib import Path
from src.data_formatting import load_features_from_tsv
from src.dtw import dtw_distance, to_feature_vectors
from src.treshold_classifier import TresholdClassifier
import os

from tqdm import tqdm

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def main():
    root = Path(__file__).resolve().parent
    enrollment = root / "data" / "SignatureVerification" / "enrollment"

    a = load_features_from_tsv(enrollment / "001-g-01.tsv")
    b = load_features_from_tsv(enrollment / "001-g-02.tsv")
    dist = dtw_distance(to_feature_vectors(a), to_feature_vectors(b), window=3)
    print(f"DTW distance (window=3) between 001-g-01 and 001-g-02: {dist:.3f}")

def threshold_computation():
    print("Computing treshold for treshold computation")
    root = Path(__file__).resolve().parent
    enrollment = os.path.join(root, "data", "SignatureVerification", "enrollment")
    tc = TresholdClassifier(enrollment)
    verification = os.path.join(root, "data", "SignatureVerification", "verification")

    labels_file = os.path.join(root, "data", "SignatureVerification", "gt.tsv")

    with open(labels_file, "r") as f:
        total_lines = sum(1 for line in f if line.strip())

    counter_true = 0
    counter_false = 0
    true_sum = 0
    false_sum = 0

    with open(labels_file, "r") as f:
        for line in tqdm(f, total=total_lines, desc="Processing signatures"):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                filename, label = parts
                file = f"{filename}.tsv"

                true_label = 1 if label == "genuine" else 0
                
                if true_label == 1:
                    counter_true += 1
                    true_sum += min(tc.compute_distances(file, verification))
                else:
                    counter_false += 1
                    false_sum += min(tc.compute_distances(file, verification))
                    
    print("1:", true_sum / counter_true)
    print("0:", false_sum / counter_false)

def threshold_main():
    print("Computing treshold classifier")
    root = Path(__file__).resolve().parent
    enrollment = os.path.join(root, "data", "SignatureVerification", "enrollment")
    tc = TresholdClassifier(enrollment)
    verification = os.path.join(root, "data", "SignatureVerification", "verification")

    labels_file = os.path.join(root, "data", "SignatureVerification", "gt.tsv")

    y_true = []
    y_pred = []

    with open(labels_file, "r") as f:
        total_lines = sum(1 for line in f if line.strip())

    with open(labels_file, "r") as f:
        for line in tqdm(f, total=total_lines, desc="Processing signatures"):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                filename, label = parts
                file = f"{filename}.tsv"

                prediction = tc.predict(file, verification)

                true_label = 1 if label == "genuine" else 0

                y_true.append(true_label)
                y_pred.append(prediction)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)  # For genuine class
    recall = recall_score(y_true, y_pred)  # For genuine class
    f1 = f1_score(y_true, y_pred)

    print(cm)

    return cm, accuracy, precision, recall, f1


if __name__ == "__main__":
    treshold_statistics = threshold_main()
    