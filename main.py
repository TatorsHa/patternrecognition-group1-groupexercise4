from pathlib import Path
from src.data_formatting import load_features_from_tsv
from src.dtw import dtw_distance, to_feature_vectors
from src.treshold_classifier import TresholdClassifier
import os

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec


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

def threshold_computation(normalize=True):
    print("Computing treshold for treshold computation")
    root = Path(__file__).resolve().parent
    enrollment = os.path.join(root, "data", "SignatureVerification", "enrollment")
    tc = TresholdClassifier(enrollment, normalize)
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

def threshold_main(normalize=True):
    print("Computing treshold classifier")
    root = Path(__file__).resolve().parent
    enrollment = os.path.join(root, "data", "SignatureVerification", "enrollment")
    tc = TresholdClassifier(enrollment, normalize)
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

    return cm, accuracy, precision, recall, f1

def plot_results(cm, accuracy, precision, recall, f1, title, filename):
    """
    The plot function have been fully generated using ClaudeAI
    """
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # 1. Confusion Matrix (left side, spans 2 rows)
    ax1 = fig.add_subplot(gs[:, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                square=True, linewidths=2, linecolor='white',
                annot_kws={'size': 20, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8})
    ax1.set_title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(['Negative', 'Positive'], fontsize=12)
    ax1.set_yticklabels(['Negative', 'Positive'], fontsize=12, rotation=0)

    # 2. Metrics Bar Chart (top right)
    ax2 = fig.add_subplot(gs[0, 1:])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy, precision, recall, f1]

    bars = ax2.barh(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metrics_values)):
        width = bar.get_width()
        ax2.text(width - 0.05, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', 
                ha='right', va='center', fontsize=14, fontweight='bold', color='white')

    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Score', fontsize=14, fontweight='bold')
    ax2.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_axisbelow(True)

    # 3. Metrics Summary Cards (bottom right)
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.axis('off')

    # Create metric cards
    card_width = 0.22
    card_height = 0.7
    x_positions = [0.05, 0.28, 0.51, 0.74]
    metrics_data = [
        ('Accuracy', accuracy, colors[0]),
        ('Precision', precision, colors[1]),
        ('Recall', recall, colors[2]),
        ('F1 Score', f1, colors[3])
    ]

    for (x_pos, (name, value, color)) in zip(x_positions, metrics_data):
        # Draw card rectangle
        rect = Rectangle((x_pos, 0.15), card_width, card_height, 
                        facecolor=color, edgecolor='black', linewidth=2,
                        transform=ax3.transAxes, alpha=0.8)
        ax3.add_patch(rect)
        
        # Add metric name
        ax3.text(x_pos + card_width/2, 0.7, name, 
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        
        # Add metric value
        ax3.text(x_pos + card_width/2, 0.45, f'{value:.3f}', 
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=24, fontweight='bold', color='white')
        
        # Add percentage
        ax3.text(x_pos + card_width/2, 0.25, f'{value*100:.1f}%', 
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=14, color='white')

    plt.tight_layout()
    plt.savefig(filename+'.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved successfully!")
    plt.show()

if __name__ == "__main__":
    # threshold_computation()
    # threshold_computation(False)

    cm_norm, accuracy_norm, precision_norm, recall_norm, f1_norm = threshold_main()
    cm, accuracy, precision, recall, f1 = threshold_main(False)

    plot_results(cm_norm, accuracy_norm, precision_norm, recall_norm, f1_norm, "Threshold Classifier Normalized", "class_normalized")
    plot_results(cm, accuracy, precision, recall, f1, "Threshold Classifier No Normalization", "class")