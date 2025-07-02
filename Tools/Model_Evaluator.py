from typing import Dict

from setfit import SetFitModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter


def evaluate_model(model: SetFitModel, test_texts: list[str], test_labels: list[int], average: str) -> dict:
    predictions = model.predict(test_texts)
    accuracy = accuracy_score(test_labels, predictions)

    unique_labels = sorted(list(set(test_labels)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average=average, labels=unique_labels
    )

    cm = confusion_matrix(test_labels, predictions, labels=unique_labels)

    class_metrics = {}
    for i, label in enumerate(unique_labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        class_metrics[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    metrics = {
        "Test label distribution": Counter(test_labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": class_metrics
    }

    return metrics

def pretty_print(metrics: Dict):
    try:
        print("\n🔍 Evaluation Metrics")
        print("-" * 40)

        # Label distribution
        print("📊 Label Distribution:")
        for label, count in metrics["Test label distribution"].items():
            print(f"  Label {label}: {count}")

        # Overall metrics
        print("\n✅ Overall Metrics:")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1']:.4f}")

        # Confusion matrix
        print("\n🧩 Confusion Matrix:")
        for row in metrics["confusion_matrix"]:
            print(" ", row)

        # Per-class metrics
        print("\n📌 Per-Class Metrics:")
        for label, vals in metrics["per_class_metrics"].items():
            print(f"  Class {label}: TP={vals['tp']}, FP={vals['fp']}, FN={vals['fn']}, TN={vals['tn']}")

        print("-" * 40)
    except:
        print("Failed to pretty print metrics")

