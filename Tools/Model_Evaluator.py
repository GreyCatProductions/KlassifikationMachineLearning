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
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
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
    print("\n🔍 Evaluation Metrics")
    print("-" * 40)

    # Label distribution
    print("📊 Label Distribution:")
    try:
        label_dist = metrics.get("Test label distribution", {})
        for label, count in label_dist.items():
            print(f"  Label {label}: {count}")
    except Exception as e:
        print(f"  ❌ Failed to print label distribution: {e}")

    # Overall metrics
    print("\n✅ Overall Metrics:")
    for key in ["accuracy", "precision", "recall", "f1"]:
        try:
            value = metrics.get(key, None)
            if isinstance(value, float):
                print(f"  {key.capitalize():<9}: {value:.4f}")
            elif isinstance(value, list):
                print(f"  {key.capitalize():<9}:")
                for i, v in enumerate(value):
                    try:
                        print(f"    Class {i}: {v:.4f}")
                    except Exception as e:
                        print(f"    ❌ Failed to format class {i} value for {key}: {e}")
            else:
                print(f"  {key.capitalize():<9}: {value}")
        except Exception as e:
            print(f"  ❌ Failed to print {key}: {e}")

    # Confusion matrix
    print("\n🧩 Confusion Matrix:")
    try:
        cm = metrics.get("confusion_matrix", [])
        if isinstance(cm, list):
            for row in cm:
                print(" ", row)
        else:
            print("  ❌ Confusion matrix is n    ot in list format.")
    except Exception as e:
        print(f"  ❌ Failed to print confusion matrix: {e}")

    # Per-class metrics
    print("\n📌 Per-Class Metrics:")
    try:
        pcm = metrics.get("per_class_metrics", {})
        for label, vals in pcm.items():
            try:
                tp = vals.get("tp", "N/A")
                fp = vals.get("fp", "N/A")
                fn = vals.get("fn", "N/A")
                tn = vals.get("tn", "N/A")
                print(f"  Class {label}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            except Exception as e:
                print(f"  ❌ Failed to print metrics for class {label}: {e}")
    except Exception as e:
        print(f"  ❌ Failed to print per-class metrics: {e}")

    print("-" * 40)
