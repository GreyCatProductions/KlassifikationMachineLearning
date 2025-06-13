# In Helpers/Model_Validator.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter

def evaluate_model(model, test_texts: list[str], test_labels: list[int]):
    predictions = model.predict(test_texts)
    accuracy = accuracy_score(test_labels, predictions)

    unique_labels = sorted(list(set(test_labels)))
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average=None, labels=unique_labels)

    cm = confusion_matrix(test_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (0, 0, 0, 0)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Test label distribution": Counter(test_labels),
        "accuracy": accuracy,
        "precision_class_0": precision[0] if len(precision) > 0 else 0.0,
        "precision_class_1": precision[1] if len(precision) > 1 else 0.0,
        "recall_class_0": recall[0] if len(recall) > 0 else 0.0,
        "recall_class_1": recall[1] if len(recall) > 1 else 0.0,
        "f1_class_0": f1[0] if len(f1) > 0 else 0.0,
        "f1_class_1": f1[1] if len(f1) > 1 else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }
    return metrics