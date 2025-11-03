from collections import Counter
from pathlib import Path
import optuna
import torch
from datasets import Dataset
from setfit import Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
from Tools import Model_Evaluator
from Tools.Model_Usage import FewShot
import Tools.GPU_Monitor


def _extract_texts_and_labels(texts: list[str], labels: list[str], train_index, test_index) \
        -> tuple[list[str], list[str], list[str], list[str]]:
    train_texts = [texts[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    test_texts = [texts[i] for i in test_index]
    test_labels = [labels[i] for i in test_index]
    return train_texts, train_labels, test_texts, test_labels


def _create_dataset(texts, labels):
    return Dataset.from_dict({"text": texts, "label": labels})

def _train_and_evaluate_model(model_to_use: Path, train_dataset: Dataset, test_dataset: Dataset, optuna_params, average) -> tuple[dict, list[str], float, float]:
    model = FewShot.load_model(model_to_use)

    arguments = TrainingArguments(
        num_epochs=optuna_params["num_epochs"],
        batch_size=optuna_params["batch_size"],
        num_iterations=optuna_params["num_iterations"],
        head_learning_rate=optuna_params["head_lr"],
        save_strategy="no",
        eval_strategy="no",
        use_amp = True
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric=lambda y_true, y_pred: {"f1": f1_score(y_true, y_pred, average=average)},
        column_mapping={"text": "text", "label": "label"},
        args=arguments
    )

    usage_list = []
    #stop_event = threading.Event()
    #monitor_thread = threading.Thread(target=Tools.GPU_Monitor.monitor_gpu, args=(5, stop_event, usage_list))
    #monitor_thread.start()

    trainer.train()
    #    stop_event.set()
    #    monitor_thread.join()

    avg_vram = sum(usage_list) / len(usage_list) if usage_list else 0
    max_vram = max(usage_list) if usage_list else 0

    metrics: dict = {}
    predictions: list[str] = []
    metrics, predictions = Model_Evaluator.evaluate_model(model, test_dataset["text"], test_dataset["label"], average)

    del model
    del trainer
    torch.cuda.empty_cache()
    return metrics, predictions, avg_vram, max_vram

def cross_validate_with_optuna(model_to_use: Path, texts: list[str], labels: list[str], n_splits: int, n_trials: int, average: str):

    all_metrics: list[list[dict]] = []

    def objective(trial):
        params = {
            "num_iterations": trial.suggest_int("num_iterations", 6, 9),
            "num_epochs": trial.suggest_int("num_epochs", 2, 4),
            "batch_size": trial.suggest_categorical("batch_size", [16]),
            "head_lr": trial.suggest_float("head_learning_rate", 1e-5, 1e-1, log=True)
        }
        print(f"Trial {trial.number}: {params}")

        #preparation
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        all_fold_metrics = []
        fold_predictions = np.empty(len(labels), dtype=object)
        vrams: list[tuple[float, float]] = []

        #cv loop
        for fold, (train_idx, test_idx) in enumerate(kfold.split(texts, labels)):
            print(f"Fold {fold + 1}/{n_splits}")

            #filtering
            train_texts, train_labels, test_texts, test_labels = _extract_texts_and_labels(texts, labels, train_idx, test_idx)
            train_dataset: Dataset = _create_dataset(train_texts, train_labels)
            test_dataset: Dataset = _create_dataset(test_texts, test_labels)

            #training
            fold_metrics, predictions, avg_vram, max_vram = _train_and_evaluate_model(model_to_use, train_dataset, test_dataset, params, average)
            f1 = fold_metrics["f1"]

            #collecting
            all_fold_metrics.append(fold_metrics)
            scores.append(f1)
            fold_predictions[test_idx] = predictions
            vrams.append((avg_vram, max_vram))

            print(f"Fold finished F1 Score: {f1}", Model_Evaluator.pretty_print(fold_metrics))

            #cleanup
            del train_dataset, test_dataset
            torch.cuda.empty_cache()

        avg_vram = np.mean([vram[0] for vram in vrams])
        max_vram = np.max([vram[1] for vram in vrams])

        unique_labels = sorted(set(labels))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, fold_predictions, labels=unique_labels, average=average, zero_division=0
        )
        combined_accuracy = accuracy_score(labels, fold_predictions)
        cm = confusion_matrix(labels, fold_predictions, labels=unique_labels)
        class_metrics = {}
        for i, label in enumerate(unique_labels):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            tn = int(cm.sum() - (tp + fp + fn))
            class_metrics[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

        trial_metrics = {
            "Test label distribution": Counter(labels),
            "accuracy": combined_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": class_metrics
        }
        print("\n=== Combined Metrics Across All Folds ===")
        Model_Evaluator.pretty_print(trial_metrics)

        print(f"Average VRAM usage: {avg_vram:.2f} MB, Max VRAM usage: {max_vram:.2f} MB")
        print(f"Trial {trial.number} completed with avg fold F1 score: {np.mean(scores)}")
        all_metrics.append(all_fold_metrics)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print(f"\n\nBest Trial Found {study.best_trial.number}:")
    print(study.best_trial)

    print("\nBest Metrics: ")
    fold_metrics_of_best_trial = all_metrics[study.best_trial.number]

    for f in range(n_splits):
        print(f"\nFold {f + 1} Metrics:")
        metrics = fold_metrics_of_best_trial[f]
        Model_Evaluator.pretty_print(metrics)

    return best_params
