import threading
from pathlib import Path
import optuna
import torch
from datasets import Dataset, concatenate_datasets
from setfit import Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
from Tools import Model_Evaluator
from Tools.Model_Usage import FewShot
import Tools.GPU_Monitor


def _extract_texts_and_labels(
    texts: list[str], labels: list[str], train_index, test_index) -> tuple[list[str], list[str], list[str], list[str]]:
    train_texts = [texts[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    test_texts = [texts[i] for i in test_index]
    test_labels = [labels[i] for i in test_index]
    return train_texts, train_labels, test_texts, test_labels


def _create_dataset(texts, labels) -> Dataset:
    return Dataset.from_dict({"text": texts, "label": labels})

def _train_and_evaluate_model(model_to_use: Path, train_dataset, test_dataset, optuna_params, average) -> tuple[float, dict, float, float]:
    model = FewShot.load_model(model_to_use)

    arguments = TrainingArguments(
        num_epochs=optuna_params["num_epochs"],
        batch_size=optuna_params["batch_size"],
        num_iterations=optuna_params["num_iterations"],
        #body_learning_rate=optuna_params["body_lr"],
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
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=Tools.GPU_Monitor.monitor_gpu, args=(5, stop_event, usage_list))
    monitor_thread.start()

    try:
        trainer.train()
    finally:
        stop_event.set()
        monitor_thread.join()

    avg_vram = sum(usage_list) / len(usage_list) if usage_list else 0
    max_vram = max(usage_list) if usage_list else 0

    metrics: dict = Model_Evaluator.evaluate_model(model, test_dataset["text"], test_dataset["label"], average)
    f1 = trainer.evaluate()["f1"]

    del model
    del trainer
    torch.cuda.empty_cache()
    return f1, metrics, avg_vram, max_vram

def cross_validate_with_optuna(model_to_use: Path, texts_add: list[str], labels_add: list[str], texts_kfold: list[str], labels_kfold: list[str], n_splits: int, n_trials: int, average: str):

    all_metrics: list[list[dict]] = []

    dataset_to_add = None
    if texts_add is not None and labels_add is not None and len(texts_add) > 0 and len(labels_add) > 0:
        dataset_to_add = _create_dataset(texts_add, labels_add)

    def objective(trial):
        params = {
            "num_iterations": trial.suggest_int("num_iterations", 9, 11),
            "num_epochs": trial.suggest_int("num_epochs", 4, 6),
            "batch_size": trial.suggest_categorical("batch_size", [16]),
            "head_lr": trial.suggest_float("head_learning_rate", 1e-5, 2e-5, log=True)
        }

        print(f"Trial {trial.number}: {params}")

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        fold_metrics = []
        vrams: list[tuple[float, float]] = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(texts_kfold, labels_kfold)):
            print(f"Fold {fold + 1}/{n_splits}")

            train_texts, train_labels, test_texts, test_labels = _extract_texts_and_labels(texts_kfold, labels_kfold, train_idx, test_idx)
            train_dataset: Dataset = _create_dataset(train_texts, train_labels)
            if dataset_to_add is not None:
                train_dataset = concatenate_datasets([train_dataset, dataset_to_add])

            test_dataset = _create_dataset(test_texts, test_labels)

            f1, metrics, avg_vram, max_vram = _train_and_evaluate_model(model_to_use, train_dataset, test_dataset, params, average)
            fold_metrics.append(metrics)
            scores.append(f1)
            vrams.append((avg_vram, max_vram))

            print(f"Fold finished F1 Score: {f1} {metrics["f1"]}", Model_Evaluator.pretty_print(metrics))

            del train_dataset
            del test_dataset
            torch.cuda.empty_cache()

        avg_vram = np.mean([vram[0] for vram in vrams])
        max_vram = np.max([vram[1] for vram in vrams])

        print(f"Average VRAM usage: {avg_vram:.2f} MB, Max VRAM usage: {max_vram:.2f} MB")
        print(f"Trial {trial.number} completed with avg fold F1 score: {np.mean(scores)}")
        all_metrics.append(fold_metrics)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print(f"\n\nBest Trial Found {study.best_trial.number}:")
    print(study.best_trial)

    print("\nMetrics: ")
    fold_metrics = all_metrics[study.best_trial.number]

    for fold in range(n_splits):
        print(f"\nFold {fold + 1} Metrics:")
        metrics = fold_metrics[fold]
        Model_Evaluator.pretty_print(metrics)

    return best_params
