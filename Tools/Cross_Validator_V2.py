import threading
from typing import Tuple, List
import optuna
import torch
from datasets import Dataset
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


def _create_dataset(texts, labels):
    return Dataset.from_dict({"text": texts, "label": labels})

def _train_and_evaluate_model(train_dataset, test_dataset, args, average) -> tuple[float, dict, float, float]:
    model = FewShot.load_model()

    arguments = TrainingArguments(
        num_epochs=args["num_epochs"],
        batch_size=args["batch_size"],
        num_iterations=args["num_iterations"],
        body_learning_rate=args["body_lr"],
        head_learning_rate=args["head_lr"],
        save_strategy="no",
        eval_strategy="epoch",
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

    metrics = Model_Evaluator.evaluate_model(model, test_dataset["text"], test_dataset["label"], average)
    f1 = trainer.evaluate()["f1"]

    del model
    del trainer
    torch.cuda.empty_cache()
    return f1, metrics, avg_vram, max_vram

def cross_validate_with_optuna(texts: list[str], labels: list[str], n_splits: int, n_trials: int, average: str):

    all_metrics = []

    def objective(trial):
        params = {
            "num_iterations": trial.suggest_int("num_iterations", 6, 10),
            "num_epochs": trial.suggest_int("num_epochs", 1, 3),
            "batch_size": trial.suggest_categorical("batch_size", [16, 24]),
            "body_lr": trial.suggest_float("body_learning_rate", 1e-6, 5e-5, log=True),
            "head_lr": trial.suggest_float("head_learning_rate", 1e-5, 1e-1, log=True)
        }

        print(f"Trial {trial.number}: {params}")

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        fold_metrics = []
        vrams: List[Tuple[float, float]] = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(texts, labels)):
            print(f"Fold {fold + 1}/{n_splits}")

            train_texts, train_labels, test_texts, test_labels = _extract_texts_and_labels(texts, labels, train_idx, test_idx)
            train_dataset = _create_dataset(train_texts, train_labels)
            test_dataset = _create_dataset(test_texts, test_labels)

            f1_score, metrics, avg_vram, max_vram = _train_and_evaluate_model(train_dataset, test_dataset, params, average)
            fold_metrics.append(metrics)
            vrams.append((avg_vram, max_vram))

            print("Fold finished")
            scores.append(f1_score)
            del train_dataset
            del test_dataset
            torch.cuda.empty_cache()

        avg_vram = np.mean([vram[0] for vram in vrams])
        max_vram = np.max([vram[1] for vram in vrams])

        print(f"Average VRAM usage: {avg_vram:.2f} MB, Max VRAM usage: {max_vram:.2f} MB")
        print(f"Trial {trial.number} completed with F1 score: {np.mean(scores)}\n metrics: {fold_metrics}")
        all_metrics.append(fold_metrics)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print("\n\nBest Trial Found:")
    print(study.best_trial)

    print("\nMetrics: ")
    metrics = all_metrics[study.best_trial.number]

    print(metrics)
    Model_Evaluator.pretty_print(metrics)

    return best_params
