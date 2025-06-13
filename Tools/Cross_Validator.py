import copy
from typing import Dict, Union
import numpy as np
import optuna
from datasets import Dataset
from setfit import Trainer
from sklearn.model_selection import StratifiedKFold
from Tools import Model_Validator
from Tools.Model_Usage import FewShot
from functools import partial

def model_init_for_tuning(trial=None):
    model = FewShot.load_model()
    return model

def apply_trial_params_to_args(base_args, trial_params):
    args_copy = copy.deepcopy(base_args)
    for k, v in trial_params.items():
        if k in ("batch_size", "num_epochs"):
            if not (isinstance(v, tuple) or isinstance(v, list)):
                setattr(args_copy, k, (v, v))
            else:
                setattr(args_copy, k, v)
        elif hasattr(args_copy, k):
            setattr(args_copy, k, v)
    return args_copy


def hp_space(trial: optuna.Trial) -> Dict[str, Union[float, int, str]]:
    params = {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-3, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
    }
    print(f"[Trial {trial.number}] Trying params: {params}")
    return params

def cross_validate(device: str, args: Trainer.args, n_splits, texts: list[str], labels: list[int]):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels_np = np.array(labels)

    all_fold_results = []
    all_best_params = []

    _model_init = partial(model_init_for_tuning)

    for fold_idx, (train_index, test_index) in enumerate(skf.split(texts, labels_np), start=1):
        print(f"\n--- Starting Fold {fold_idx}/{n_splits} ---")

        train_texts = [texts[i] for i in train_index]
        train_labels = [labels[i] for i in train_index]
        test_texts = [texts[i] for i in test_index]
        test_labels = [labels[i] for i in test_index]

        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

        def objective(trial: optuna.Trial):
            trial_params = hp_space(trial)

            trial_args = apply_trial_params_to_args(args, trial_params)

            trainer = Trainer(
                model_init=model_init_for_tuning,
                args=trial_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )
            trainer.train()
            metrics = trainer.evaluate()
            print(metrics)
            return metrics.get("f1", 0.0)

        print("Performing manual Optuna hyperparameter search...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        best_score = study.best_value

        print(f"Best hyperparameters for Fold {fold_idx}: {best_params}")
        print(f"Best score for Fold {fold_idx}: {best_score:.4f}")
        all_best_params.append(best_params)

        final_model_for_fold = _model_init(None)
        final_model_for_fold.to(device)

        best_args = apply_trial_params_to_args(args, best_params)

        print(f"Training Fold {fold_idx} with best hyperparameters completed. Evaluating model...")

        final_model = model_init_for_tuning()
        final_model.to(device)

        final_trainer = Trainer(
            model=final_model,
            args=best_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        final_trainer.train()

        metrics = Model_Validator.evaluate_model(final_model, test_texts, test_labels)
        all_fold_results.append(metrics)

    print("\n--- Cross-Validation and Hyperparameter Tuning Complete ---")
    print("Best hyperparameters found per fold:")
    for i, params in enumerate(all_best_params):
        print(f"Fold {i+1}: {params}")

    print("\nEvaluation results for each fold (with best hyperparameters):")
    for i, results in enumerate(all_fold_results):
        print(f"Fold {i+1}: Accuracy={results['accuracy']:.4f}, F1-Class 1={results['f1_class_1']:.4f}")

    avg_accuracy = np.mean([res['accuracy'] for res in all_fold_results])
    print(f"\nAverage Accuracy across all folds: {avg_accuracy:.4f}")

    return all_fold_results, all_best_params