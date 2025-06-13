import optuna
from datasets import Dataset
from setfit import Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
from Tools.Model_Usage import FewShot


def cross_validate_with_optuna(texts: list[str], labels: list[int], n_splits=3, n_trials=5):
    def objective(trial):
        num_iterations = trial.suggest_int("num_iterations", 1, 8)
        epochs = trial.suggest_int("num_epochs", 1, 4)
        batch_size: int = trial.suggest_categorical("batch_size", [2, 4, 8])

        print(f"Starting trial {trial.number} with params: "
              f"num_iterations={num_iterations}, num_epochs={epochs}, batch_size={batch_size}")


        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []

        for fold, (train_index, test_index) in enumerate(kfold.split(texts, labels)):
            print(f"Fold {fold + 1}/{n_splits}: Train indices {len(train_index)}, Test indices {len(test_index)}")

            train_texts = [texts[i] for i in train_index]
            train_labels = [labels[i] for i in train_index]
            test_texts = [texts[i] for i in test_index]
            test_labels = [labels[i] for i in test_index]

            train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
            test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

            model = FewShot.load_model()

            arguments = TrainingArguments(
                num_epochs = epochs,
                batch_size=batch_size,
                num_iterations=num_iterations,
                save_strategy="no"
            )

            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                metric=lambda y_true, y_pred: {"f1" : f1_score(y_true, y_pred, average="binary")},
                column_mapping={"text": "text", "label": "label"} , # Ensures proper mapping
                args=arguments
            )

            trainer.train()
            result = trainer.evaluate()
            scores.append(result["f1"])

        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    print("Best Trial:")
    print(study.best_trial)
    return study.best_params
