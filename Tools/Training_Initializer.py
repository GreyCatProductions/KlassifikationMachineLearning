import copy
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from pandas import read_csv
from Tools import Cross_Validator_V2, CSV_Tools
from Tools.Model_Usage import FewShot
from setfit import Trainer

def _combine_datasets(training_data_folder: Path, text_column: str, label_column: str):
    if not training_data_folder.exists():
        raise FileNotFoundError(f"training data folder {training_data_folder} does not exist! Can not train!")

    combined = []
    for dataset in training_data_folder.iterdir():
        if not CSV_Tools.verify_csv_structure(dataset):
            print(f"{dataset} is not a csv file! Skipping it!")
            continue

        df = read_csv(dataset, sep=None, engine="python")

        valid = CSV_Tools.verify_columns(df, [text_column, label_column])
        if not valid:
            print(f"{dataset} does not contain the required columns: {text_column}, {label_column}. Skipping it!")
            continue

        combined.append(df)
    return combined

def start_training(training_data_folder: Path, text_column: str, label_column: str, model_save_location: Path,
                   settings: dict):
    n_splits = settings.get("n_splits")
    n_trials = settings.get("n_trials")
    average = settings.get("average")

    combined_datasets = _combine_datasets(training_data_folder, text_column, label_column)

    if not combined_datasets:
        print("No valid datasets found. Aborting.")
        return

    full_df = pd.concat(combined_datasets, ignore_index=True)
    print(f"Loaded {len(full_df)} total examples for training.")

    texts: list[str] = full_df[text_column].astype(str).tolist()
    labels: list[str] = full_df[label_column].astype(str).tolist()

    if not torch.cuda.is_available():
        print("Warning: No GPU available. Training will be done on CPU, which may be slow.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("CUDA available:", torch.cuda.is_available())

    print("Starting cross validation with optuna with", n_splits, "folds and", n_trials, "trials")
    print(f"Amount of labels: {len(labels)}, Amount of texts: {len(texts)}")

    best_params_per_fold = Cross_Validator_V2.cross_validate_with_optuna(texts, labels, n_splits, n_trials, average)
    print("Cross validation completed.")

    return
    if all_best_params_per_fold:
        print("\n--- Average Best Hyperparameters Across Folds ---")

        avg_params = {}
        for param_name in all_best_params_per_fold[0].keys():
            values = [p[param_name] for p in all_best_params_per_fold]
            if isinstance(values[0], (int, float)):
                avg_params[param_name] = sum(values) / len(values)
            else:
                counts = Counter(values)
                avg_params[param_name] = counts.most_common(1)[0][0]
        print(avg_params)

        final_model_args = copy.deepcopy(base_args)
        for k, v in avg_params.items():
            if hasattr(final_model_args, k):
                setattr(final_model_args, k, v)
            if k == "batch_size":
                final_model_args.setfit_batch_size = v
            elif k == "num_iterations":
                final_model_args.num_iterations = v
            elif k == "num_epochs":
                final_model_args.num_epochs = v
        print("\nFinal model will be trained with these average parameters.")
    else:
        print("No best hyperparameters found from cross-validation. Training final model with base_args.")
        final_model_args = base_args

    print("Training final model on full dataset...")
    final_model = FewShot.load_model()
    final_model.to(device)

    final_dataset = Dataset.from_pandas(full_df.rename(columns={text_column: "text", label_column: "label"}))

    final_trainer = Trainer(model=final_model, args=final_model_args, train_dataset=final_dataset)
    final_trainer.train()

    model_save_location.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(str(model_save_location))
    print(f"Final model trained on full dataset and saved at {model_save_location}.")




