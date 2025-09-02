from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from pandas import read_csv, DataFrame
from Tools import Cross_Validator_V2, CSV_Tools
from Tools.Model_Usage import FewShot
from setfit import Trainer, SetFitModel, TrainingArguments


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

        print(f"{dataset.name} loaded with", len(df), "examples.")
        combined.append(df)
    return combined

def start_cross_validation_training_with_optuna(training_data_folder: Path, text_column: str, label_column: str, model_to_use: Path, model_save_location: Path,
                                                settings: dict):
    n_splits = settings.get("n_splits")
    n_trials = settings.get("n_trials")
    average = settings.get("average")

    print(f"Loading data from {training_data_folder} with text column '{text_column}' and label column '{label_column}'")

    combined_datasets = _combine_datasets(training_data_folder, text_column, label_column)

    if not combined_datasets:
        print("No valid datasets found. Aborting.")
        return

    full_df = pd.concat(combined_datasets, ignore_index=True)
    print(f"Loaded {len(full_df)} total examples for training.")

    texts: list[str] = full_df[text_column].astype(str).tolist()
    labels: list[str] = full_df[label_column].astype(str).tolist()

    print_info(n_splits, n_trials, labels)

    best_params: dict[str, any] = Cross_Validator_V2.cross_validate_with_optuna(model_to_use, texts, labels, n_splits, n_trials, average)
    print("Cross validation completed.")

    print("Training final model on full dataset with optimal fold average parameters...")
    model_save_location.mkdir(parents=True, exist_ok=True)

    final_arguments = TrainingArguments(
        num_epochs=best_params["num_epochs"],
        batch_size=best_params["batch_size"],
        num_iterations=best_params["num_iterations"],
        head_learning_rate=best_params["head_lr"],
        save_strategy="no",
        eval_strategy="epoch",
        use_amp=True
    )

    final_model = train_final_model(model_save_location, full_df, text_column, label_column, final_arguments)
    final_model.save_pretrained(str(model_save_location))

    print(f"Final model trained on full dataset and saved at {model_save_location}.")

def start_training(training_data_folder: Path, text_column: str, label_column: str, model_save_location: Path,
                   parameters: TrainingArguments):

    print(f"Loading data from {training_data_folder} with text column '{text_column}' and label column '{label_column}'")

    combined_datasets = _combine_datasets(training_data_folder, text_column, label_column)

    if not combined_datasets:
        print("No valid datasets found. Aborting.")
        return

    full_df = pd.concat(combined_datasets, ignore_index=True)
    print(f"Loaded {len(full_df)} total examples for training.")

    print(f"Training model with parameters {parameters}")

    final_model = train_final_model(model_save_location, full_df, text_column, label_column, parameters)
    model_save_location.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(str(model_save_location))

    print(f"Final model trained on full dataset and saved at {model_save_location}.")


def print_info(n_splits, n_trials, labels):
    if not torch.cuda.is_available():
        print("Warning: No GPU available. Training will be done on CPU, which may be slow.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("CUDA available:", torch.cuda.is_available())

    print("Starting cross validation with optuna with", n_splits, "folds and", n_trials, "trials")
    print("Filtering dones. Loaded texsts and labels")
    print("Labels:", set(labels))

def train_final_model(model_save_location: Path, full_df: DataFrame, text_column: str, label_column: str, params: TrainingArguments) -> SetFitModel:
    final_model = FewShot.load_model(model_save_location)

    final_dataset = Dataset.from_pandas(full_df.rename(columns={text_column: "text", label_column: "label"}))
    final_trainer = Trainer(model=final_model, args=params, train_dataset=final_dataset)
    final_trainer.train()
    return final_model
