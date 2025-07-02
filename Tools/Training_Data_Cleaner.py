import shutil
from pathlib import Path
import pandas as pd

def clear_folder(folder: Path):
    if folder.exists() and folder.is_dir():
        for item in folder.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

def filter_training_data(training_data_unfiltered_folder: Path, training_data_filtered_folder: Path, text_columns: list[str],
                         combine_into_column : str, label_column: str, translation: dict):
    if not Path.exists(training_data_unfiltered_folder):
        print(f"Unable to clean training data! {training_data_unfiltered_folder} does not exist")
        return

    if not Path.exists(training_data_filtered_folder):
        print(f"Target folder {training_data_filtered_folder} does not exist. Creating it.")
        training_data_filtered_folder.mkdir(parents=True, exist_ok=True)
    else:
        clear_folder(training_data_filtered_folder)

    for csv_file in training_data_unfiltered_folder.iterdir():
        if not csv_file.is_file() or not csv_file.suffix.lower() == ".csv":
            print(f"{csv_file} is not a csv file! Skipping it!")
            continue

        df = pd.read_csv(csv_file, delimiter=';')
        df = df.replace(translation)

        missing_columns = [col for col in text_columns if col not in df.columns]
        if len(missing_columns) > 0:
            print(f"{csv_file} is missing one of the columns {text_columns}. Skipping it")
            continue

        df_cleaned = df[text_columns + [label_column]].dropna()

        df_cleaned[combine_into_column] = df_cleaned[text_columns].astype(str).agg(" ".join, axis=1)

        df_cleaned = df_cleaned.dropna(subset=[label_column])
        df_cleaned[label_column] = df_cleaned[label_column].astype(str)

        df_cleaned = df_cleaned[[combine_into_column, label_column]]

        df_cleaned.to_csv(training_data_filtered_folder / csv_file.name, index=False, encoding="utf-8")


