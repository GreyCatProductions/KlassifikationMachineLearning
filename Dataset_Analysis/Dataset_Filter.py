from pathlib import Path
import pandas as pd

from Tools import CSV_Tools


def filter_csv_by_label(target: Path, target_label_column: str, labels_to_filter_by: list[str]):
    if not CSV_Tools.verify_csv_structure(target):
        raise FileNotFoundError(f"Target file {target} does not exist or is not a csv!")

    df = pd.read_csv(target, sep=None, engine="python")

    before = len(df)
    filtered_df = df[df[target_label_column].isin(labels_to_filter_by)]
    after = len(filtered_df)

    print(f"Filtered {before} rows down to {after} rows")

    return filtered_df

if __name__ == "__main__":
    target_file = Path("./stackshare_labeled.csv")
    filter_file = Path("./Classifications_Product(sample_to_classify).csv")
    output_folder = Path("./filtered_dataset")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = Path(f"{output_folder}/{target_file.name}_filtered.csv")

    filter_label_column = "Final"
    if not CSV_Tools.verify_columns(filter_file, [filter_label_column]):
        raise ValueError(f"Filter file {filter_file} is missing required column: {filter_label_column}")

    df = pd.read_csv(filter_file, sep=None, engine="python")
    labels_to_filter = df[filter_label_column].unique().tolist()

    target_label_column = "label"
    if not CSV_Tools.verify_columns(target_file, [target_label_column]):
        raise ValueError(f"Target file {target_file} is missing required column: {target_label_column}")

    try:
        filtered_data = filter_csv_by_label(target_file, target_label_column, labels_to_filter)
        filtered_data = filtered_data.dropna(how="all")
        with output_path.open('w', encoding='utf-8', newline='') as f:
            filtered_data.to_csv(f, index=False)
    except Exception as e:
        print(f"Error filtering dataset: {e}")