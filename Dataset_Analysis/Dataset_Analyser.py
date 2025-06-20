from pathlib import Path
import pandas as pd

def analyse_dataset(path: Path, text_column: str, label_column: str) -> dict:
    if not path.exists() or not path.is_file() or not path.suffix.lower() == ".csv":
        raise FileNotFoundError(f"Dataset file {path} does not exist or is not a csv!")

    df = pd.read_csv(path, delimiter=";", encoding="utf-8")
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Dataset is missing required columns: "
                         f"{text_column}, {label_column}. Available columns: {df.columns.tolist()}")

    unique_labels = df[label_column].unique()

    return{
    "total_examples": len(df),
    "unique_labels": unique_labels.tolist(),
    "unique_labels_amount": len(unique_labels),
    "label_distribution": df[label_column].value_counts().to_dict()
    }

if __name__ == "__main__":
    dataset_path = Path("./Classifications_Product(sample_to_classify).csv")
    text_col = "Description"
    label_col = "Final"

    try:
        analysis = analyse_dataset(dataset_path, text_col, label_col)
        print("Dataset Analysis:", analysis)
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

