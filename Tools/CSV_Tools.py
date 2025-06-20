from pathlib import Path
import pandas as pd
from pandas import DataFrame


def verify_csv_structure(path: Path) -> bool:
    return path.exists() and path.is_file() and path.suffix.lower() == ".csv"

def verify_columns(input_data, required_columns: list[str]) -> bool:
    if isinstance(input_data, Path):
        if not input_data.exists() or not input_data.is_file() or input_data.suffix.lower() != ".csv":
            print(f"File {input_data} does not exist or is not a CSV file.")
            return False
        try:
            df = pd.read_csv(input_data, sep=None, engine="python")
        except Exception as e:
            print(f"Error reading file {input_data}: {e}")
            return False
    elif isinstance(input_data, DataFrame):
        df = input_data
    else:
        raise TypeError("verify_columns() expected a Path or DataFrame")

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False

    return True
