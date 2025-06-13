import pandas as pd
from pathlib import Path
from typing import List

def extract_data(unzipped_folder: Path, target: Path, relevant_columns: List[str], name_column = str, text_column = str):
    target.mkdir(parents=True, exist_ok=True)

    for organization in unzipped_folder.iterdir():
        name = organization.name
        organization_repos_csv = organization / "organization_repos.csv"

        try:
            df = pd.read_csv(organization_repos_csv, sep=";", encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {organization_repos_csv}: {e}")
            continue

        missing = [col for col in relevant_columns if col not in df.columns]
        if missing:
            print(f"missing column: {missing} - {organization}")
            continue

        df[text_column] = df[relevant_columns].astype(str).agg(" ".join, axis=1)
        combined = df[[text_column]].copy()
        combined["repo_name"]= df[name_column]

        output_path = target / f"{name}.csv"
        combined.to_csv(output_path, index=False, encoding="utf-8")



