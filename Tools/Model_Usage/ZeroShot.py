from pathlib import Path
from typing import List

import pandas as pd
from transformers import pipeline
import torch

def load_model(task: str, model: str):
    print("Loading model...")
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(task, model = model, device=device)

def classify(extracted_folder: Path, target_folder: Path, labels: List[str], text_column: str):
    target_folder.mkdir(exist_ok=True)

    print("Loading model")
    classifier = load_model("zero-shot-classification", "facebook/bart-large-mnli")

    for extracted_csv in extracted_folder.iterdir():
        if not extracted_csv.is_file() or not extracted_csv.suffix.lower() == ".csv":
            print(f"{extracted_csv} is not a file or wrong format!")
            continue

        evaluated_df = pd.read_csv(extracted_csv, sep=",", encoding="utf-8")
        print(f"Classifying {extracted_csv.name} with {len(evaluated_df)} rows")

        if text_column not in evaluated_df.columns:
            print(
                f"Column {text_column} not found in {extracted_csv.name}. Available columns: {evaluated_df.columns.tolist()}")
            continue

        results = []
        for idx, row in evaluated_df.iterrows():
            text = row[text_column]
            result = classifier(text, candidate_labels=labels)
            rounded_scores = [round(score, 3) for score in result['scores']]
            score_dict = dict(zip(result['labels'], rounded_scores))
            results.append(score_dict)

        evaluated_df["result"] = results
        output_path = target_folder / extracted_csv.name
        evaluated_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved evaluated file to {output_path}")
