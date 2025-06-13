import os
from pathlib import Path

import pandas as pd
from setfit import SetFitModel

def load_model():
    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
    default_save_path = Path("./downloaded_model")

    # 1. Check if a local scratch path is provided via environment variable (for HPC)
    hpc_local_path_str = os.getenv("SETFIT_MODEL_PATH")
    if hpc_local_path_str:
        hpc_local_path = Path(hpc_local_path_str)
        if hpc_local_path.exists() and any(hpc_local_path.iterdir()):
            print(f"Loading model from HPC local scratch: {hpc_local_path}")
            return SetFitModel.from_pretrained(str(hpc_local_path))
        else:
            print(f"Warning: SETFIT_MODEL_PATH set to {hpc_local_path}, but model not found or empty.")
    else:
        print("SETFIT_MODEL_PATH not set. Skipping HPC local scratch check.")

    # 2. Fallback to default local download path
    if default_save_path.exists() and any(default_save_path.iterdir()):
        print(f"Loading model from default local path: {default_save_path}.")
        return SetFitModel.from_pretrained(str(default_save_path))
    else:
        # 3. Download the model if not found anywhere
        print(f"Downloading model {model_name}...")
        model = SetFitModel.from_pretrained(model_name)
        print(f"Saving model to default local path: {default_save_path}...")
        model.save_pretrained(default_save_path)
        return model

def load_trained_model(save_path: Path) -> SetFitModel:
    if not save_path.exists():
        raise FileNotFoundError(f"Model save path {save_path} does not exist.")

    print(f"Loading trained model from {save_path}")

    model = SetFitModel.from_pretrained(str(save_path))
    return model

def classify(model: SetFitModel, extracted_folder: Path, text_column: str, output_folder: Path):
    if not extracted_folder.exists():
        print("Extracted folder does not exist. Cant classify")
        return None

    output_folder.mkdir(parents=True, exist_ok=True)

    for csv_file in extracted_folder.iterdir():
        if not csv_file.is_file() or not csv_file.suffix.lower() == ".csv":
            print(f"{csv_file} is not a csv file! Skipping it!")
            continue

        df = pd.read_csv(csv_file)
        if text_column not in df.columns:
            print(f"Column {text_column} not found in {csv_file}. Available columns: {df.columns.tolist()} Skipping.")
            continue

        texts = df[text_column].astype(str).tolist()

        probabilities = model.predict_proba(texts)

        predictions = []
        confidences = []

        for probs in probabilities:
            pred_label = int(probs.argmax())
            confidence = probs[pred_label]
            predictions.append(pred_label)
            confidences.append(round(float(confidence), 2))

        df['prediction'] = predictions
        df['confidence'] = confidences

        output_path = output_folder / csv_file.name
        df.to_csv(output_path, index=False)
