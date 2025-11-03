from pathlib import Path
import pandas as pd

path = Path("Classifications_Data_Align(classification_tim).csv")
raw_df = pd.read_csv(path, sep=None, engine="python")

labeler_one = "label Tim"
labeler_two = "label Kai"
undef_value = "Code"
decision_col = "decision"

df = raw_df.copy()

for idx, row in df.iterrows():
    val1 = row[labeler_one]
    val2 = row[labeler_two]

    if pd.isna(row[decision_col]):
        if not pd.isna(val1) and not pd.isna(val2) and val1 != val2:
            print(f"Warning: Row {idx} conflict: {val1} vs {val2}")
            continue

        if pd.isna(val1) and pd.isna(val2):
            df.at[idx, decision_col] = undef_value
            continue

        df.at[idx, decision_col] = val1 if not pd.isna(val1) else val2

out_path = path.with_name(path.stem + "_filled.csv")
df.to_csv(out_path, index=False)
print(f"Saved df to: {out_path}")