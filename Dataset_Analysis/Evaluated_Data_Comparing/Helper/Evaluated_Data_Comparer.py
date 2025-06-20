from pathlib import Path
import pandas as pd

def _verify(file1:Path, file2:Path) -> bool:
    if not file1.is_file() or not file1.suffix.lower() == ".csv":
        print(f"{file1} is not a csv file! Skipping it!")
        return False

    if not file2.exists():
        print(f"File {file2.name} does not exist in the second folder! Skipping it!")
        return False

    return True

def _verify_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    if list(df1.columns) != list(df2.columns):
        return False

    if len(df1) != len(df2):
        return False
    return True

def _verify_columns(df: pd.DataFrame, prediction_column: str, confident_column: str) -> bool:
    if prediction_column not in df.columns or confident_column not in df.columns:
        print(f"Columns {prediction_column} or {confident_column} not found in the DataFrame!")
        return False
    return True


def analyse_difference(evaluated_folder_1: Path, evaluated_folder_2: Path,
                       prediction_column_1: str, confident_column_1: str,
                       prediction_column_2: str, confident_column_2: str):
    if not evaluated_folder_1.exists() or not evaluated_folder_2.exists():
        print("One of the evaluated folders does not exist!")
        return

    all_average_confidences_1 = []
    all_average_confidences_2 = []
    amount_true_predictions_1 = 0
    amount_true_predictions_2 = 0
    amount_false_predictions_1 = 0
    amount_false_predictions_2 = 0
    total_attunement = 0
    total_difference = 0
    file_count = 0

    for file1 in evaluated_folder_1.iterdir():
        file2 = evaluated_folder_2 / file1.name

        if not _verify(file1, file2):
            continue

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        if not _verify_similarity(df1, df2):
            print(f"Files {file1.name} and {file2.name} have different structures or lengths.")
            continue

        if not _verify_columns(df1, prediction_column_1, confident_column_1) or not _verify_columns(df2,
                                                                                                    prediction_column_2,
                                                                                                    confident_column_2):
            print(f"Columns {prediction_column_1} or {confident_column_1} not found in {file1.name} or {file2.name}!")
            continue

        average_confidence_1 = df1[confident_column_1].mean()
        average_confidence_2 = df2[confident_column_2].mean()

        all_average_confidences_1.append(average_confidence_1)
        all_average_confidences_2.append(average_confidence_2)
        amount_true_predictions_1 += df1[prediction_column_1].value_counts().get(1, 0)
        amount_true_predictions_2 += df2[prediction_column_2].value_counts().get(1, 0)
        amount_false_predictions_1 += df1[prediction_column_1].value_counts().get(0, 0)
        amount_false_predictions_2 += df2[prediction_column_2].value_counts().get(0, 0)

        file_count += 1

        for index, row in df1.iterrows():
            prediction1 = row[prediction_column_1]
            confidence1 = row[confident_column_1]
            prediction2 = df2.at[index, prediction_column_2]
            confidence2 = df2.at[index, confident_column_2]

            if prediction1 == prediction2:
                total_attunement += 1
            else:
                total_difference += 1

    if file_count == 0:
        print("No valid file pairs found.")
        return

    text_output = f"""
--- Overall Analysis Results ---

Compared Files: {file_count}
Evaluated Folder 1: {evaluated_folder_1.name}
Evaluated Folder 2: {evaluated_folder_2.name}

[PREDICTIONS]
True predictions in Folder 1: {amount_true_predictions_1}
True predictions in Folder 2: {amount_true_predictions_2}
False predictions in Folder 1: {amount_false_predictions_1}
False predictions in Folder 2: {amount_false_predictions_2}
Attunement: {total_attunement}
Difference: {total_difference}
Compliance: {(total_attunement / (total_attunement + total_difference)) * 100:.2f}%

[CONFIDENCE]
Average Confidence in Folder 1: {sum(all_average_confidences_1) / len(all_average_confidences_1):.2f}
Average Confidence in Folder 2: {sum(all_average_confidences_2) / len(all_average_confidences_2):.2f}

Total Matching Predictions: {total_attunement}
Total Different Predictions: {total_difference}
"""

    with open("analysis_results.txt", "w") as file:
        file.write(text_output)

    print("Overall analysis saved to 'analysis_results.txt'")





