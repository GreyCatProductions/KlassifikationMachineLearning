from pathlib import Path
from Tools import Unzipper
from Helper import Evaluated_Data_Comparer

def main():
    print("Comparer started")

    ziped_evaulated_folder = Path("Ziped_Evaluated_Data")
    unziped_evaluated_folder = Path("./Unziped_Evaluated_Data")

    print("Unzipping evaluated data from", ziped_evaulated_folder, "to", unziped_evaluated_folder)
    Unzipper.unzip_all(ziped_evaulated_folder, unziped_evaluated_folder)
    print("Unzipping completed. Now starting comparison.")

    folders = [child for child in unziped_evaluated_folder.iterdir() if child.is_dir()]
    if len(folders) != 2:
        print(f"Expected 2 folders in {unziped_evaluated_folder}, but found {len(folders)}. Aborting comparison.")
        return

    evaluated_folder_1, evaluated_folder_2 = folders

    print(f"Comparing:\n 1: {evaluated_folder_1}\n 2: {evaluated_folder_2}")

    prediction_column_1 = "prediction"
    confidence_column_1 = "confidence"
    prediction_column_2 = "prediction"
    confidence_column_2 = "confidence"
    Evaluated_Data_Comparer.analyse_difference(evaluated_folder_1, evaluated_folder_2,
                                               prediction_column_1, confidence_column_1,
                                               prediction_column_2, confidence_column_2)


if __name__ == "__main__":
    main()