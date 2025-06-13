from pathlib import Path

from Tools.Github_Data_Preparation_UI import prepare_data_for_evaluation
from Tools.Model_Usage import ZeroShot


def main():
    project_root = Path(__file__).resolve().parent
    zips_folder = project_root / "Github_Data_Ziped"
    unziped_folder = project_root / "Github_Data_Unziped"
    extracted_folder = project_root / "Github_Data_Extracted"
    relevant_columns = ["Description", "readme"]
    name_column = "RepoName"
    text_column = "Text"

    prepare_data_for_evaluation(zips_folder, unziped_folder, extracted_folder, relevant_columns, name_column, text_column)

    print("Classifying")
    labels = ["data-driven journalism", "not data-driven"]
    evaluated_folder = project_root / "Github_Data_Evaluated_ZeroShot"
    ZeroShot.classify(extracted_folder, evaluated_folder, labels, text_column)

    print("Finished")

if __name__ == "__main__":
    main()