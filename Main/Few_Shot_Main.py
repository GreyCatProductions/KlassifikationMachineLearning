from pathlib import Path
from Tools.Github_Data_Preparation_UI import prepare_data_for_evaluation
from Tools.Model_Usage import FewShot


def main():
    project_root = Path(__file__).resolve().parent
    zips_folder = project_root / "Github_Data_Ziped"
    unziped_folder = project_root / "Github_Data_Unziped"
    extracted_folder = project_root / "Github_Data_Extracted"
    evaluated_folder = project_root / "Github_Data_Evaluated_FewShot"
    relevant_columns = ["Description", "readme"]
    name_column = "RepoName"
    text_column = "Text"

    prepare_data_for_evaluation(zips_folder, unziped_folder, extracted_folder, relevant_columns, name_column, text_column)

    print("Loading model")
    model_to_use = Path("./Few_Shot_V4")
    model = FewShot.load_trained_model(model_to_use)
    print("Model loaded. Classifying data...")
    FewShot.classify(model, extracted_folder, text_column, evaluated_folder)
    print("Finished classification. Results saved to", evaluated_folder)


if __name__ == "__main__":
    main()