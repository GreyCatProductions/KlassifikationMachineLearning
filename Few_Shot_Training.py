from pathlib import Path
from Tools import Training_Initializer, Training_Data_Cleaner


def main():
    training_data_unfiltered = Path("Training_Data_Unfiltered")
    training_data_filtered = Path("Training_Data_Filtered")
    text_columns = ["RepoName", "Description", "readme"] #Those get combined into one column
    combined_into_column = "combined_text"
    label_column = "Solved"
    model_save_location = Path("Trained_Model")
    label0 = ["No", "Halb"]
    label1 = "Yes"

    print("Filtering training data")
    Training_Data_Cleaner.filter_training_data(training_data_unfiltered,
                                               training_data_filtered, text_columns, combined_into_column, label_column, label0, label1)

    print("Starting to train")
    Training_Initializer.start_training(training_data_filtered, combined_into_column, label_column, model_save_location)
    print(f"Finished training. Saved model to {model_save_location}")

if __name__ == "__main__":
    main()