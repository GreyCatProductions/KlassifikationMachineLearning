from pathlib import Path
from Tools import Training_Initializer, Training_Data_Cleaner

def prepare_data(training_data_filtered, combined_into_column, label_column):
    training_data_unfiltered = Path("Training_Data_Unfiltered")
    text_columns = ["RepoName", "Description", "readme"] #Those get combined into one column
    label0 = ["No", "Halb"]
    label1 = "Yes"

    print("Filtering training data")
    Training_Data_Cleaner.filter_training_data(training_data_unfiltered,
                                               training_data_filtered, text_columns,
                                               combined_into_column, label_column, label0, label1)

def main():
    data_already_prepared = True
    model_save_location = Path("Trained_Model")
    training_data_filtered = Path("Training_Data_Filtered") #dataset path
    combined_column = "text" #text column from dataset
    label_column = "label" #label column from dataset

    settings = {
    "n_trials": 10,
    "n_splits": 5,
    "average": "weighted"
    }

    if not data_already_prepared:
        prepare_data(training_data_filtered, combined_column, label_column)

    print("Starting to train")
    Training_Initializer.start_training(training_data_filtered, combined_column,
                                        label_column, model_save_location, settings)

    print(f"Finished training. Saved model to {model_save_location}")

if __name__ == "__main__":
    main()