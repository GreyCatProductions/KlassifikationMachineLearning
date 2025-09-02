from pathlib import Path

from setfit import TrainingArguments

from Tools import Training_Initializer, Training_Data_Cleaner

def prepare_data(training_data_filtered, combined_into_column, label_column):
    training_data_unfiltered = Path("Training_Data_Unfiltered")
    text_columns = ["RepoName", "Description", "readme"] #Those get combined into one column
    translation = {
#        "Yes": "Data Journalism",
#        "No": "Not Data Journalism",
#        "Halb": "Half Data Journalism"
    }

    print("Filtering training data")
    Training_Data_Cleaner.filter_training_data(training_data_unfiltered,
                                               training_data_filtered, text_columns,
                                               combined_into_column, label_column, translation)

def main():
    data_already_prepared = True
    base_model_location = Path("downloaded_model")
    model_save_location = Path("Trained_Model_Stackshare_and_Github")
    training_data_filtered = Path("Training_Data_Filtered") #dataset path
    combined_column = "text" #text column from dataset
    label_column = "label" #label column from dataset

    if not data_already_prepared:
        prepare_data(training_data_filtered, combined_column, label_column)

    cross_validate = True
    if cross_validate:
        print("Starting optuna cross validation training...")
        settings = {
            "n_trials": 30,
            "n_splits": 5,
            "average": "weighted"
        }
        Training_Initializer.start_cross_validation_training_with_optuna(training_data_filtered, combined_column,
                                                                         label_column, base_model_location, model_save_location, settings)
    else:
        print("Starting standard training...")
        parameters = TrainingArguments(
            num_epochs=5,
            batch_size=16,
            num_iterations=12,
            head_learning_rate=1.0719864040714887e-05,
            save_strategy="no",
            eval_strategy="no", #no eval needed for this training
            use_amp=True
        )
        Training_Initializer.start_training(training_data_filtered, combined_column, label_column, model_save_location, parameters)

    print(f"Finished training. Saved model to {model_save_location}")

if __name__ == "__main__":
    main()