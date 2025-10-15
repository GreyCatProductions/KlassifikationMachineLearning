from pathlib import Path
from Tools.Model_Usage import FewShot


def main():
    data_folder = Path("./Training_Data_Filtered") #dataset path
    column_to_classify = "text" #text column from dataset
    model_to_use = Path("Trained_Model_Data_Journalism") #path to the trained model
    output_folder = Path("./Classified_Output")

    print("Checking for valid paths")
    if not data_folder.exists():
        print("Data folder does not exist. Cant classify")
        return

    if not model_to_use.exists():
        print("Model folder does not exist. Cant classify")
        return

    print("Loading model..")
    model = FewShot.load_trained_model(model_to_use)
    print("Starting classification...")
    FewShot.classify(model, data_folder, column_to_classify, output_folder)
    print("Finished. Results saved to", output_folder)

if __name__ == "__main__":
    main()