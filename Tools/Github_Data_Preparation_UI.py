from pathlib import Path
from Tools import Github_Data_Extractor, Unzipper


def prepare_data_for_evaluation(zips_folder: Path, unziped_folder: Path,
                                extracted_folder: Path, relevant_columns: [str], name_column: str, text_column: str) -> bool:

    zips_folder.mkdir(exist_ok=True)
    unziped_folder.mkdir(exist_ok=True)
    extracted_folder.mkdir(exist_ok=True)

    if len(relevant_columns) == 0:
        print("Relevant columns can not be 0!")
        return False

    print("Unzipping files")
    try:
        Unzipper.unzip_all(zips_folder, unziped_folder)
    except Exception as e:
        print("Failed to Unzip. " + str(e))
        return False


    print("Extracting unzipped files")
    try:
        Github_Data_Extractor.extract_data(unziped_folder, extracted_folder, relevant_columns, name_column, text_column)
    except Exception as e:
        print("Failed to extracted from unzipped files!" + str(e))
        return False

    return True