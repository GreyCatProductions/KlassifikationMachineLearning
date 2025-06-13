import os
from pathlib import Path
import zipfile
from Tools import Renamer


def unzip_all(folder: Path, extract_to: Path):
    if not folder.exists():
        print("Zipped folder does not exist. Creating it")
        os.mkdir("../Github_Data_Ziped")
        return

    zip_files = list(folder.glob("*.zip"))
    if not zip_files:
        print(f"No ZIP files found in: {folder}")
        return

    renamed_zip_files = Renamer.rename_zips(zip_files)

    successes = 0
    failures = 0

    for zip_path in renamed_zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                target_folder = extract_to
                target_folder.mkdir(parents=True, exist_ok=True)
                zip_ref.extractall(target_folder)
                successes += 1
        except:
            print(f"Failed to unzip {zip_path}")
            failures += 1

    print(f"Unzipping finished. Successes = {successes}, Failures = {failures}")