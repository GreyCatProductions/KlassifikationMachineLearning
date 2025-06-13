from pathlib import Path
from typing import List


def rename(csv_file: Path) -> str:
    base_name = csv_file.stem
    return base_name.split("-")[0]

def rename_zips(zips: List[Path]):
    seen = {}
    renamed_files = []

    for zip_file in zips:
        prefix = rename(zip_file)
        count = seen.get(prefix, 0)

        if count == 0:
            new_name = f"{prefix}.zip"
        else:
            new_name = f"{prefix}_{count}.zip"

        seen[prefix] = count + 1

        new_path = zip_file.with_name(new_name)
        zip_file.rename(new_path)
        renamed_files.append(new_path)
    return renamed_files