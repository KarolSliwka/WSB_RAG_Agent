import shutil
from pathlib import Path

def save_uploaded_file(uploaded_file, dest_folder: Path) -> Path:
    """
    Saves an uploaded file to the destination folder.
    """
    dest_folder.mkdir(parents=True, exist_ok=True)
    file_path = dest_folder / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def move_to_processed(processed_dir, file_path: Path) -> Path:
    """
    Copies a file to the processed directory without removing it from its source.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / file_path.name
    shutil.copy2(file_path, dest)
    return dest
