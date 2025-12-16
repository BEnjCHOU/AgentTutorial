from pathlib import Path
from fastapi import UploadFile
import shutil
import os


def check_data_folder_exists() -> bool:
    """Check if the data folder exists."""
    data_path = Path("./data")
    return data_path.exists() and data_path.is_dir()

def create_data_folder() -> bool:
    """Create the data folder if it does not exist."""
    data_path = Path("./data")
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating data folder: {e}")
        return False

def save_file_to_data_folder(file: UploadFile) -> tuple[bool, Path, str: None]:
    """Save an uploaded file to the data folder."""
    try:
        data_path = Path("./data")
        data_path.mkdir(parents=True, exist_ok=True)
        file_path = data_path / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return True, file_path, None
    except Exception as e:
        print(f"Error saving file: {file.filename}; {e}")
        return False, file_path, str(e)

def delete_file(filename: str):
    """Deletes file from disk. Note: Deleting from vector store requires DocID tracking."""
    filepath = Path("data") / filename
    if filepath.exists():
        os.remove(filepath)
        # Full vector deletion implementation requires tracking ref_doc_ids 
        # or recreating the index, but we'll stick to file deletion for now.
        print(f"File {filename} deleted from disk.")
    else:
        print("The file does not exist.")

