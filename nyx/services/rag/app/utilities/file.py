# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import shutil
from pathlib import Path
# External:
from fastapi import UploadFile


# ==============================
# FUNCTIONS
# ==============================

async def save_temp_file(
    temp_path:str|Path,
    file:UploadFile
) -> Path:
    """
    Save the given file in a temporal directory.

    Args:
        temp_path (str|Path): Where to save the file.
        file (UploadFile): File to save.

    Returns:
        Path: Path where the file was saved.
    """

    # Checks if file object is correct.
    if file.filename is None: raise ValueError("File doesn't contains filename property.")
    
    # Creates Path instance to manage access.
    dest_path:Path = Path(temp_path) / file.filename

    # Save file content.
    await file.seek(0)
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Returns path.
    return dest_path