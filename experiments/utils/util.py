import os

def create_folder(folder_path):
    """
    Create a folder if it doesn't exist and return its path
    Args:
        folder_path: Path to the folder to create
    Returns:
        str: Path to the created/existing folder
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path 