import pandas as pd
from tkinter import filedialog


def select_csv_file():
    """
    Opens a file dialog and allows the user to select a CSV file.
    """
    file_path = filedialog.askopenfilename(
        title="Select CSV Dataset",
        filetypes=[
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
    )

    return file_path


def load_csv(file_path):
    """
    Loads the selected CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df

    except Exception as e:
        raise ValueError(f"Unable to load CSV file.\n\nError: {e}")