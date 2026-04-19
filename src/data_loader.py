import os
import pandas as pd

def get_data_path() -> str:
    """
    Returns the absolute path to the student_data.csv file.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "student_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    return data_path

def load_student_data() -> pd.DataFrame:
    """
    Loads the student study dataset as a pandas DataFrame.
    """
    csv_path = get_data_path()
    df = pd.read_csv(csv_path)
    return df

if __name__ == "__main__":
    df = load_student_data()
    print(df.head())

