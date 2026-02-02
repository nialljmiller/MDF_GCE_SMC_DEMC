# standalone_plots/io.py
import os
import pandas as pd

def load_combined_table(path):
    """
    Load the combined_full_evaluation_table.* into a DataFrame.

    path can be a directory (containing the PKL) or the PKL/CSV file itself.
    No error handling on purpose, per your rules.
    """
    if os.path.isdir(path):
        pkl_path = os.path.join(path, "combined_full_evaluation_table.pkl")
        csv_path = os.path.join(path, "combined_full_evaluation_table.csv")
        if os.path.exists(pkl_path):
            df = pd.read_pickle(pkl_path)
        else:
            df = pd.read_csv(csv_path)
    else:
        if path.endswith(".pkl"):
            df = pd.read_pickle(path)
        else:
            df = pd.read_csv(path)

    # make sure loss exists and is numeric
    if "loss" not in df.columns and "fitness" in df.columns:
        df["loss"] = df["fitness"]

    df["loss"] = pd.to_numeric(df["loss"])
    return df
