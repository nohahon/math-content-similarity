""" Processes the zbmath_open dataset and creates a new dataset for training embeddings.
doi.org/10.5281/zenodo.6448360
"""
import pandas as pd
import ast
from tqdm import tqdm
import argparse
import os
import logging

tqdm.pandas()
logging.basicConfig(
    level=logging.INFO,
    format="\x1b[32;1m" + "%(message)s" + "\x1b[0m",
)


def preprocess_mscs(msc_list):
    """
    Preprocesses a list of MSC codes by converting it to a string and replacing single quotes with double quotes.
    If the input is already in list format, it is returned as is. Otherwise, it is wrapped in a list.

    Args:
    msc_list (list or str): The list of MSC codes to preprocess.

    Returns:
    str: The preprocessed list of MSC codes as a string.
    """
    msc_list = str(msc_list).strip().replace("'", '"')
    if "[" in msc_list:
        return msc_list
    else:
        return str([f"{msc_list}"])


def detect_general_mscs(msc_list):
    """
    Detects if a list of MSC codes contains general codes.

    Args:
    msc_list (list): A list of MSC codes.

    Returns:
    bool: True if the list contains general codes, False otherwise.
    """
    for msc in ast.literal_eval(msc_list):
        if "-" in str(msc) or "xx" in str(msc):
            return True
    return False


def get_length_msc(msc_list):
    """
    Returns the length of a list of MSC (Mathematics Subject Classification) codes.

    Args:
    msc_list (str): A string representation of a list of MSC codes.

    Returns:
    int: The length of the list of MSC codes.
    """
    msc_list = ast.literal_eval(msc_list)
    return len(msc_list)


def load_data(file_path):
    """
    Load data from a CSV or Parquet file.

    Args:
        file_path (str): The path to the file to load.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    logging.info("Loading data...")
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    return df


def clean_data(d):
    """
    Cleans the input DataFrame by removing NaN values, cleaning MSCs, and removing outliers.

    Args:
        d (pandas.DataFrame): Input DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """

    print("Removing NaNs...")
    len_a = len(d)
    d = d[~d.text.isna()]
    d = d[~d.title.isna()]
    d = d[~d.msc.isna()]
    len_b = len(d)
    print(f"Removed {len_a - len_b} rows.")

    print("Cleaning MSCs...")
    len_a = len(d)
    d["msc"] = d["msc"].progress_apply(preprocess_mscs)
    d = d[~d["msc"].apply(detect_general_mscs)]
    len_b = len(d)
    print(f"Removed {len_a - len_b} rows.")

    print("Removing outliers...")
    len_a = len(d)
    d["text_len"] = d["text"].progress_apply(lambda x: len(x))
    q_low = d["text_len"].quantile(0.15)
    q_hi = d["text_len"].quantile(0.85)
    d = d[(d["text_len"] < q_hi) & (d["text_len"] > q_low)]
    len_b = len(d)
    print(f"Removed {len_a - len_b} rows.")

    return d


def transform_data(d):
    """
    Transforms the input data by merging on MSCs, removing duplicate pairs, and removing pairs with too short MSCs.

    Args:
        d (pandas.DataFrame): Input data with columns 'msc', 'de', and 'en'.

    Returns:
        pandas.DataFrame: Transformed data with columns 'msc', 'de_x', 'de_y', 'en_x', 'en_y', and 'msc_len'.
    """

    print("Merging on MSCs...")
    merged = d.merge(d, on="msc", suffixes=("_x", "_y"))
    merged = merged[merged.de_x != merged.de_y]
    merged = merged[merged.de_x < merged.de_y]
    print(f"Created {len(merged)} pairs.")

    print("Removing pairs with too short MSCs...")
    len_a = len(merged)
    merged["msc_len"] = merged["msc"].progress_apply(get_length_msc)
    merged_final = merged[merged["msc_len"] >= 2]
    len_b = len(merged_final)
    print(f"Removed {len_a - len_b} pairs.")
    print(f"Created final number of {len(merged_final)} pairs.")
    return merged_final


def create_splits(df):
    """
    Creates train, dev, and test splits from the input DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'msc', 'de_x', 'de_y', 'en_x', 'en_y', and 'msc_len'.

    Returns:
        pandas.DataFrame: Train split.
        pandas.DataFrame: Dev split.
        pandas.DataFrame: Test split.
    """
    print("Creating splits...")
    df = df.sample(frac=1, random_state=42)
    train = df.iloc[: int(len(df) * 0.8)]
    dev = df.iloc[int(len(df) * 0.8) : int(len(df) * 0.9)]
    test = df.iloc[int(len(df) * 0.9) :]
    return train, dev, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        nargs="?",
        help="The path to the file to process.",
        default=f"{os.getcwd()}/data/out.csv",
    )

    args = parser.parse_args()

    df = load_data(args.file_path)
    df = clean_data(df)
    df = transform_data(df)

    train, dev, test = create_splits(df)

    logging.info("Saving data...")
    train.to_parquet(args.file_path.replace(".csv", "_train.parquet"))
    dev.to_parquet(args.file_path.replace(".csv", "_dev.parquet"))
    test.to_parquet(args.file_path.replace(".csv", "_test.parquet"))
    logging.info("Done.")


if __name__ == "__main__":
    main()
