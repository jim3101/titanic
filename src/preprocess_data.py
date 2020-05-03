import pandas as pd


def load_raw_data(path_to_csv):
    return pd.read_csv(path_to_csv)

def preprocess_raw_data(df):
    return df