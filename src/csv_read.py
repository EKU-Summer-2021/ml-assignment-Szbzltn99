import pandas as pd


def csv_read(file):
    """
    reading the csv file
    """
    csv_df = pd.read_csv(file)
    return csv_df
