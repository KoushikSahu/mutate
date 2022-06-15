from csv import QUOTE_NONE
import csv
from config import Config
import pandas as pd
import csv


def eda():
    train_df = pd.read_csv(Config.train_path, sep='\t', quoting=csv.QUOTE_NONE)
    valid_df = pd.read_csv(Config.validate_path,
                           sep='\t', quoting=csv.QUOTE_NONE)

    print(train_df.head())
    print(valid_df.head())
