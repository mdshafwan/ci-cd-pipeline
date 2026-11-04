import os
import pandas as pd

data_path = 'data/customer_churn_dataset-testing-master.csv'

def validate():
    print('step1: Checking the data')

    if not os.path.exists(data_path):
        print('Error: Dataset was not found')
        exit(1)

    df = pd.read_csv('data/customer_churn_dataset-testing-master.csv')

    if df.empty:
        print("The dataset was empty")
        exit(1)

    null_ratio = df.isnull().sum().sum() / len(df)

    if null_ratio > 0.1:
        print(f"Error : Too many missing value({null_ratio})")
        exit(1)

    print(f"Data quality Passed ! Total rows :{len(df)},Columns:{len(df.columns)}")
    exit(0)

if __name__ == "main":
    validate()