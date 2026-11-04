import os

def test_dataset():
    assert os.path.exists("data/customer_churn_dataset-testing-master.csv")

def test_model():
    assert os.path.exists("model.pkl")

def test_test_data_created():
    assert os.path.exists("set.pkl")