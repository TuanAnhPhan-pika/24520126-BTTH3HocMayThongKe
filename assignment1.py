from linear_regression import LinearRegression
from pathlib import Path

import pandas as pd
import numpy as np

def organize_data(data: pd.DataFrame):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    return (X, y)

def convert_month2int(month: str) -> int:
    month_dict = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12
    }
    
    return month_dict[month]

def convert_date2int(date: str):
    date_dict = {
        "sun": 1,
        "mon": 2,
        "tue": 3,
        "wed": 4,
        "thu": 5,
        "fri": 6,
        "sat": 7
    }
    
    return date_dict[date]

    
def preprocess_data(df: pd.DataFrame):
    df["month"] = df["month"].map(convert_month2int)
    df["day"] = df["day"].map(convert_date2int)
    df["area"] = np.log1p(df["area"])
    
    return df



if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "data" / "forestfires.csv"
    data = pd.read_csv(data_path)
    data = preprocess_data(data)
    data = data.sample(frac=1, ignore_index=True)
    
    train_data = data.iloc[:414]
    test_data = data.iloc[414:]
    
    X_train, y_train = organize_data(train_data)
    X_test, y_test = organize_data(test_data)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_predicted = lr_model.predict(X_test)
    rmse = lr_model.evaluate(y_test, y_predicted)

    print("The root mean squared error is: ", rmse)
