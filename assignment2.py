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
    df = df.copy()
    df["month"] = df["month"].map(convert_month2int)
    df["day"] = df["day"].map(convert_date2int)
    df["area"] = np.log1p(df["area"])
    
    return df


def standardize_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    train_data = train_data.copy()
    test_data = test_data.copy()
    continuous_variables = ["FFMC", "DMC", "DC", "ISI", "temp", "wind", "RH", "rain"]
    for variable in continuous_variables:
        mean = train_data[variable].mean()
        std = train_data[variable].std()
        if std == 0:
            std = 1.0
        train_data[variable] = (train_data[variable] - mean) / std
        test_data[variable] = (test_data[variable] - mean) / std
    return train_data, test_data


def train_and_evaluate(X_train, y_train, X_test, y_test):
    X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
    X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model.evaluate(y_test, predictions)


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "data" / "forestfires.csv"
    data = pd.read_csv(data_path)
    data = preprocess_data(data)
    data = data.sample(frac=1, random_state=42, ignore_index=True)
    
    train_data = data.iloc[:414]
    test_data = data.iloc[414:]
    
    X_train, y_train = organize_data(train_data)
    X_test, y_test = organize_data(test_data)
    
    train_data_std, test_data_std = standardize_data(train_data, test_data)
    X_train_std, _ = organize_data(train_data_std)
    X_test_std, _ = organize_data(test_data_std)

    rmse_original = train_and_evaluate(X_train, y_train, X_test, y_test)
    rmse_standardized = train_and_evaluate(
        X_train_std, y_train, X_test_std, y_test
    )

    continuous_variables = ["FFMC", "DMC", "DC", "ISI", "temp", "wind", "RH", "rain"]
    print(pd.DataFrame({
        "Feature": continuous_variables,
        "Mean": train_data_std[continuous_variables].mean().to_numpy(),
        "Variance": train_data_std[continuous_variables].var().to_numpy(),
    }).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Original features:     {rmse_original:.6f}")
    print(f"Standardized features: {rmse_standardized:.6f}")
    print(f"Absolute difference:   {abs(rmse_original - rmse_standardized):.10f}")
    
