from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import root_mean_squared_error

from assignment2 import organize_data, preprocess_data, standardize_data
from linear_regression import LinearRegression


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "data" / "forestfires.csv"
    data = preprocess_data(pd.read_csv(data_path))
    data = data.sample(frac=1, random_state=42, ignore_index=True)

    train_data = data.iloc[:414]
    test_data = data.iloc[414:]
    train_data, test_data = standardize_data(train_data, test_data)

    X_train, y_train = organize_data(train_data)
    X_test, y_test = organize_data(test_data)

    X_train_bias = np.column_stack((np.ones(X_train.shape[0]), X_train))
    X_test_bias = np.column_stack((np.ones(X_test.shape[0]), X_test))

    manual_model = LinearRegression()
    manual_model.fit(X_train_bias, y_train)
    y_pred_manual = manual_model.predict(X_test_bias)

    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)

    y_actual = np.expm1(y_test)
    rmse_manual = root_mean_squared_error(y_actual, np.expm1(y_pred_manual))
    rmse_sklearn = root_mean_squared_error(y_actual, np.expm1(y_pred_sklearn))

    print(f"RMSE (NumPy):        {rmse_manual:.6f}")
    print(f"RMSE (Scikit-learn): {rmse_sklearn:.6f}")
