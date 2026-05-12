"""
Problem : Car Speed vs Fuel Efficiency
Model   : Polynomial Regression using LinearRegression
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


DATA_PATH = "../../files/car_speed_mpg_dataset.xlsx"
df = pd.read_excel(DATA_PATH, skiprows=2)

# print("============ DF Head ============")
# print(df.head())
# print("\n============ DF Describe ============")
# print(df.describe())
# print("\n============ DF Null Check ============")
# print(df.isnull().sum())

df = df.iloc[1:]  # drop row 0 → the description string row ✅
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["mpg"])
df = df.reset_index(drop=True)

print("============ DF Head ============")
print(df.head())

x = df.drop("mpg", axis=1)
y = df["mpg"]

x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )


pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("linear_regression", LinearRegression()),
    ])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)


def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    y_test_values = np.array(y_test)
    y_pred_values = np.array(y_pred)
    non_zero_mask = y_test_values != 0
    mape = np.mean(
        np.abs((y_test_values[non_zero_mask] - y_pred_values[non_zero_mask]) / y_test_values[non_zero_mask])
    ) * 100

    print("============ Model Evaluation ============")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f} mpg")
    print(f"RMSE     : {rmse:.2f} mpg")
    print(f"MSE      : {mse:.2f}")
    print(f"MAPE     : {mape:.1f}%")


def plot_predictions(x_test, y_test, y_pred, pipeline):
    plt.figure(figsize=(8, 5))
    plt.scatter(x_test["speed_mph"], y_test, label="Actual MPG", alpha=0.7)

    # Build a full DataFrame with all columns, fill others with median
    speed_range = np.linspace(x_test["speed_mph"].min(), x_test["speed_mph"].max(), 300)

    x_line = pd.DataFrame(
        np.tile(x_test.median().values, (300, 1)),  # fill all cols with median
        columns=x_test.columns
    )
    x_line["speed_mph"] = speed_range  # only vary speed_mph

    y_line = pipeline.predict(x_line)
    plt.plot(speed_range, y_line, color="red", lw=2, label="Predicted MPG")

    plt.xlabel("Speed (mph)")
    plt.ylabel("Fuel Efficiency (mpg)")
    plt.title("Car Speed vs Fuel Efficiency")
    plt.legend()
    plt.show()


def custom_prediction(pipeline, feature_columns):
    print("\n============ Custom MPG Prediction ============")

    values = []
    for column in feature_columns:
        value = float(input(f"Enter {column}: "))
        values.append(value)

    custom_car = pd.DataFrame([values], columns=feature_columns)
    predicted_mpg = pipeline.predict(custom_car)[0]
    print(f"Predicted fuel efficiency: {predicted_mpg:.2f} mpg")


evaluate_model(y_test, y_pred)
plot_predictions(x_test, y_test, y_pred, pipeline)
custom_prediction(pipeline, x_train.columns.tolist())

