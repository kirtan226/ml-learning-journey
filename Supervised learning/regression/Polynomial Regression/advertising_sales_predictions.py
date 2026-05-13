"""
Problem : Advertising Spend vs Revenue Prediction
Model   : Polynomial Regression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from matplotlib import pyplot as plt

df = pd.read_csv('../../files/Advertising_Budget_and_Sales.csv')
df = df.drop(['Unnamed: 0'], axis=1)

print("============ DF Head ============")
print(df.head())

print("============ DF Columns ============", df.columns)

print("============ DF Describe ============", df.describe())
print("============ DF info ============", df.info())
# print("============ DF null Check ============", df.isnull().sum())

# Split in feature and target

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())

])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)


def evaluate_model(pipeline, x_train, y_train, x_test, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    y_test_values = np.array(y_test)
    y_pred_values = np.array(y_pred)

    non_zero_mask = y_test_values != 0

    mape = np.mean(
        np.abs(
            (
                y_test_values[non_zero_mask]
                - y_pred_values[non_zero_mask]
            )
            /
            y_test_values[non_zero_mask]
        )
    ) * 100

    train_score = pipeline.score(x_train, y_train)
    test_score = pipeline.score(x_test, y_test)

    print("\n============ Model Evaluation ============")

    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"MSE      : {mse:.2f}")
    print(f"MAPE     : {mape:.2f}%")

    print("\n============ Pipeline Score ============")
    print(f"Train Score : {train_score:.4f}")
    print(f"Test Score  : {test_score:.4f}")

def plot_predictions(pipeline, x_train, y_train, x_test, y_test, y_pred):
    plt.figure(figsize=(8, 6))

    plt.scatter(x_test["TV Ad Budget ($)"],
        y_test,
        alpha=0.7,
        label="Actual Sales"
    )

    tv_range = np.linspace(
        x_test["TV Ad Budget ($)"].min(),
        x_test["TV Ad Budget ($)"].max(),
        300
    )

    x_line = pd.DataFrame(
        np.tile(x_test.median().values, (300, 1)),
        columns=x_test.columns
    )

    x_line["TV Ad Budget ($)"] = tv_range
    y_line = pipeline.predict(x_line)

    plt.plot(
        tv_range,
        y_line,
        linewidth=2,
        label="Predicted Sales"
    )

    plt.xlabel("TV Advertising Spend")
    plt.ylabel("Sales Revenue")
    plt.title("Advertising Spend vs Sales")

    plt.legend()
    plt.show()


def custom_prediction(model):

    print("\n============ Custom Prediction ============")
    tv = float(input("Enter TV advertising spend: "))
    radio = float(input("Enter Radio advertising spend: "))
    newspaper = float(input("Enter Newspaper advertising spend: "))

    custom_data = pd.DataFrame([[
        tv,
        radio,
        newspaper
    ]], columns=["TV Ad Budget ($)", "Radio Ad Budget ($)", "Newspaper Ad Budget ($)"])

    prediction = model.predict(custom_data)[0]
    print(f"\nPredicted Sales Revenue: {prediction:.2f}")



evaluate_model(pipeline,x_train, y_train, x_test, y_test, y_pred )
plot_predictions(pipeline,x_train, y_train, x_test, y_test, y_pred)
custom_prediction(pipeline)
