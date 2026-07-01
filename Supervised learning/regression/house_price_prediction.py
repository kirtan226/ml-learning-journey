import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt

data = pd.read_excel("../files/regression_pricing_data.xlsx")
print(data)

# ==================================================
# Problem 1: Basic Linear Regression
# Predict Price using only Size_sqft
# ==================================================

x_basic = data[['Size_sqft']]
y_basic = data[['Price']]



x_basic_train, x_basic_test, y_basic_train, y_basic_test = train_test_split(
    x_basic, y_basic, test_size=0.2, random_state=42
)

basic_model = LinearRegression()
basic_model.fit(x_basic_train, y_basic_train)

predict_for_square_fit = int(input("Enter size sqft for basic prediction: "))

basic_custom_input = pd.DataFrame(
    [[predict_for_square_fit]],
    columns=["Size_sqft"]
)

basic_prediction = basic_model.predict(basic_custom_input)

print("\n========== Problem 1: Basic Linear Regression ==========")
print(f"Predicted Price for {predict_for_square_fit} sqft:", basic_prediction[0][0])


# ==================================================
# Problem 2: Multiple Linear Regression
# Use Size_sqft, Bedrooms, Age_years, Distance_city_km
# ==================================================

x_multiple = data[['Size_sqft', 'Bedrooms', 'Age_years', 'Distance_city_km']]
y_multiple = data[['Price']]


# ==================================================
# Problem 3: Train/Test Split + Evaluation
# Using Basic Model
# ==================================================

x_multiple_train, x_multiple_test, y_multiple_train, y_multiple_test = train_test_split(
    x_multiple, y_multiple, test_size=0.2, random_state=42
)

multiple_model = LinearRegression()
multiple_model.fit(x_multiple_train, y_multiple_train)

size = int(input("\nEnter size sqft for multiple regression: "))
bedrooms = int(input("Enter bedrooms: "))
age = int(input("Enter age years: "))
distance = int(input("Enter distance from city km: "))

multiple_custom_input = pd.DataFrame(
    [[size, bedrooms, age, distance]],
    columns=['Size_sqft', 'Bedrooms', 'Age_years', 'Distance_city_km']
)

multiple_prediction = multiple_model.predict(multiple_custom_input)

print("\n========== Problem 2: Multiple Linear Regression ==========")
print("Predicted Price:", multiple_prediction[0][0])


basic_y_pred = basic_model.predict(x_basic_test)
print("y basic test ---",y_basic_test )
print("basic_y_pred ---",basic_y_pred )
basic_mae = mean_absolute_error(y_basic_test, basic_y_pred)
basic_mse = mean_squared_error(y_basic_test, basic_y_pred)

print("\n========== Problem 3: Basic Model Evaluation ==========")
print("MAE:", basic_mae)
print("MSE:", basic_mse)


# Multiple model evaluation also
multiple_y_pred = multiple_model.predict(x_multiple_test)

multiple_mae = mean_absolute_error(y_multiple_test, multiple_y_pred)
multiple_mse = mean_squared_error(y_multiple_test, multiple_y_pred)

print("\n========= = Problem 3: Multiple Model Evaluation ==========")
print("MAE:", multiple_mae)
print("MSE:", multiple_mse)



# ==================================================
# Problem 4: Visualization
# Size vs Price + Regression Line + Custom Prediction
# ==================================================
print("\n========= = Problem 4: Visualization ==========")

plt.scatter(x_basic, y_basic, color="blue", label="Actual Price")

plt.plot(
    x_basic,
    basic_model.predict(x_basic),
    color="red",
    label="Regression Line"
)

plt.scatter(
    predict_for_square_fit,
    basic_prediction[0][0],
    color="green",
    s=70,
    label=f"Predicted ({predict_for_square_fit} sqft)"
)

plt.xlabel("Size_sqft")
plt.ylabel("Price")
plt.title("Size vs Price Regression")
plt.ticklabel_format(style='plain', axis='y')
plt.legend()
plt.show()



# ==================================================
# Problem 5: Feature Importance Thinking
# ==================================================

print("\n========== Problem 5: Basic Model Coefficient ==========")
print("Size_sqft Coefficient:", basic_model.coef_[0][0])
print("Intercept:", basic_model.intercept_[0])

print("\n========== Problem 5: Multiple Model Feature Importance ==========")

for feature, coef in zip(x_multiple.columns, multiple_model.coef_[0]):
    print(feature, ":", coef)

print("Intercept:", multiple_model.intercept_[0])