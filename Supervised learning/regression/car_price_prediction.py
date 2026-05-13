import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


'''
👉👉👉 Task : Predict car price based on car features
'''

car_data = pd.read_excel('../files/car_price_regression_dataset.xlsx')
print("===================================")
print("===== Sheet data head =====")
print(car_data.head())
print("===================================")
# print(car_data.columns)

# ===================================
# 👉 Problem 1 (Basic) : Predict price using only: "Engine_cc"
# ===================================

x = car_data[['Engine_cc']]
y = car_data[['Price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(x_train, y_train)

print("===================================")
print("=============== Problem 1 (Basic) ====================")

engine_cc_user_input = int(input("\nEnter car engine cc: "))

engine_cc_column = pd.DataFrame([[engine_cc_user_input]], columns=["Engine_cc"])
basic_prediction = model.predict(engine_cc_column)
print(f"Prediction result for engine cc {engine_cc_user_input}: {basic_prediction}")
print("===================================")

# ===================================
# 👉 Problem 2 (Real-world) Predict price using:
# 'Engine_cc', 'Mileage_kmpl', 'Age_years', 'Brand_value', 'Owner_count'
# ===================================

x_multi = car_data[['Engine_cc', 'Mileage_kmpl', 'Age_years', 'Brand_value', 'Owner_count']]
y_multi = car_data[['Price']]

x_multi_train , x_multi_test, y_multi_train, y_multi_test = train_test_split(x_multi, y_multi, test_size = 0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(x_multi_train, y_multi_train)
print("===================================")
print("================ Problem 2 ===================")
multi_engine_cc_user_input = int(input("\nEnter car engine cc: "))
multi_Mileage_kmpl_user_input = int(input("\nEnter car Milage KMPL: "))
multi_Age_years_user_input = int(input("\nEnter car Age Years: "))
multi_Brand_value_user_input = int(input("\nEnter car Brand Value: "))
multi_Owner_count_user_input = int(input("\nEnter car Owner Count: "))



multi_user_input_columns = pd.DataFrame(
   [[multi_engine_cc_user_input, multi_Mileage_kmpl_user_input, multi_Age_years_user_input, multi_Brand_value_user_input, multi_Owner_count_user_input]],
    columns=["Engine_cc", "Mileage_kmpl", "Age_years", "Brand_value", "Owner_count"])

multiple_data_prediction = multi_model.predict(multi_user_input_columns)
print("Predicted price using multiple data :", multiple_data_prediction)
print("===================================")

# ===================================
# 👉 Problem 3 (Evaluation): Calculate:
# 'MAE', 'MSE', 'RMSE', 'Error %'
# ===================================
print("===================================")
print("================ Problem 3 (Basic) ===================")
basic_y_pred = model.predict(x_test)
MAE = mean_absolute_error(y_test, basic_y_pred)
MSE = mean_squared_error(y_test, basic_y_pred)

print("MAE using basic model : ",MAE)
print("MSE using basic model : ",MSE)


print("===================================")
print("================ Problem 3 (Multiple) ===================")
multi_y_pred = multi_model.predict(x_multi_test)
MAE = mean_absolute_error(y_multi_test, multi_y_pred)
MSE = mean_squared_error(y_multi_test, multi_y_pred)
print("MAE using multiple model : ",MAE)
print("MSE using multiple model : ",MSE)
print("===================================")

# ===================================
# Problem 4 (Visualization): Plot:
# 'Engine vs Price', 'Regression line', 'Add one predicted point'
# ===================================
print("===================================")
print("================ Problem 4: Visualization (Basic) ===================")

plt.scatter(x, y, color="blue", label="Actual Price")

plt.plot(
    x,
    model.predict(x),
    color="red",
    label="Regression Line"
)

plt.scatter(
    engine_cc_user_input,
    basic_prediction[0][0],
    color="green",
    s=70,
    label=f"Predicted ({engine_cc_user_input} cc)"
)

plt.xlabel("Engine CC")
plt.ylabel("Price")
plt.title("Engine CC vs Price Regression")
plt.ticklabel_format(style='plain', axis='y')
plt.legend()
plt.show()
print("===================================")

# ===================================
# Problem 5 (Business Thinking) : Answer
# 👉 Which increases price most?
# 👉 Which decreases price?
# 👉 Why does Owner_count reduce price?
# 👉 Why does Brand_value increase price?
# ===================================

print("===================================")
print("================ Problem 5: Business Thinking ===================")
print("\nFeature Importance (Multiple Model):")

for feature, coef in zip(x_multi.columns, multi_model.coef_[0]):
    print(f"{feature} : {coef}")

print("\nIntercept:", multi_model.intercept_[0])

print("\n👉 Interpretation:")

for feature, coef in zip(x_multi.columns, multi_model.coef_[0]):
    if coef > 0:
        print(f"{feature} increases price")
    else:
        print(f"{feature} decreases price")

print("\n👉 Business Answers:")

print("- Engine_cc increases price because more power cars are expensive")
print("- Mileage_kmpl may increase or decrease depending on model behavior")
print("- Age_years decreases price because older cars lose value")
print("- Brand_value increases price because premium brands cost more")
print("- Owner_count decreases price because more owners reduce trust/value")
print("===================================")
