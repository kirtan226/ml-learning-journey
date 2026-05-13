import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# sample data
data = {
    "Experience": [1, 2, 3, 4, 5],
    "Salary": [20000, 25000, 30000, 35000, 40000]
}

df = pd.DataFrame(data)
print(df)

x = df[["Experience"]]
y = df["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)
print("X train =====",x_train)
print("X test =====",x_test)
print("y train =====",y_train)
print("Y test =====",y_test)
model = LinearRegression()
model.fit(x_train, y_train)

prediction = model.predict([[3.3]])
print("Predicted:", prediction)
print("Actual:", y_test.values)


mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)

print("MAE:", mae)
print("MSE:", mse)

plt.scatter(x, y)
plt.plot(x, model.predict(x), color='red')
# plt.show()