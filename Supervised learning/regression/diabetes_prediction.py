import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

diabetes = datasets.load_diabetes()

# print(diabetes.keys()) # dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.DESCR)

# diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X = diabetes.data
# print(diabetes_X)

X_train = diabetes_X[:-30]
X_test = diabetes_X[-30:]

y_train = diabetes.target[:-30]
y_test = diabetes.target[-30:]

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Prediction: ", prediction)

mae = mean_absolute_error(y_test, prediction)
print("MAE: ",mae)

mse = mean_squared_error(y_test, prediction)
print("MSE: ",mse)

print("weights: ", model.coef_)
print("intercept: ", model.intercept_)

# plt.scatter(X_test , y_test , c=y_test)
# plt.plot(X_test , prediction, c='red')
# plt.show()

# MAE:  43.49286787904107
# MSE:  3035.060115291269
# weights:  [941.43097333]
# intercept:  153.39713623331644


# ===============
# (2)
# ===============
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn import datasets
# from sklearn.linear_model import LinearRegression
# from matplotlib import pyplot as plt
#
# diabetes = datasets.load_diabetes()
#
# # print(diabetes.keys()) # dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# # print(diabetes.DESCR)
#
# diabetes_X = np.array([[1], [2], [3]])
# # print(diabetes_X)
#
# X_train = diabetes_X
# X_test = diabetes_X
#
# y_train = np.array([3,2,4])
# y_test = np.array([3,2,4])
#
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# prediction = model.predict(X_test)
#
# print("Prediction: ", prediction)
#
# mae = mean_absolute_error(y_test, prediction)
# print("MAE: ",mae)
#
# mse = mean_squared_error(y_test, prediction)
# print("MSE: ",mse)
#
# print("weights: ", model.coef_)
# print("intercept: ", model.intercept_)
#
# plt.scatter(X_test , y_test , c=y_test)
# plt.plot(X_test , prediction, c='red')
# plt.show()
#
# # MAE:  43.49286787904107
# # MSE:  3035.060115291269
# # weights:  [941.43097333]
# # intercept:  153.39713623331644