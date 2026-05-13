'''
Problem : Predict whether a flower is Iris virginica or not
'''

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


iris = load_iris()

# print(iris)
# print(list(iris.keys())) # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
# print(iris['data'].shape)

x = iris['data'][:, 3:]
y = (iris['target'] == 2 ).astype(int)
print("=============")
print(x)
print("=============")
print(y)

model = LogisticRegression()
model.fit(x, y)

prediction = model.predict([[2.5]])
print("=============")
print(prediction)

# visualizations

# Random number array
array = np.linspace(0, 3, 101).reshape(-1, 1)
print("=============")
# print(array)

# probability to check generated array
prob = model.predict_proba(array)
print("=============")
print(prob)

plt.plot(array, prob[:,1], "g-", label="Iris virginica")
plt.show()