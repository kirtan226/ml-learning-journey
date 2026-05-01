from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data

# ==============================================
# Option (1) : using dataset provided by sklearn
# ==============================================

iris = load_iris()
# print("======= ",iris.DESCR)
feature =iris.data
target = iris.target


print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

print('======== feature =========', feature[0])
print('======== target =========', target[0])

# ==============================================
# Option (2) : Use data from csv file
# ==============================================

# iris_data = pd.read_csv("../files/IRIS.csv")
# print(iris_data.head())
#
# feature = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
# target = iris_data['species']
#
# encoder = LabelEncoder()
# target = encoder.fit_transform(target)
# # print("encoded_target :", target)

# Classifier
classifier = KNeighborsClassifier()
classifier.fit(feature, target)


# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

sepal_length = float(input("Please Enter the sepal length(cm): "))
sepal_width = float(input("Please Enter the sepal width (cm): "))
petal_length = float(input("Please Enter the petal length (cm): "))
petal_width = float(input("Please Enter the petal width (cm): "))

#  ====== For option (1) ======
prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# ====== For option (2) ======
# input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
#                           columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
# prediction = classifier.predict(input_data)


# print(type(prediction))

labels = ("Iris setosa", "Iris versicolor", "Iris virginica")
predicted_class = prediction[0]
# print(predicted_class)
print("========================")
print(predicted_class, ":", labels[predicted_class])
print("========================")
