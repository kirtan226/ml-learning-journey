'''
Problem: Customer Purchase Prediction
- Predict whether a customer will purchase a product or not
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mpl_toolkits.mplot3d import Axes3D

# ===============
# Load data
# ===============
data = pd.read_excel('../../files/customer_purchase_data.xlsx')
# print(data.head())

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
# print("=============")
# print(x)
# print("=============")
# print(y)

# ===============
# Train model
# ===============
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# ===============
# evaluation
# ===============
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ===============
# Load data
# ===============
age = int(input("Enter Customer age: "))
salary = float(input("Enter Customer salary: "))
browsing_time = float(input("Enter browsing Time: "))

user_input = pd.DataFrame([[age, salary, browsing_time]], columns = ['Age', 'Salary', 'Browsing_Time'])
prediction = model.predict(user_input)

label = ("Not Purchased", "Purchased")
print(prediction[0], "-", label[prediction[0]])

# =================
# Probability
# =================
prob = model.predict_proba(user_input)

print("Not Purchased Probability:", round(prob[0][0], 3))
print("Purchased Probability:", round(prob[0][1], 3))


# =================
# Visualization
# =================

not_purchased = data[data["Purchased"] == 0]
purchased = data[data["Purchased"] == 1]

plt.scatter(not_purchased['Age'], not_purchased['Salary'], label='Not Purchased', color='red')
plt.scatter(purchased['Age'], purchased['Salary'], label='Purchased', color='green')

plt.scatter(age, salary, c='blue', label='Input', marker='*', s=150)

plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# =================
# 3D chart
# =================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(not_purchased['Age'], not_purchased['Salary'], not_purchased['Browsing_Time'], color='red', label='Not Purchased')
# ax.scatter(purchased['Age'], purchased['Salary'], purchased['Browsing_Time'], color='green', label='Purchased')
#
# ax.scatter(age, salary, browsing_time, color='blue', s=100, label='Input')
#
# ax.set_xlabel('Age')
# ax.set_ylabel('Salary')
# ax.set_zlabel('Browsing Time')
#
# plt.legend()
# plt.show()