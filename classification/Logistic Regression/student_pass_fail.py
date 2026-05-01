import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import pyplot as plt

file_data = pd.read_excel('../../files/student_pass_fail.xlsx')
# print(file_data.head())

# Features and target
x = file_data[['Study_Hours', 'Attendance']]
y = file_data['Result']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(x,y , test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# print("Predicted:", Y_pred)
# print("Actual:", Y_test.values)

# Evaluation
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# user input prediction
study_hours = float(input("Enter study hours: "))
attendance = int(input("Enter attendance: "))

user_input = pd.DataFrame([[study_hours, attendance]], columns=['Study_Hours', 'Attendance'])
prediction = model.predict(user_input)

labels = ("Fail", "Pass")
print("Prediction:", prediction[0], "-", labels[prediction[0]])


# Probability (IMPORTANT)
prob = model.predict_proba(user_input)

print("Fail Probability:", prob[0][0])
print("Pass Probability:", prob[0][1])

# Visualization
fail = file_data[file_data['Result'] == 0]
passed = file_data[file_data['Result'] == 1]

plt.scatter(fail['Study_Hours'], fail['Attendance'], label='Fail', marker='x')
plt.scatter(passed['Study_Hours'], passed['Attendance'], label='Pass', marker='o')

# Plot user input point
plt.scatter(study_hours, attendance, label='Your Input', marker='*', s=200)

plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.title("Data Distribution")
plt.legend()
plt.show()