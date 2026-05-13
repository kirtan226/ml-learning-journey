import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# =====================================
# Problem : If study hours increase, probability of passing increases.
# =====================================


data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [30, 40, 45, 50, 60, 65, 70, 80, 85, 90],
    "Result": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print(df.head())

x = df[['Study_Hours', 'Attendance']]
y = df['Result']

classifier = KNeighborsClassifier()
classifier.fit(x, y)

study_hours  = float(input("Please enter the study hours: "))
attendance = int(input("Please enter the attendance: "))


user_input = pd.DataFrame(
    [[study_hours, attendance]],
    columns=['Study_Hours', 'Attendance']
)

prediction = classifier.predict(user_input)

labels = ("Fail", "Pass")
predicted_class = prediction[0]
print("Prediction: ", predicted_class ,"-" , labels[predicted_class])

# Separate data based on result
fail = df[df['Result'] == 0]
passed = df[df['Result'] == 1]

plt.scatter(fail['Study_Hours'], fail['Attendance'], label='Fail', marker='x')
plt.scatter(passed['Study_Hours'], passed['Attendance'], label='Pass', marker='o')

# Plot user input point
plt.scatter(study_hours, attendance, label='Your Input', marker='*', s=200)

plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.title("KNN Classification Visualization")
plt.legend()

plt.show()