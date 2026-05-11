"""
# Support Vector Machine (SVM)

Problem Statement:
Predict whether a customer will buy a product
based on customer-related features.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

df = pd.read_excel('../../files/svm_customer_dataset.xlsx')
# print("=============== DF Head ===============\n",df.head())

# print("=============== DF Describe ===============\n",df.describe())

# print("=============== DF Info ===============\n")
# df.info()

# print("=============== DF Null value count ===============\n", df.isnull().sum())

# Feature and target
x = df.drop("Will_Buy_Product", axis=1)
y = df["Will_Buy_Product"]

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])

svm_pipeline.fit(x_train, y_train)

y_pred = svm_pipeline.predict(x_test)
y_prob = svm_pipeline.predict_proba(x_test)[:, 1]

def evaluation_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("=============== Accuracy score ===============\n" , accuracy)

    cm = confusion_matrix(y_test, y_pred)
    print("=============== Confusion matrix ===============\n" , cm)

    clfr = classification_report(y_test, y_pred)
    print("=============== Classification report ===============\n" , clfr)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    plt.imshow(cm)

    plt.title("Confusion matrix")
    plt.colorbar()

    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

def plot_roc_curve(y_test, y_prob):

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.show()


evaluation_model(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_prob)

mapping = {
    0: 'Not buy',
    1: 'Buy',
}

def custom_input_predictions():
    print("=============== Custom predictions ===============\n")

    age =int(input("Enter your age: "))
    annual_income = float(input("Enter your annual income: "))
    spending_score = float(input("Enter your spending score: "))
    credit_score = float(input("Enter your credit score: "))
    work_experience_score = float(input("Enter your work experience score: "))
    savings_score = float(input("Enter your savings score: "))

    prediction_data = pd.DataFrame(
        [[age,annual_income, spending_score, credit_score, work_experience_score, savings_score]],
        columns= ["Age", "Annual_Income", "Spending_Score", "Credit_Score", "Work_Experience", "Savings"]
    )

    prediction = svm_pipeline.predict(prediction_data)
    print(f"Customer will '{mapping[prediction[0]]}' the product : ", prediction[0])

    prediction_probability = svm_pipeline.predict_proba(prediction_data)[0][1]
    print("=============== prediction probability ===============\n" , prediction_probability)

custom_input_predictions()