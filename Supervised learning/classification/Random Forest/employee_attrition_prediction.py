"""
Project: Employee Attrition Prediction
Model: Random Forest Classifier

Goal: Predict whether an employee will Leave or Stay.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


df = pd.read_excel('../../files/employee_attrition_dataset.xlsx')
# print("=========== File data ===========")
# print(df.head())

# print("=========== Describe ===========")
# print(df.describe())

# print("=========== Info ===========")
print(df.info())

# print("=========== Null value check ===========")
# print(df.isnull().sum())

# encode Categorical column
remote_encoder = LabelEncoder()
overtime_encoder = LabelEncoder()
attrition_encoder = LabelEncoder()

df['Remote_Work'] = remote_encoder.fit_transform(df['Remote_Work'])
print("\nRemote_Work classes:", remote_encoder.classes_)

df['Overtime'] = overtime_encoder.fit_transform(df['Overtime'])
print("\nOvertime classes:", overtime_encoder.classes_)

df['Attrition'] = attrition_encoder.fit_transform(df['Attrition'])
print("\nAttrition classes:", attrition_encoder.classes_)

# x & Y feature-target split
x = df.drop(['Attrition'], axis=1)
y = df['Attrition']

# print("\n========== Features ==========")
# print(x.head())
# print("\n========== Target ==========")
# print(y.head())

# train-test data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,)

# pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
])

model = pipeline.fit(x_train, y_train)

# predictions
y_pred = pipeline.predict(x_test)
# print(y_pred)

def evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)

    print("\n========== Accuracy ==========")
    print(accuracy)

    print("\n========== Confusion Matrix ==========")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n========== Classification Report ==========")
    print(classification_report(y_test, y_pred))



def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title("Employee Attrition Confusion Matrix")
    plt.show()


def get_feature_importance():
    rf_model = pipeline.named_steps["classifier"]

    importance_df = pd.DataFrame({
        "Feature": x.columns,
        "Importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========== Feature Importance ==========")
    print(importance_df)

    return importance_df


def feature_importance_graph(importance_df):
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xticks(rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.show()

# print("=========== columns ==============")
# columns = list(df.columns)
# print(columns)

attrition_mapping = {
    0:'Leave',
    1:'Stay',
}

def predict_employee_data():
    age = int(input("Enter Age: "))
    salary = float(input("Enter Salary: "))
    experience_years = float(input("Enter Experience Years: "))
    working_hours = float(input("Enter Working Hours: "))
    project_handeled = int(input("Enter Project Handed: "))
    satisfaction_score = float(input("Enter Satisfaction Score: "))
    remote_work = int(input("Enter Remote Work | No-0 | Yes-1 : "))
    overtime = int(input("Enter Overtime | No-0 | Yes-1: "))

    new_employee = pd.DataFrame(
        [[age, salary, experience_years, working_hours, project_handeled, satisfaction_score, remote_work, overtime]],
        columns=['Age', 'Salary', 'Experience_Years', 'Work_Hours_Per_Day', 'Projects_Handled',
                 'Satisfaction_Score', 'Remote_Work', 'Overtime']
    )

    prediction = pipeline.predict(new_employee)
    print("Employee may: ", attrition_mapping[prediction[0]])

# evaluation
evaluation(y_test, y_pred)

display_confusion_matrix(y_test, y_pred)

importance_df = get_feature_importance()

feature_importance_graph(importance_df)

predict_employee_data()