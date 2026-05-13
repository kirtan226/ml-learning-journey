"""
# Support Vector Machine (SVM)

Titanic Survival Prediction Project
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

df = pd.read_excel('../../files/titanic_survival_dataset.xlsx')
# print("=============== DF Head ===============\n",df.head())

# print("=============== DF Describe ===============\n",df.describe())

print("=============== DF Info ===============\n")
df.info()
# print(list(df.columns))

# print("=============== DF Null value count ===============\n", df.isnull().sum())

# features and target split
x = df.drop("Survived", axis=1)
y = df["Survived"]

# numerical and categorical features split for
numerical_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch"
]

categorical_features = [
    "Sex",
    "Embarked",
    "Pclass"
]


numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessors = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model_pipeline = Pipeline([
    ("preprocessor", preprocessors),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_pipeline.fit(x_train, y_train)
y_pred = model_pipeline.predict(x_test)
y_prob = model_pipeline.predict_proba(x_test)[:, 1]

def evaluate_model(y_test, y_pred):
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

def plot_roc_curve(y_true, y_prob):

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.show()

evaluate_model(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_prob)

mapping = {
    0: "Did NOT Survive",
    1: "Survive"
}

def custom_input_passenger():
    print("\n=============== Custom Passenger Prediction ===============")
    pclass = int(input(
                """
        Select Passenger Class:
        1 -> First Class
        2 -> Second Class
        3 -> Third Class
        
        Enter choice: """
            ))

    sex_choice = int(input(
                """
        Select Gender:
        1 -> Male
        2 -> Female
        
        Enter choice: """
            ))

    age = float(input("Enter Age: "))
    sibsp = int(input("Enter SibSp count: "))
    parch = int(input("Enter Parch count: "))
    fare = float(input("Enter Fare amount: "))

    embarked_choice = int(input(
                """
        Select Embarked Port:
        1 -> C (Cherbourg)
        2 -> Q (Queenstown)
        3 -> S (Southampton)
        
        Enter choice: """
    ))

    # Mapping values
    sex_mapping = {
        1: "male",
        2: "female"
    }

    embarked_mapping = {
        1: "C",
        2: "Q",
        3: "S"
    }

    # Convert numerical input to actual categories
    sex = sex_mapping.get(sex_choice)
    embarked = embarked_mapping.get(embarked_choice)

    # Create dataframe
    custom_input = pd.DataFrame(
        [[pclass, sex, age, sibsp, parch, fare, embarked]],
        columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    )

    # Prediction
    custom_prediction = model_pipeline.predict(custom_input)[0]
    prediction_probability = model_pipeline.predict_proba(custom_input)[0][1]
    print("\n=============== Prediction Result ===============")

    print(f"Passenger will '{mapping[custom_prediction]}'")
    print(f"Prediction Probability: {prediction_probability:.2f}")

custom_input_passenger()