"""
Problem:
Spam Detection using Decision Tree Classifier

Goal:
Predict whether a message/email is spam or ham.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt

original_df = pd.read_csv('../../files/spam.csv', encoding="latin-1")
# print("========== Head data ==========")
# print(df.head())

# print("========== DF describe ==========")
# print(original_df.describe())

# print("========== DF info ==========")
# print(original_df.info())

# Remove extra unused columns
df = original_df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.drop_duplicates()
df = df.dropna()
print("original DF shape:",original_df.shape ,"\nDf shape: ", df.shape)

# rename column names
df.columns = ['label', 'message']
# print("========== Head data ==========")
# print(df.head())

encoder = LabelEncoder()

# encode into "ham": 0, "spam": 1
df['label'] = encoder.fit_transform(df['label'])

x = df['message']
y = df['label']

# train test split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# convert text into numbers
vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english"
    )

x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)
# print("========== convert text into numbers ==========")
# print("X train vectorized:",x_train_vectorized)

# train model
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=30,
    random_state=42
)
model.fit(x_train_vectorized, y_train)


def evaluate_model(model, x_test_vectorized, y_test):
    predictions = model.predict(x_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)
    print("========== Accuracy score ==========\nAccuracy:", accuracy)

    cm = confusion_matrix(y_test, predictions)
    print("========== Confusion matrix ==========\ncm:", cm)

    cr = classification_report(y_test, predictions, target_names=["Ham", "Spam"],
            zero_division=0)
    print("========== Classification report ==========\nclassification_report:", cr)

label_mapping = {
    0: "Ham",
    1: "Spam"
}

def user_input_prediction(model, vectorizer):
    user_message = input("Enter message/email text: ")

    user_message_vectorized = vectorizer.transform([user_message])
    prediction = model.predict(user_message_vectorized)

    print(f"\nPrediction Result: {label_mapping[prediction[0]]}")

def visualize_confusion_matrix(model, x_test_vectorized, y_test):
    predictions = model.predict(x_test_vectorized)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Ham", "Spam"]
    )

    plt.title("Confusion Matrix - Spam Detection")
    plt.show()


def visualize_label_count(df):
    df["label"].value_counts().plot(kind="bar")

    plt.title("Ham vs Spam Count")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()


def visualize_feature_importance(model, vectorizer, top_n=20):
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Word": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n)

    print("\n========== Top Important Words ==========")
    print(importance_df)

    plt.figure(figsize=(12, 6))
    plt.barh(importance_df["Word"], importance_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Words")
    plt.title("Top Important Words for Spam Detection")
    plt.gca().invert_yaxis()
    plt.show()

def visualize_tree(model, vectorizer):
    """
    Visualize Decision Tree for Spam Detection.
    """

    plt.figure(figsize=(25, 12))

    plot_tree(
        model,
        feature_names=vectorizer.get_feature_names_out(),
        class_names=["Ham", "Spam"],
        filled=True,
        max_depth=3,
        fontsize=8
    )

    plt.title("Decision Tree - Spam Detection")
    plt.show()

# Evaluation
evaluate_model(model, x_test_vectorized, y_test)

# Visualization
visualize_confusion_matrix(model, x_test_vectorized, y_test)
visualize_label_count(df)
visualize_feature_importance(model, vectorizer)
visualize_tree(model, vectorizer)

# user input
user_input_prediction(model, vectorizer)