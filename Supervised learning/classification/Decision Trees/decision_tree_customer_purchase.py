'''
Problem : Predict whether a customer will purchase a product or not based on:
- Age
- Annual Salary
- Browsing Time
- Previous Purchases
- Product Category

'''

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

file_df = pd.read_excel('../../files/customer_purchase_decision_tree_dataset.xlsx')
print("=========== File DF ===========")
# print(file_df)
print(file_df.head())

print("=========== Check for null values ===========")
# print(file_df.isnull().sum())

print("=========== Describe File ===========")
# print(file_df.describe())

category_encoder = LabelEncoder()
target_encoder = LabelEncoder()

file_df['Product_Category'] = category_encoder.fit_transform(file_df['Product_Category'])
file_df['Purchased'] = target_encoder.fit_transform(file_df['Purchased'])

x = file_df.drop("Purchased", axis=1)
y = file_df["Purchased"]

# print("=========== value of X ===========")
# print(x)
#
# print("=========== value of Y ===========")
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify=y)

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    min_samples_split=2,
    random_state=42
)
model.fit(x_train, y_train)

def evaluate_model(model, x_test, y_test):
    """
    Tests model performance.
    """

    predictions = model.predict(x_test)

    print("\n========== Model Evaluation ==========")
    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

evaluate_model(model, x_test, y_test)

category_mapping = {
    1 : 'Fashion',
    2 : 'Electronics',
    3 : 'Furniture',
}

def pridict_customer(model, category_encoder, target_encoder):

    print("\n========== Customer Purchase Prediction based on user input ==========")
    age = int(input("Please Enter the Age: "))
    annual_salary = float(input("Please Enter the annual salary: "))
    browsing_time = float(input("Please Enter the browsing time: "))
    previous_purchases = int(input("Please Enter the previous purchases count: "))

    print("\nProduct Categories:")
    print("1 -> Fashion")
    print("2 -> Electronics")
    print("3 -> Furniture")

    product_category_number = int(
        input("Enter product category number: ")
    )

    # Convert number -> category name
    product_category_name = category_mapping[product_category_number]


    encoded_category = category_encoder.transform([product_category_name])[0]

    user_data = pd.DataFrame(
        [[age, annual_salary, browsing_time, previous_purchases, encoded_category ]],
        columns=['Age', 'Annual_Salary', 'Browsing_Time', 'Previous_Purchases', 'Product_Category']
    )

    predictions = model.predict(user_data)
    result = target_encoder.inverse_transform(predictions)

    print("\nPrediction Result:", result[0])

pridict_customer(model, category_encoder, target_encoder)


def show_feature_importance(model, x_train):
    """
    Shows which features are important for prediction.
    """

    importance_df = pd.DataFrame({
        "Feature": x_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\n========== Feature Importance ==========")
    print(importance_df)

show_feature_importance(model, x_train)

def visualize_tree(model, x_train):
    """
    Visualizes Decision Tree.
    """

    plt.figure(figsize=(16, 8))

    plot_tree(
        model,
        feature_names=x_train.columns,
        class_names=["No", "Yes"],
        filled=True
    )

    plt.title("Decision Tree - Customer Purchase Prediction")
    plt.show()

visualize_tree(model, x_train)