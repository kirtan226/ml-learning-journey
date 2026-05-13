"""
Project: Telecom Customer Churn Prediction
Model: Random Forest Classifier

Goal:
Predict whether a telecom customer will churn or not using
customer details, churn/billing data, and internet service data.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pandas.api.types import is_object_dtype, is_string_dtype

churn_df = pd.read_csv('../../files/churn_data.csv')
customer_df = pd.read_csv('../../files/customer_data.csv')
internet_df = pd.read_csv('../../files/internet_data.csv')

# print(" =========== Churn df =========== ", churn_df.shape)
# print(churn_df.head(2))
# print(churn_df.isnull().sum())

# print(" =========== customer df =========== ", customer_df.shape)
# print(customer_df.head(2))
# print(customer_df.isnull().sum())

# print(" =========== internet service df =========== ", internet_df.shape)
# print(internet_df.head(2))
# print(internet_df.isnull().sum())

# merge all files in on DF

df = churn_df.merge(customer_df, how='inner', on='customerID')
df = df.merge(internet_df, how='inner', on='customerID')
# print(" =========== merged df =========== ", df.shape)
# print(df.info())

# ========= Data cleanings
df = df.drop(['customerID'], axis=1)
# print(df['TotalCharges'])

# charges column contains data in str, so convert into numeric s
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# check for null value
# print(df.isnull().sum())

df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
# print(" =========== Null check after fill null values =========== ")
# print(df.isnull().sum())

encoder = LabelEncoder()
df['Churn'] = encoder.fit_transform(df['Churn'])

x = df.drop(['Churn'], axis=1)
y = df['Churn']

x = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42,class_weight='balanced')
model.fit(x_train, y_train)

def evaluation(x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n======= Accuracy =======", accuracy)
    print("\n======= Confusion Matrix =======", cm)
    print("\n======= Classification Report =======",report)

def feature_importance(x, model):
    importances = model.feature_importances_
    feature_names = x.columns

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)
    print("=========== Feature importance ===========")
    print(feature_importance_df)
    return feature_importance_df


def featureimportancechart(feature_importance_df):

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance - Customer Churn Prediction")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

evaluation(x_test, y_test)

feature_importance_df = feature_importance(x, model)

featureimportancechart(feature_importance_df)


# single_customer = x_test.iloc[[0]]
# single_prediction = model.predict(single_customer)
#
# print("\nSingle Customer Prediction:")
#
# if single_prediction[0] == 1:
#     print("Customer is likely to Churn")
# else:
#     print("Customer is likely to Stay")
def predict_customer_churn():
    print("\n======= Enter Customer Details =======")

    tenure = int(input("Enter tenure: "))
    monthly_charges = float(input("Enter Monthly Charges: "))
    total_charges = float(input("Enter Total Charges: "))
    senior_citizen = int(input("Senior Citizen (0/1): "))

    phone_service = input("Phone Service (Yes/No): ")
    contract = input("Contract (Month-to-month/One year/Two year): ")
    paperless_billing = input("Paperless Billing (Yes/No): ")
    payment_method = input(
        "Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): "
    )

    gender = input("Gender (Male/Female): ")
    partner = input("Partner (Yes/No): ")
    dependents = input("Dependents (Yes/No): ")

    multiple_lines = input("Multiple Lines (Yes/No/No phone service): ")
    internet_service = input("Internet Service (DSL/Fiber optic/No): ")

    online_security = input("Online Security (Yes/No/No internet service): ")
    online_backup = input("Online Backup (Yes/No/No internet service): ")
    device_protection = input("Device Protection (Yes/No/No internet service): ")
    tech_support = input("Tech Support (Yes/No/No internet service): ")

    streaming_tv = input("Streaming TV (Yes/No/No internet service): ")
    streaming_movies = input("Streaming Movies (Yes/No/No internet service): ")

    user_data = pd.DataFrame([{
        'tenure': tenure,
        'PhoneService': phone_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
    }])

    # Convert categorical columns
    user_data = pd.get_dummies(user_data)

    # Match training columns
    user_data = user_data.reindex(columns=x.columns, fill_value=0)

    prediction = model.predict(user_data)

    print("\n======= Prediction Result =======")

    if prediction[0] == 1:
        print("Customer is likely to Churn")
    else:
        print("Customer is likely to Stay")

predict_customer_churn()