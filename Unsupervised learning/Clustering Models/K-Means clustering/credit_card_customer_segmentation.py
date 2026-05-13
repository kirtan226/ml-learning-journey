"""
Problem: Credit Card Customer Segmentation

- A bank has thousands of credit card holders. The bank wants to understand how different customers actually use their cards — some
people carry a high balance and pay the minimum, others pay in full every month, some make frequent small purchases, others make rare
large ones.

- Your goal is to use K-Means to group customers by their spending behavior so the bank can offer the right product to the right
customer — e.g. offer installment plans to high-balance users, cashback cards to frequent buyers, etc.

- Again, there are no predefined labels — pure unsupervised learning.
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ========= Load file =========
df = pd.read_csv('../../files/CC_GENERAL.csv')

# print("============ DF Head ============")
# print(df.head())

# print("============ DF INFO ============\n", df.info())
# print("============ DF Description ============\n", df.describe())
# print("============ DF Null Check ============\n", df.isnull().sum())
# print("============ DF Columns ============\n", list(df.columns))

# ========= Remove Unnecessary Columns =========
df.drop("CUST_ID", axis=1, inplace=True)

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Transform Data
scaled_data = pipeline.fit_transform(df)

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,
                    init='k-means++',
                    random_state=42
                     )
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# print("======= Wcss =======\n", wcss)

#============= Plot Elbow Graph =============

plt.figure(figsize=(10, 10))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow plot')
plt.xlabel('Number of clusters')
plt.ylabel('Wcss')
plt.show()

#============= Silhouette Score =============
def check_silhouette_score(scaled_data):
    print("========== Silhouette Score ==========\n")
    for i in range(2,11):
        kmeans = KMeans(
            n_clusters=i,
            init='k-means++',
            random_state=42
        )

        labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        print(f"K={i}  Silhouette Score={score}")

# check_silhouette_score(scaled_data)

# =============  Train K-Means Model =============
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

clusters = kmeans.fit_predict(scaled_data)

df['cluster'] = clusters

#============= Analyze Clusters =============
cluster_analysis = df.groupby('cluster').mean()
print("========== cluster analysis ==========\n",cluster_analysis)

#============= PCA Visualization =============

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
# print(pca_data )

#============= Plot cluster =============
plt.figure(figsize=(8, 8))
plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=clusters
    ,cmap='viridis'
)

plt.title('Credit card customer segmentation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


cluster_names = {
    0: "Premium Responsible Users",
    1: "High Risk Credit Users",
    2: "Dormant Customers",
    3: "Installment Buyers",
    4: "Frequent Shoppers"
}


def predict_customer_cluster():

    print("\n======= Customer Segmentation Prediction =======")

    balance = float(input("Enter Balance: "))
    balance_frequency = float(input("Enter Balance Frequency (0-1): "))
    purchases = float(input("Enter Total Purchases: "))
    oneoff_purchases = float(input("Enter One-off Purchases: "))
    installment_purchases = float(input("Enter Installment Purchases: "))
    cash_advance = float(input("Enter Cash Advance: "))
    purchases_frequency = float(input("Enter Purchases Frequency (0-1): "))
    oneoff_purchase_frequency = float(input("Enter One-off Purchase Frequency (0-1): "))
    purchases_installment_frequency = float(input("Enter Installment Purchase Frequency (0-1): "))
    cash_advance_frequency = float(input("Enter Cash Advance Frequency (0-1): "))
    cash_advance_trx = int(input("Enter Cash Advance Transactions: "))
    purchases_trx = int(input("Enter Purchase Transactions: "))
    credit_limit = float(input("Enter Credit Limit: "))
    payments = float(input("Enter Payments: "))
    minimum_payments = float(input("Enter Minimum Payments: "))
    prc_full_payment = float(input("Enter Percentage Full Payment (0-1): "))
    tenure = int(input("Enter Tenure: "))

    # Create DataFrame
    new_customer = pd.DataFrame([[
        balance,
        balance_frequency,
        purchases,
        oneoff_purchases,
        installment_purchases,
        cash_advance,
        purchases_frequency,
        oneoff_purchase_frequency,
        purchases_installment_frequency,
        cash_advance_frequency,
        cash_advance_trx,
        purchases_trx,
        credit_limit,
        payments,
        minimum_payments,
        prc_full_payment,
        tenure
    ]], columns=df.columns[:-1])

    # Transform using pipeline
    scaled_customer = pipeline.transform(new_customer)

    # Predict cluster
    cluster = kmeans.predict(scaled_customer)[0]

    print("\nPredicted Cluster:", cluster)

    print(
        "Customer Category:",
        cluster_names[cluster]
    )

predict_customer_cluster()