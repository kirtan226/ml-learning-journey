"""
Problem: Mall Customer Segmentation
Model: K-Means clustering

- You are the owner of a shopping mall. Through membership cards, you have collected basic data about your customers — their age,
gender, annual income, and a spending score (a value from 1–100 assigned based on how much and how often they spend).

goal: Use K-Means clustering to group customers into distinct segments — for example, "high income, low spenders" vs "low income,
high spenders" — so your marketing team can design targeted strategies for each group.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../../files/Mall_Customers.csv")
# print("============ DF Head ============")
# print(df.head())

# print("============ DF INFO ============\n", df.info())
# print("============ DF Description ============\n", df.describe())
# print("============ DF Null Check ============\n", df.isnull().sum())
# print("============ DF Columns ============\n", list(df.columns))

# ============ Select Features =================
# x = df[['Annual Income (k$)', 'Spending Score (1-100)']]
# ===== OR ======
x = df.iloc[:,-2:]

# ============ Scale Data =================
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)
# print("======== Scaled X ========\n", scaled_x)

# =============  Find Best Number of Clusters  -> Elbow Method  =============
wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )

    kmeans.fit(scaled_x)
    wcss.append(kmeans.inertia_)

print("======== Wcss =======\n",wcss)

# =============  Plot Elbow Graph =============
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.show()

# =============  Train K-Means Model =============
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

y_kmeans = kmeans.fit_predict(scaled_x)

# ============= Add Clusters To Dataset =============
df['Cluster'] = y_kmeans
print(df.head())


# ============= Visualize Clusters =============
plt.figure(figsize=(10,6))

plt.scatter(
    scaled_x[:,0],
    scaled_x[:,1],
    c=y_kmeans,
    cmap='viridis'
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    c='red',
    label='Centroids'
)

plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.legend()
plt.show()

# # ============= Understand Each Cluster =============
print(
    df.groupby('Cluster')[
        ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
    ].mean()
)

'''
Cluster	    Meaning
0	        Normal customers
1	        VIP high-value customers
2	        Young impulsive spenders
3	        Rich but inactive customers
4	        Low-value customers
'''
cluster_names = {
    0: "Normal customers",
    1: "VIP high-value customers",
    2: "Young impulsive spenders",
    3: "Rich but inactive customers",
    4: "Low-value customers"
}


# ============ Custom Customer ============

print("======= Custom predictions ========")
income = float(input("Enter the annual income: "))
spending_score = float(input("Enter the annual spending score: "))

new_customer = pd.DataFrame(
    [[income, spending_score]],
    columns=['Annual Income (k$)', 'Spending Score (1-100)']
)
# Scale new data
scaled_customer = scaler.transform(new_customer)

# Predict cluster
predicted_cluster = kmeans.predict(scaled_customer)

print("Predicted Cluster:", predicted_cluster[0])


print(
    "Customer Category:",
    cluster_names[predicted_cluster[0]]
)