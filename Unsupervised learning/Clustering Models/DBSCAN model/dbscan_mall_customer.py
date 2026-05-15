"""
problem : Mall Customers Revisited with DBSCAN
Model: DBSCAN

- You are the owner of a shopping mall. Through membership cards, you have collected basic data about your customers — their age,
gender, annual income, and a spending score (a value from 1–100 assigned based on how much and how often they spend).

goal: Use K-Means clustering to group customers into distinct segments — for example, "high income, low spenders" vs "low income,
high spenders" — so your marketing team can design targeted strategies for each group.
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============ Load File ============
df = pd.read_csv("../../files/Mall_Customers.csv")
# print("============ DF Head ============")
# print(df.head())

# print("============ DF INFO ============\n", df.info())
# print("============ DF Description ============\n", df.describe())
# print("============ DF Null Check ============\n", df.isnull().sum())
print("============ DF Columns ============\n", list(df.columns))


# ============ Select Feature ============
# x = df[['Annual Income (k$)', 'Spending Score (1-100)']]
# ==== OR ====
x = df.iloc[:,-2:]
feature_columns = x.columns
# print("========== Selected Feature columns ===========\n",feature_columns)


# ============ Pipeline ============
pipeline = Pipeline([
    ('scaler', StandardScaler()),
])

# ============ ScaleData ============
scaled_data = pipeline.fit_transform(x)
# print("====== scaled data ====\n", scaled_data)
# print("====== scaled data ====\n", type(scaled_data))


# ============ Train DBSCAN Model ============
# Note : DBSCAN does not support predict() , transform() , so don't put DBSCAN in Pipeline
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(scaled_data)

df['Cluster'] = clusters

print("\n============ UNIQUE CLUSTERS ============\n")
print(df['Cluster'].unique())

"""
IMPORTANT:
-1 means NOISE / OUTLIER
"""

print("\n============ CLUSTER COUNTS ============\n")
print(df['Cluster'].value_counts())

def visualize_clusters(scaled_data, clusters):

    plt.figure(figsize=(8, 6))

    unique_clusters = sorted(set(clusters))

    for cluster in unique_clusters:

        # Get points of current cluster
        cluster_points = scaled_data[clusters == cluster]

        # Noise cluster handling
        if cluster == -1:
            label = "Noise / Outlier (-1)"
        else:
            label = f"Cluster {cluster}"

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=label
        )

    plt.title("Mall Customer Segmentation using DBSCAN")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")

    plt.legend()
    plt.show()

visualize_clusters(scaled_data, clusters)

# ========== Cluster Analysis ===========
cluster_analysis = df.groupby('Cluster')[
    ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
].mean()

print("\n============ CLUSTER ANALYSIS ============\n")
print(cluster_analysis)


# =========================================================
# CLUSTER MEANINGS
# =========================================================

cluster_names = {
    -1: "Noise / Outlier Customers",
    0: "Regular Customers",
    1: "High Spending Customers",
    2: "Low Spending Customers"
}

