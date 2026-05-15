"""
problem : Customer Location Clustering with Outlier Detection
Model: DBSCAN

Problem Statement:
- A ride-booking company wants to identify high-demand customer areas
  using customer booking locations.

Goal:
- Group nearby customer locations into clusters
- Detect isolated booking locations (outliers/noise)
- Understand hotspot regions for better business planning
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============ Load File ============
df = pd.read_csv("../../files/car_delhi.csv",  sep=';')
# print("============ DF Head ============")
# print(df.head())

# print("============ DF INFO ============\n", df.info())
# print("============ DF Description ============\n", df.describe())
# print("============ DF Null Check ============\n", df.isnull().sum())
print("============ DF Columns ============\n", list(df.columns))

# ============ Select Feature ============

# x = df[['lat', 'lng']]
# ====== OR =====
x  = df.iloc[:,2:4]
feature_column = x.columns
# print("========== Selected Feature columns ===========\n",feature_column)

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


# Add clusters into dataframe
df['Cluster'] = clusters

print("\n============ UNIQUE CLUSTERS ============\n")
print(df['Cluster'].unique())

"""
IMPORTANT:
-1 means NOISE / OUTLIER
"""

print("\n============ CLUSTER COUNTS ============\n")
print(df['Cluster'].value_counts())

def visualize_clusters(scaled_data):

    plt.figure(figsize=(10, 7))

    unique_clusters = sorted(set(clusters))

    for cluster in unique_clusters:

        # Select points of current cluster
        cluster_points = scaled_data[clusters == cluster]

        # Noise cluster
        if cluster == -1:
            label = "Noise / Outlier (-1)"
        else:
            label = f"Cluster {cluster}"

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=label
        )

    plt.title("Customer Location Clustering using DBSCAN")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")

    plt.legend()
    plt.show()

visualize_clusters(scaled_data)

# =========================================================
# CLUSTER ANALYSIS
# =========================================================

cluster_analysis = df.groupby('Cluster')[
    ['lat', 'lng']
].mean()

print("\n============ CLUSTER ANALYSIS ============\n")
print(cluster_analysis)


# =========================================================
# CLUSTER MEANINGS
# =========================================================

cluster_names = {
    -1: "Outlier / Isolated Booking Locations",
    0: "High Demand Area",
    1: "Regular Booking Area",
    2: "Dense Customer Hotspot",
    3: "Low Traffic Booking Zone"
}

print("\n============ CLUSTER INTERPRETATION ============\n")

for cluster_id in sorted(df['Cluster'].unique()):

    if cluster_id in cluster_names:
        print(
            f"Cluster {cluster_id} --> "
            f"{cluster_names[cluster_id]}"
        )
    else:
        print(
            f"Cluster {cluster_id} --> "
            f"Additional Customer Zone"
        )


# =========================================================
# OUTLIER ANALYSIS
# =========================================================

outliers = df[df['Cluster'] == -1]

print("\n============ OUTLIER CUSTOMERS ============\n")
print(outliers[['booking_id', 'lat', 'lng']].head())


# =========================================================
# BUSINESS UNDERSTANDING
# =========================================================

print("""
=========================================================
BUSINESS INSIGHTS
=========================================================

1. Dense clusters represent popular booking areas.

2. Businesses can:
   - place more drivers nearby
   - create service hubs
   - optimize delivery routes

3. Noise points (-1):
   - isolated customer bookings
   - rare locations
   - unusual booking areas

4. DBSCAN automatically detects:
   - natural groups
   - hotspot regions
   - outliers

5. Unlike KMeans:
   - no need to choose K manually
   - detects isolated points automatically
=========================================================
""")