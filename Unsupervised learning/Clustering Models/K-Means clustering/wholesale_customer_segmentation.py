"""
Problem : Wholesale Customer Segmentation

- A wholesale distributor wants to group customers based on annual spending patterns across product categories like: Milk, Grocery,
Frozen foods, Detergents, Fresh products

- Goal: identify customer groups ,understand buying behavior, improve business targeting
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ========= Load file =========
df = pd.read_csv('../../files/wholesale_customers_cata.csv')
# print("============ DF Head ============")
# print(df.head())

# print("============ DF INFO ============\n", df.info())
# print("============ DF Description ============\n", df.describe())
# print("============ DF Null Check ============\n", df.isnull().sum())
# print("============ DF Columns ============\n", list(df.columns))

# ========= Remove unnecessary columns =========
df = df.iloc[:,2:]
# print("============ DF Columns ============\n", list(df.columns))

# ========= Scale Data =========
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(df)

# ========= Apply PCA =========
# pca = PCA(n_components=2)
# pca_data = pca.fit_transform(scaled_data)


# ========= PIPELINE (ONLY SCALER + PCA) =========
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])

pca_data = pipeline.fit_transform(df)
print("\nOriginal Shape:", df.shape)
print("PCA Reduced Shape:", pca_data.shape)



print("\n============ EXPLAINED VARIANCE ============\n")
explained_variance = (
    pipeline.named_steps['pca'].explained_variance_ratio_
)
print(explained_variance)

print(
    "\nTotal Variance Retained:",
    np.sum(explained_variance)
)



# Elbow method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, init='k-means++')
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

print("====== Wcss ======", wcss)

def plot_elbow_chart(wcss):
    plt.figure(figsize=(7,7))

    plt.plot(range(1,11),wcss,marker='o' )
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()

plot_elbow_chart(wcss)


#====== Choosen best K from elbow graph ======
best_k = 5

# ====== Train model ======
kmeans = KMeans(
    n_clusters=best_k,
    init='k-means++',
    random_state=42,
    n_init=10
)
clusters = kmeans.fit_predict(pca_data)

feature_columns = df.columns

# ====== Add Clusters ======
df['Cluster'] = clusters

def visualize_clusters(pca_data, clusters):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=clusters,
        cmap='viridis'
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c='red',
        s=300,
        label='Centroids'
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.title("Wholesale Customer Segmentation")

    plt.legend()
    plt.show()

visualize_clusters(pca_data, clusters)

#============= Analyze Clusters =============
cluster_analysis = df.groupby('Cluster').mean()
print("========== cluster analysis ==========\n",cluster_analysis)

# =========================================================
# CLUSTER MEANINGS
# =========================================================

cluster_names = {
    0: "Retail / Grocery Shops",
    1: "Fresh Product Businesses",
    2: "Heavy Wholesale Buyers",
    3: "Small Regular Customers",
    4: "Premium Enterprise Customers"
}

# ====== silhouette score ======
score = silhouette_score(pca_data, clusters)

print("\nSilhouette Score:", score)

def predict_customer():

    fresh = float(input("Fresh spending: "))
    milk = float(input("Milk spending: "))
    grocery = float(input("Grocery spending: "))
    frozen = float(input("Frozen spending: "))
    detergents = float(input("Detergents spending: "))
    delicassen = float(input("Delicassen spending: "))

    new_data = pd.DataFrame([[
        fresh,
        milk,
        grocery,
        frozen,
        detergents,
        delicassen
    ]], columns=feature_columns)

    # Apply pipeline transformation
    transformed_data = pipeline.transform(new_data)

    # Predict cluster
    cluster = kmeans.predict(transformed_data)

    print("\nPredicted Cluster:", cluster[0])

    print(
        "Customer Category:",
        cluster_names[cluster[0]]
    )
predict_customer()