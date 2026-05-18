"""
Problem — Iris Flower Species Clustering
Model — Gaussian Mixture Model
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

# ============ Load File ============
df = pd.read_csv("../../files/IRIS.csv")
# print("============ DF Head ============")
# print(df.head())

print("============ DF INFO ============\n", df.info())
# print("============ DF Description ============\n", df.describe())
# print("============ DF Null Check ============\n", df.isnull().sum())
print("============ DF Columns ============\n", list(df.columns))

# ============ Select Features ============
# x = df.drop("species", axis=1)
# ==== OR ====
x = df.iloc[:, 0:4]
# print(x.columns)

y = df["species"]


# ============ Pipeline ============
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("gmm", GaussianMixture(
        n_components=3,
        covariance_type='spherical',
        n_init=20,
        random_state=42
    ))
])

pipeline.fit(x)

# ========= Predict Clusters ==========
clusters = pipeline.predict(x)
df['Cluster'] = clusters

print(x.head())
scaled_x = pipeline.named_steps['scaler'].transform(x)

# ========= Evaluation =========
def evaluation(scaled_x, clusters):
    score = silhouette_score(scaled_x, clusters, metric='euclidean')
    print("====== silhouette score ===== ",score)

evaluation(scaled_x , clusters)

# ========= PCA for Visualization =========
pca = PCA(n_components=2)
x_pca = pca.fit_transform(scaled_x)
# print(x_pca)

# ========= Visualization =========
def visualization(x_pca, clusters):
    plt.figure(figsize=(8,8))

    cluster_names = {
        0: "Versicolor Group",
        1: "Setosa Group",
        2: "Virginica Group"
    }

    for cluster in range(3):
        plt.scatter(
            x_pca[clusters == cluster, 0],
            x_pca[clusters == cluster, 1],
            label=cluster_names[cluster]
        )

    plt.title("GMM Clustering on Iris Dataset")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

# visualization(x_pca, clusters)

# ========= Compare Actual Species =========
print("\n========== Cluster vs Species ==========")

comparison = pd.crosstab(
    df["Cluster"],
    df["species"]
)

print(comparison)
