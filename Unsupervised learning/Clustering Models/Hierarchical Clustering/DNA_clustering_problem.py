"""
Problem: Scientists want to group DNA samples based on genetic similarities without using labels directly.
Model : Hierarchical Clustering
Goal:
- Discover natural DNA groups
- Identify similar species/genetic patterns
- Analyze biological similarity
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage


# ============ Load File ============
df = pd.read_csv("../../files/synthetic_dna_dataset.csv")
# print("============ DF Head ============")
# print(df.head())

# print("============ DF INFO ============\n", df.info())
# print("============ DF Description ============\n", df.describe())
# print("============ DF Null Check ============\n", df.isnull().sum())
print("============ DF Columns ============\n", list(df.columns))

# ============ Select Feature : Use only numerical columns ============
# features = ['GC_Content', 'AT_Content', 'Sequence_Length', 'Num_A', 'Num_T', 'Num_C', 'Num_G', 'kmer_3_freq']
# X = df[features]

# =========== OR =============

x = df.iloc[:,2:10]
features_column = x.columns
print("========== Selected Feature columns ===========\n",features_column)


# ============ Pipeline ============
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
])

transformed_data = pipeline.fit_transform(x)


# =========== Create Dendrogram ===========
def create_dendrogram(transformed_data):
    linked = linkage(transformed_data, method='ward')
    plt.figure(figsize=(10,10))
    dendrogram(linked)

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

create_dendrogram(transformed_data)

# ============== Apply Hierarchical Clustering ==============
hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')

clusters = hc.fit_predict(transformed_data)
df['Cluster'] = clusters

# ================== silhouette score ===================
score = silhouette_score(transformed_data, clusters)
print("Silhouette Score :", score)


# ============== Visualize Clusters ==============

def visualize_clusters(clusters):
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(
        transformed_data[:,0],
        transformed_data[:,1],
        c=clusters,
    )

    plt.title('Hierarchical Clustering Visualization')
    plt.xlabel('PCA COMPONENT 1')
    plt.ylabel('PCA COMPONENT 2')
    plt.show()

visualize_clusters(clusters)

print("============ Compare with Actual Labels ===========\n", list(df.columns))
print(df[['Class_Label', 'Cluster']].head(20))


