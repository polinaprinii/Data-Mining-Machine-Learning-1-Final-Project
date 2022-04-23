# Importing necessary libraries:
import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as df

# Importing our two datasets into numpy arrays:
# np1 - Unstandardised dataset.
df1 = pd.read_csv(
    "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv")

np1 = df1.to_numpy()
#print(np1, '\n')

# Standardised and reduced dataset.
df2 = pd.read_csv(
    "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_PCA.csv")

np2 = df2.to_numpy()
#print(np2, '\n')

# Standardising data using StandardScaler()
features_np1 = np1

scaler = StandardScaler()
scaled_features_np1 = scaler.fit_transform(features_np1)

# print(scaled_features_np1[:5])

# Setting the k means clustering paramentres
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

# Fitting the k means to our standardised non dimensionality reduced dataset.
kmeans.fit(scaled_features_np1)

# Deriving statistics:
# The lowest SSE value
print(round(kmeans.inertia_, 2), '\n')

# Final locations of the centroid
print(kmeans.cluster_centers_, '\n')

# The number of iterations required to converge
print(kmeans.n_iter_, '\n')

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features_np1)
    sse.append(kmeans.inertia_)

# Plotting results.
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)

print(kl.elbow)