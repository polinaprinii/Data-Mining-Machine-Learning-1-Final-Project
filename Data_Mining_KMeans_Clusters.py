"""
This file deals with step four of the KDD methodology.
The application of the Unsupervised Learning, K-Means Clustering is evaluated here.
"""

# Importing necessary libraries:

import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib. pyplot as plt

# Importing our two datasets into numpy arrays:
# np1 - Unstandardised dataset.
df1 = pd.read_csv(
    "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv")

np1 = df1.to_numpy()
# print(np1, '\n')

# Standardised and reduced dataset.
df2 = pd.read_csv(
    "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_PCA.csv")

np2 = df2.to_numpy()
#print(np2, '\n')

# Standardising data from np1 using StandardScaler()
features_np1 = np1

scaler = StandardScaler()
scaled_features_np1 = scaler.fit_transform(features_np1)
# print(scaled_features_np1[:5])

# Determining the best number of clusters to apply to our K-Means Clustering algorithm.
kmeans_kwargs_np1 = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse_np1 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs_np1)
    kmeans.fit(scaled_features_np1)
    sse_np1.append(kmeans.inertia_)

# Plotting results.
plt.plot(range(1, 11), sse_np1, '--bo', label='line with marker')
plt.title('Determining No. of Clusters for non Dimensionality Reduced Data')
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Total within clusters Sum of Squares")
plt.show()

kl_np1 = KneeLocator(
    range(1, 11), sse_np1, curve="convex", direction="decreasing")

print('The optimal number of clusters prior to PCA application is: ', kl_np1.elbow, '\n')

# Setting the k means clustering parameters as per the results of the elbow method for the first test - np1
kmeans_np1 = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)

# Fitting the k means to our standardised non dimensionality reduced dataset.
kmeans_np1.fit(scaled_features_np1)

# Deriving statistics:
# The lowest SSE value
print(round(kmeans_np1.inertia_, 2), '\n')

# Final locations of the centroid
print(kmeans_np1.cluster_centers_, '\n')

# The number of iterations required to converge
print(kmeans_np1.n_iter_, '\n')

# Plotting the results of K-Means Clustering for np1.
# predict the labels of clusters.
label_np1 = kmeans_np1.fit_predict(scaled_features_np1)

# Getting unique labels
u_labels_np1 = np.unique(label_np1)

# plotting the results:
for i in u_labels_np1:
    plt.scatter(scaled_features_np1[label_np1 == i, 0], scaled_features_np1[label_np1 == i, 11], label=i)
plt.legend()
plt.title('Flight Delay Data, K= 4')
plt.show()

"""
As np2 has already been standardised during the Transformation stage, we go straight to determining the number of clusters
to apply to the K-Means clusters.
"""
# Determining the best number of clusters to apply to our K-Means Clustering algorithm.
kmeans_kwargs_np2 = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse_np2 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs_np2)
    kmeans.fit(np2)
    sse_np2.append(kmeans.inertia_)

# Plotting results.
plt.plot(range(1, 11), sse_np2, '--bo', label='line with marker')
plt.xticks(range(1, 11))
plt.title('Determining No. of Clusters after PCA Application')
plt.xlabel("Number of Clusters")
plt.ylabel("Total within clusters Sum of Squares")
plt.show()

kl_np2 = KneeLocator(
    range(1, 11), sse_np2, curve="convex", direction="decreasing")

print('The optimal number of cluster after PCA application is: ', kl_np2.elbow, '\n')

# Setting the k means clustering parameters as per the results of the elbow method for the first test - np1
kmeans_np2 = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)

# Fitting the k means to our standardised non dimensionality reduced dataset.
kmeans_np2.fit(np2)

# Deriving statistics:
# The lowest SSE value
print(round(kmeans_np2.inertia_, 2), '\n')

# Final locations of the centroid
print(kmeans_np2.cluster_centers_, '\n')

# The number of iterations required to converge
print(kmeans_np2.n_iter_, '\n')

# Plotting the results of K-Means Clustering for np2.
# predict the labels of clusters.
label_np2 = kmeans_np2.fit_predict(scaled_features_np1)

# Getting unique labels and centroids
centroids = kmeans.cluster_centers_
u_labels_np2 = np.unique(label_np2)

for i in u_labels_np2:
    plt.scatter(np2[label_np2 == i , 0] , np2[label_np2 == i , 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend(['Cluster1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], loc = 'best')
plt.title('Flight Delay Data, K= 4')
plt.show()


# check how many of the samples were correctly labeled

correct_labels = sum(np2 == label_np2)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, np2.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(np2.size)))

