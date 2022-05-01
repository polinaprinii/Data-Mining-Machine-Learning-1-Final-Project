"""
This file deals with step four of the KDD methodology.
The application of the Unsupervised Learning, K-Means Clustering is evaluated here.
"""

# Importing necessary libraries:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
sns.set_context('notebook')
plt.style.use('fivethirtyeight')
from warnings import filterwarnings
filterwarnings('ignore')

# Importing dataframe.
df = pd.read_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_PCA.csv")
print(df.head(2))

# Plot the raw data
plt.figure(figsize=(8, 8))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Visualization of raw data')
plt.show()

# Determining number of cluster to apply.
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12), size=(1080, 720)).fit(df)
visualizer.show()

# Applying k-means algorithm.
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)
print(kmeans.predict(df), '\n')
print(kmeans.inertia_, '\n')
print(kmeans.n_iter_, '\n')
print(kmeans.cluster_centers_, '\n')
print(kmeans.labels_,'\n')

# Plotting results
sns.scatterplot(data=df, x="principal component 1", y="principal component 2", hue=kmeans.labels_).set(
    title='K-means Cluster: k=4')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            marker="X", c="r", s=80, label="centroids")
plt.title('K-means Clustering: k=4')
plt.legend()
plt.show()

# Checking how many of the samples were correctly labeled
correct_labels = sum(kmeans.labels_)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, df.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(df.size)))