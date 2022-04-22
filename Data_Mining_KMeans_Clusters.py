# Importing necessary libraries:
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Importing our two datasets into numpy arrays:
# np1 - Unstandardised dataset.
np1 = np.genfromtxt("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv")

# Standardised and reduced dataset.
np2 = np.genfromtxt("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_PCA.csv")



