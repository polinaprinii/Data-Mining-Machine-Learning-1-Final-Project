"""
This file continues step four of the KDD methodology - Data Mining.
Here we look to apply K-Nearest-Neighbour Regression analysis to our dataset.
"""

# Importing needed libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

# Importing data.
df = pd.read_csv('/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv')
