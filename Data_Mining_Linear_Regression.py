"""
This file deals with step four of the KDD methodology where various data mining machine learning algorithms are applied
to the selected, pre-processed and tranformed.
Linear Regression is applied and results evaluated.
"""

# Importing necessary libraries.
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd

# For purposes of exploration we import both datasets, pre PCA and after PCA application.
df_prePCA = pd.read_csv(
      "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv"
)
# df statistics:
# print(df_prePCA.info, "\n")
# print(df_prePCA.describe(), "\n")
# print(df_prePCA.head(), "\n")
# print(list(df_prePCA.columns.values), '\n')

df_PCA = pd.read_csv(
      "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_PCA.csv"
)
# df statistics:
# print(df_PCA.info, "\n")
# print(df_PCA.describe(), "\n")
# print(df_PCA.head(), "\n")
# print(list(df_PCA.columns.values), '\n')

