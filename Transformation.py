# Importing all necessary packages:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Cleansed_Flight_Weather.csv")

print(df.shape)

cor = df.corr()

plt.figure(figsize = (20, 10))
sns.heatmap(cor, annot = True)
plt.show()

array = df.values
X = array[:,0:34]
Y = array[:,34]

importance = mutual_info_classif(X,Y)
feat_importance = pd.Series(importance, df.columns[0: len(df.columns)-1])
feat_importance.plot(kind='barh', color='teal')