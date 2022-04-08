# Importing all necessary packages:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Cleansed_Flight_Weather.csv")

print(df.shape)

cor = df.corr()

plt.figure(figsize = (20, 10))
sns.heatmap(cor, annot = True)
plt.show()