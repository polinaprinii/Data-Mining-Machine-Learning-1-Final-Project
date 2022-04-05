"""
This file deals with the pre-processing aspect of the project in prep for machine learning techniques.
Here we will cover multiple check to ensure the data has no missing values and if so address this issue.
Followed by that we will ensure that outliers are reduced to a minimum.
Finally, we will apply Dimensionality Reduction to allow for better performance due to the large size of the dataset.
"""

# Importing all needed libraries:
import pandas as pd
import numpy as np
import missingno as msno
from matplotlib import pyplot as plt
import exrex

# Importing our file:

df = pd.read_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight_Weather.csv")
print(df.head(5), "\n", df.shape, "\n")

# Matrix to visualise all missing values.
msno.bar(df)
msno.matrix(df)
msno.heatmap(df)
plt.show()

"""
We can see a number of missing values are present within the datasets, though not many.
Manly present within the actual departure and actual arrival columns.
"""

# Pandas can detect "standard missing values" such as N/A or blank cells.
# As we are working with a considerably large amount of data, we will count the number of missing values.
print("Below are the number of missing values within each column present: ", "\n", "\n", df.isnull().sum(), "\n")

# Start filling and or replacing the null/missing values.
df['tail_number'] = df['tail_number'].fillna(exrex.getone(r'(\d "N" + {1,4}){4}'))







