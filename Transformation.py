# Importing all necessary packages:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA, KernelPCA, NMF
import numpy as np
from sklearn.preprocessing import StandardScaler

# Importing our csv file.
df = pd.read_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Cleansed_Flight_Weather.csv")

# Ensuring the dataframe loaded correctly.
print(df.dtypes)

"""
Running the data through a number of feature selection techniques to understand the data prior to undertaking the 
dimensionality reduction through Principal Component Analysis.
Once applied we will look to undertake K-Means clustering on the data pre PCA and after PCA.
"""

# Mapping our dataframe to a correlation matrix.
cor = df.corr(method='pearson')

# Plotting the matrix.
plt.figure(figsize = (20, 10))
sns.heatmap(cor, annot = True)
plt.show()

"""
As we prepare the data analysis, we must note that most features within the dataset / dataframe are of string type which
are being interpreted as object type.
Thus we will be applying both feature selection based on the findings from the correlation matrix and information gain results,
and dimensionality reduction as the features we are considerable 
"""

# Filtering out string/object type columns as Information Gain can't comprehend columns of such type.
columns = ['scheduled_elapsed_time', 'departure_delay', 'arrival_delay', 'delay_carrier', 'delay_weather',
           'delay_national_aviation_system', 'delay_security', 'delay_late_aircarft_arrival', 'HourlyDryBulbTemperature_x',
           'HourlyPrecipitation_x', 'HourlyStationPressure_x', 'HourlyVisibility_x', 'HourlyWindSpeed_x',
           'HourlyDryBulbTemperature_y', 'HourlyPrecipitation_y', 'HourlyStationPressure_y', 'HourlyVisibility_y',
           'HourlyWindSpeed_y']

# Setting an array for the Information Gain to interpret.
array = df[columns].values
X = array[:, 0:17]
Y = array[:, 17]

# Running the information gain.
importance = mutual_info_classif(X,Y)
feat_importance = pd.Series(importance, df[columns].columns[0: len(df[columns].columns)-1])
feat_importance.plot(kind='bar', color='teal', figsize = (18, 10))
plt.show()

"""
Following the results of both the Correlation Matrix and the Mutual Information Gain, the following features are moved 
forward based on the outlined thresholds:

Correlation Matrix, any feature above 0.25 are considered as having a strong measure and direction of the linear association
between two variables:
- departure_delay
- arrival_delay
- delay_carrier
- delay_weather
- delay_national_aviation_system
- delay_late_aircraft_arrival
- HourlyDryBulbTemperature_x
- HourlyPrecipitation_x
- HourlyVisibility_x
- HourlyDryBulbTemperature_y
- HourlyPrecipitation_y
- HourlyVisibility_y

The above selection is supported by the Mutual Information Gain.
"""
selection = ['departure_delay', 'arrival_delay', 'delay_carrier', 'delay_weather','delay_national_aviation_system',
             'delay_late_aircarft_arrival', 'HourlyDryBulbTemperature_x', 'HourlyPrecipitation_x', 'HourlyVisibility_x',
             'HourlyDryBulbTemperature_y', 'HourlyPrecipitation_y', 'HourlyVisibility_y']

# Setting selection to a new dataframe which will be exported before and after PCA dimensionality reduction.
new_df = df[selection]

# Extracting selection prior to PCA application.