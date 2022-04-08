# Importing all necessary packages:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# Importing our csv file.
df = pd.read_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Cleansed_Flight_Weather.csv")

table = df.to_numpy()

print(df.dtypes)


# Mapping our dataframe to a correlation matrix.
cor = df.corr()

# Plotting the matrix.
plt.figure(figsize = (20, 10))
sns.heatmap(cor, annot = True)
plt.show()

columns = ['scheduled_elapsed_time', 'departure_delay', 'arrival_delay', 'delay_carrier', 'delay_weather',
           'delay_national_aviation_system', 'delay_security', 'delay_late_aircarft_arrival', 'HourlyDryBulbTemperature_x',
           'HourlyPrecipitation_x', 'HourlyStationPressure_x', 'HourlyVisibility_x', 'HourlyWindSpeed_x',
           'HourlyDryBulbTemperature_y', 'HourlyPrecipitation_y', 'HourlyStationPressure_y', 'HourlyVisibility_y',
           'HourlyWindSpeed_y']

array = df[columns].values
X = array[:,0:17]
Y = array[:,17]

importance = mutual_info_classif(X,Y)
feat_importance = pd.Series(importance, df[columns].columns[0: len(df[columns].columns)-1])
feat_importance.plot(kind='barh', color='teal', figsize = (18, 10))
plt.show()