# Importing all necessary packages:
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importing our csv file.
df = pd.read_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Cleansed_Flight_Weather.csv")

# Ensuring the dataframe loaded correctly.
# print(df.dtypes)

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

# Filtering out the target column - Origin Airport
target = ['origin_airport']

# Assigning the selection to a new dataframe which will be standardised, and dimensionality reduction applied.
selection_df = df[selection]

# Now we proceed to standardising the feature selection.
scaled_features = df[selection]
# print(scaled_features.head(5))
scaled_features = StandardScaler().fit_transform(scaled_features)

""" 
The Standard Scaler transforms the df into a numpy array after standardised. 
Due to this we re-format the numpy array back into a dataframe as PCA cannot be applied to an array. 
"""
scaled_features_df = pd.DataFrame(scaled_features, index=df[selection].index, columns=df[selection].columns)

# Sanity check to see the re-format was successful.
# print(scaled_features_df.head(5), "\n")
# print(scaled_features_df.shape)

# Now we move to reducing the dimension of our standardised dataframe from 12 columns to 6 columns.
pca = PCA(n_components=6) # setting the limit of reduction.
principalComponents = pca.fit_transform(scaled_features_df)
# print(principalComponents)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2',
                          'principal component 3', 'principal component 4',
                          'principal component 5', 'principal component 6'])
# print(principalDf.head(5))
# print(principalDf.shape)

# Lastly we concatenate the reduced dataset with a single column from the original source which specifies the origin airport.
finalDf = pd.concat([principalDf, df[target]], axis = 1)
# print(finalDf.head(5))

# Visualise results.
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['JFK', 'LAX', 'MIA']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['origin_airport'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# Next we move to extract 2 files.
# First we filter our second selection.
final_selection = ['origin_airport','departure_delay', 'arrival_delay', 'delay_carrier', 'delay_weather','delay_national_aviation_system',
                    'delay_late_aircarft_arrival', 'HourlyDryBulbTemperature_x', 'HourlyPrecipitation_x', 'HourlyVisibility_x',
                    'HourlyDryBulbTemperature_y', 'HourlyPrecipitation_y', 'HourlyVisibility_y']

extract_df = df[final_selection]

# Extracting selection prior to PCA application as we will look to apply K-Means clustering to both pre PCA and after PCA.
def export():
# Restrict file from duplicating.
    if os.path.exists(
        "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv"):
        pass

    if os.path.exists(
        "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_PCA.csv"):
        pass

    else:
# Export to csv
        extract_df.to_csv(
            "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv",
            index=False, encoding='utf-8-sig')

        finalDf.to_csv(
            "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_PCA.csv",
            index=False, encoding='utf-8-sig')

export() # Due to laptop constrictions we will look to import the exported file following the pre-processing step.

# Lastly we check how much information (variance) can be attributed to each of the principal components.
print(sum(pca.explained_variance_ratio_))