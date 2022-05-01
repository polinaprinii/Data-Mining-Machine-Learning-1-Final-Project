"""
This file deals with step four of the KDD methodology where various data mining machine learning algorithms are applied
to the selected, pre-processed and transformed.
Linear Regression is applied and results evaluated.
"""

# Importing necessary libraries.
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# For purposes of exploration we import both datasets, pre PCA and after PCA application.
# Dataframe 1: Pre-PCA
df_prePCA = pd.read_csv(
      "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv"
)
# Visualising the Dataframe 1.
df_prePCA.plot(figsize=(25,5))
plt.title('Visualising df1 prior to PCA application')
plt.legend(bbox_to_anchor=(1,1), loc="upper left")
plt.show()

# Dataframe 1 Statistics:
print(df_prePCA.info, "\n")
print(df_prePCA.describe(), "\n")
print(df_prePCA.head(), "\n")
print(list(df_prePCA.columns.values), '\n')

# Identifying linear relationships between our target variable and feature variables within Dataframe 1.
plt.scatter(df_prePCA['departure_delay'], df_prePCA['arrival_delay'])
plt.title('Linear Relationship: y - Departure Delay & x - Arrival Delay')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['delay_carrier'])
plt.title('Linear Relationship: y - Departure Delay & x - Delay Carrier')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['delay_national_aviation_system'])
plt.title('Linear Relationship: y - Departure Delay & x - Delay National Aviation System')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['delay_late_aircarft_arrival'])
plt.title('Linear Relationship: y - Departure Delay & x - Delay Late Aircraft Arrival')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['actual_arrival_dt'])
plt.title('Linear Relationship: y - Departure Delay & x - Actual Arrival Date')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['HourlyDryBulbTemperature_x'])
plt.title('Linear Relationship: y - Departure Delay & x - HourlyDryBulbTemperature_x')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['HourlyPrecipitation_x'])
plt.title('Linear Relationship: y - Departure Delay & x - HourlyPrecipitation_x')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['HourlyVisibility_x'])
plt.title('Linear Relationship: y - Departure Delay & x - HourlyVisibility_x')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['HourlyDryBulbTemperature_y'])
plt.title('Linear Relationship: y - Departure Delay & x - HourlyDryBulbTemperature_y')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['HourlyPrecipitation_y'])
plt.title('Linear Relationship: y - Departure Delay & x - HourlyPrecipitation_y')
plt.show()
plt.scatter(df_prePCA['departure_delay'], df_prePCA['HourlyVisibility_y'])
plt.title('Linear Relationship: y - Departure Delay & x - HourlyVisibility_y')
plt.show()

# Next we move to separating our features from our target for Dataframe 1.
X = df_prePCA[['arrival_delay','delay_carrier', 'delay_weather', 'delay_national_aviation_system',
               'delay_late_aircarft_arrival', 'actual_arrival_dt', 'HourlyDryBulbTemperature_x', 'HourlyPrecipitation_x',
               'HourlyVisibility_x', 'HourlyDryBulbTemperature_y', 'HourlyPrecipitation_y', 'HourlyVisibility_y']]

Y = df_prePCA['departure_delay']

# Secondly we split the data into 30% test and 70% train for Dataframe 1:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Apply Multiple Linear Regression Analysis for Dataframe 1:
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

# Predicting our dependent variable for Dataframe 1.
Y_pred = regr.predict(X_test)

# Applying Ordinary Least Squared Regression to Dataframe 1.
model = sm.OLS(Y_test, X_test).fit()
print_model = model.summary()

# Interpreting the results for Dataframe 1:
print('Summary of Results - Dataframe 1 - Pre PCA: \n',print_model, '\n')
print('The R-squared value for Dataframe 1 is : ', round(regr.score(X_test, Y_test), 2), '\n')
print('The Mean squared error for Dataframe 1 is : %.3f'
      % mean_squared_error(Y_test, Y_pred), '\n')
print('The Root-mean-square deviation for Dataframe 1 is : %.3f'
      % mean_squared_error(Y_test, Y_pred, squared=False), '\n')
print('The Mean-absolute-percentage-error for Dataframe 1 : %.3f'
      % mean_absolute_percentage_error(Y_test, Y_pred), '\n')

# As there are not much good statistics from Dataframe 2 we comment out the code but keep it for educational purposes.
# Dataframe 2: PCA
# df_PCA = pd.read_csv(
#       "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_PCA.csv"
# )
# # Visualising Dataframe 2.
# df_PCA.plot(figsize=(25,5))
# plt.title('Visualising df2 after PCA application')
# plt.legend(bbox_to_anchor=(1,1), loc="upper left")
# plt.show()
#
# # Dataframe 2 Statistics:
# # print(df_PCA.info, "\n")
# # print(df_PCA.describe(), "\n")
# # print(df_PCA.head(), "\n")
# # print(list(df_PCA.columns.values), '\n')
#
# # Identifying linear relationships between our target variable and feature variables within Dataframe 2.
# plt.scatter(df_PCA['principal component 1'], df_PCA['principal component 2'])
# plt.show()

# Assign features and targets:
# x = np.array(df_PCA['principal component 1']).reshape(-1, 1)
# y = np.array(df_PCA['principal component 2']).reshape(-1, 1)

# Dropping any NA values.
# df_PCA.dropna(inplace = True)

# Secondly we split the data into 30% test and 70% train for Dataframe 2:
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Apply Multiple Linear Regression Analysis for Dataframe 2:
# regression = linear_model.LinearRegression()
# regression.fit(x_train, y_train)

# Predicting our dependent variable for Dataframe 2.
# y_pred = regression.predict(x_test)

# Applying Ordinary Least Squared Regression to Dataframe 2.
# model = sm.OLS(y_test, x_test).fit()
# print_model = model.summary()

# Interpreting the results for Dataframe 1:
# print('Summary of Results - Dataframe 2 - PCA = 2: \n',print_model, '\n')
# print('The R-squared for Dataframe 2 is : ', round(regression.score(x_test, y_test), 2), '\n')
# print("The Mean squared error for Dataframe 2 is : %.3f"
#       % mean_squared_error(y_test, y_pred), '\n')
# print('The Root-mean-square deviation for Dataframe 2 is : %.3f'
#       % mean_squared_error(y_test, y_pred, squared=False), '\n')
# print('The Mean-absolute-percentage-error for Dataframe 2 is : %.3f'
#       % mean_absolute_percentage_error(y_test, y_pred), '\n')
