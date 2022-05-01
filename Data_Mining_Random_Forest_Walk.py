"""
This file continues step four of the KDD methodology - Data Mining.
Here we look to apply Random Forest walk analysis to our dataset.
"""

# Importing needed libraries:
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Importing dataset.
df = pd.read_csv('/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv')
print(df.head(5))

# Next we move to separating our features from our target.
X = df.drop('departure_delay', axis=1)
y = df['departure_delay']

# Splitting the data into a train and test group using a 30%/70% split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Standardising data:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Applying Random Forest Regression to our data.
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)

# Running predictions:
y_pred = regressor.predict(X_test)

# Plotting predictions:
g = plt.scatter(y_test, y_pred)
g.axes.set_yscale('log')
g.axes.set_xscale('log')
g.axes.set_xlabel('True Values ')
g.axes.set_ylabel('Predictions ')
g.axes.axis('equal')
g.axes.axis('square')
plt.show()

g = plt.plot(y_test - y_pred,marker='o',linestyle='')
plt.show()

# Assigning the OLS model.
model = sm.OLS(y_test, X_test).fit()
print_model = model.summary()

rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# Display the performance metrics
print('Summary of Results Random Forest Regression : \n', print_model, '\n')
print('The R-squared is : ', round(regressor.score(X_test, y_test), 3), '\n')
print('The Mean squared error is : %.3f'
      % mean_squared_error(y_test, y_pred), '\n')
print('The Root-mean-square deviation for is : %.3f'
      % rmse, '\n')
print('The Mean-absolute-percentage-error is : %.3f'
      % mean_absolute_percentage_error(y_test, y_pred), '\n')


