"""
This file continues step four of the KDD methodology - Data Mining.
Here we look to apply K-Nearest-Neighbour Regression analysis to our dataset.
"""

# Importing needed libraries:
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import metrics
import matplotlib.pyplot as plt


# Importing data:
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

# Determining the optimal K-Value.
n = len(y)
index = []
error = []
for i in range(1, 20):
    knn = KNeighborsRegressor(n_neighbors = i)
    knn.fit(X_test, y_test)
    y_pred = knn.predict(X_test)
    rss = sum((y_pred - y_test)**2)
    rse = math.sqrt(rss/(n-2))
    index.append(i)
    error.append(rse)

# Plotting the optimum K-value.
plt.plot(index, error)
plt.xlabel('Values of K')
plt.ylabel('RSE Error')
plt.title('Graph: K VS Error')
plt.show()

# Applying K-Nearest Neighbour Regression Analysis:
regr = KNeighborsRegressor(2)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

model = sm.OLS(y_test, X_test).fit()
print_model = model.summary()

# Interpreting the results:
print('Summary of Results KNN Regression : \n', print_model, '\n')
print('The R-squared is : ', round(regr.score(X_test, y_test), 3), '\n')
print('The Mean squared error is : %.3f'
      % mean_squared_error(y_test, y_pred), '\n')
print('The Root-mean-square deviation for is : %.3f'
      % rmse, '\n')
print('The Mean-absolute-percentage-error is : %.3f'
      % mean_absolute_percentage_error(y_test, y_pred), '\n')

# Measuring accuracy on Testing Data
print('Accuracy',100 - (np.mean(np.abs((y_test - y_pred) / y_test)) * 100))

# Plotting the predicted results:
x_ax = range(9000)
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred, lw=1.5, color="red", label="predicted")
plt.legend()
plt.show()

# TODO: Determine & resolve 'ValueError: operands could not be broadcast together with shapes' when identifying optimum K value.



