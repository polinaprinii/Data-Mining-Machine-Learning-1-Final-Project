"""
This file concludes step four of the KDD methodology.
Here we implement an ensemble model based on the Boosting method.
Boosting is a sequential method–it aims to prevent a wrong base model from affecting the final output.
"""

# Importing needed libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler

# Loading data:
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

#
# define the model
model = AdaBoostRegressor()
# fit the model on the whole dataset
model.fit(X_train, y_train)
# make predictions
y_pred = model.predict(X_test)
print('Prediction:', y_pred, '\n')

print('AdaBoosting Score: ', model.score(X_test, y_test), '\n')

# initializing the boosting module with default parameters
model_2 = GradientBoostingRegressor()

# training the model on the train dataset
model_2.fit(X_train, y_train)

# predicting the output on the test dataset
pred_final = model_2.predict(X_test)
print('GradientBoosting Score: ', model_2.score(X_test, y_test), '\n')




