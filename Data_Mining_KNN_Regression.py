"""
This file continues step four of the KDD methodology - Data Mining.
Here we look to apply K-Nearest-Neighbour Regression analysis to our dataset.
"""

# Importing needed libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Importing data:
df = pd.read_csv('/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Selection_pre_PCA.csv')
print(df.head(5))

# Next we move to separating our features from our target.
X = df.drop('departure_delay', axis=1)
y = df['departure_delay']

# Splitting the data into a train and test group using a 20%/80% split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Standardising data:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Identifying the appropriate K value.
test_error = []
iterations = list(range(1, 31))
for k in range(1, 31):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)

    test_error.append(error)
    print(k, error)


plt.figure(figsize=(8, 4), dpi=150)
plt.plot(range(1, 31), test_error)
plt.show()


# plt.plot(index, error)
# plt.xlabel('Values of K')
# plt.ylabel('RSE Error')
# plt.title('Graph: K VS Error')
# plt.show()

# knn = KNeighborsRegressor(n_neighbors = 3)
# knn.fit(X, y)
