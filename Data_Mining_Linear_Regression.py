"""
This file deals with step four of the KDD methodology where various data mining machine learning algorithms are applied
to the selected, pre-processed and tranformed.
Linear Regression is applied and results evaluated.
"""

# Importing necessary libraries.
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd