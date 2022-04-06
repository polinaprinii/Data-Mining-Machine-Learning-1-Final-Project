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
import random
from random import randrange
from datetime import timedelta
from datetime import datetime

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
"""
 We first look at the tail number column.
 Using the random functionality in Python we structure a tail number.
"""
n1 = "%03d" % random.randint(0, 999)
n2 = random.randint(65, 90)
n3 = random.randint(65, 90)
tailnumber = "N" + n1 + chr(n2) + chr(n3)
print(tailnumber)

# # Applying the change to the tail number column.
df['tail_number'] = df['tail_number'].fillna(tailnumber)


# Next we move to addressing the missing values within the actual departure and arrival columns.
# First we build a function which will create a random date-time value.
def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60)
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)

# Set start and end date, for simplicity purposes we will generate dates for the month of December.
d1 = datetime.strptime('1/12/2019 00:00', '%d/%m/%Y %H:%M')
d2 = datetime.strptime('31/12/2019 23:55', '%d/%m/%Y %H:%M')

# Fill all missing values within the actual departure date time column.
df['actual_departure_dt'] = df['actual_departure_dt'].fillna(random_date(d1, d2).strftime("%Y-%m-%d %H:%M"))

print("Below are the number of missing values within each column present: ", "\n", "\n", df.isnull().sum(), "\n")
