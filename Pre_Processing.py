"""
This file deals with step two of the KDD methodology, the pre-processing aspect of the project in prep for
machine learning techniques.
Here we will cover multiple check to ensure the data has no missing values and if so address this issue.
Followed by that we will ensure that outliers are reduced to a minimum.
Finally, we will apply Dimensionality Reduction to allow for better performance due to the large size of the dataset.
"""

# Importing all needed libraries:
import os
import pandas as pd
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
tail_number = "N" + n1 + chr(n2) + chr(n3)


# # Applying the change to the tail number column.
df['tail_number'] = df['tail_number'].fillna(tail_number)


# Next we move to addressing the missing values within the actual departure and arrival columns.
# First we build a function which will create a random date-time value.
def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


# Set start and end date, for simplicity purposes we will generate dates for the month of December.
d1 = datetime.strptime('1/12/2019 00:00', '%d/%m/%Y %H:%M')
d2 = datetime.strptime('31/12/2019 23:55', '%d/%m/%Y %H:%M')

# Fill all missing values within the actual departure date time column.
df['actual_departure_dt'] = df['actual_departure_dt'].fillna(random_date(d1, d2).strftime("%Y-%m-%d %H:%M"))

# Now we move to addressing missing values within the actual arrive date time column.
"""
We achieve this by using the previously filled missing values from actual departure date time column and add time.
For simplicity reasons, we add an extra hour to the already given date-time. 
This may cause bias however, to ensure there are no missing values present in the data we will proceed.
"""
df['actual_arrival_dt'] = df['actual_arrival_dt'].fillna(pd.to_datetime(df['actual_departure_dt']) +
                                                         pd.to_timedelta(1, unit='H'))

# Next we move to addressing missing values for the Station x and Station y columns.
df['STATION_x'] = df['STATION_x'].fillna(method='ffill')
df['STATION_y'] = df['STATION_y'].fillna(method='ffill')

"""
Lastly we address the columns outlining weather information. 
It is assumed that the missing values are not missing values and simply represent a lack of precipitation and so one
for a given time stamps.
To ensure an absence of missing values throughout the data we will modify all 0 values by 0.01 to avoid high basis.
"""
columns = ['HourlyDryBulbTemperature_x', 'HourlyPrecipitation_x', 'HourlyStationPressure_x', 'HourlyVisibility_x',
           'HourlyWindSpeed_x', 'HourlyDryBulbTemperature_y', 'HourlyPrecipitation_y', 'HourlyStationPressure_y',
           'HourlyVisibility_y', 'HourlyWindSpeed_y']

df[columns] = df[columns].fillna('0.01')

print("Below all column should indicate a zero count after being addressed: ", "\n", "\n", df.isnull().sum(), "\n")

# Moving on to identify any outliers.
"""
We will be using the standard box plot to identify outliers.
"""
# Checking data type for all columns:
print(df.dtypes)

# Converting object type columns to numeric type columns.
df[['HourlyDryBulbTemperature_x', 'HourlyPrecipitation_x', 'HourlyStationPressure_x', 'HourlyVisibility_x',
    'HourlyWindSpeed_x', 'HourlyDryBulbTemperature_y', 'HourlyPrecipitation_y', 'HourlyStationPressure_y',
    'HourlyVisibility_y', 'HourlyWindSpeed_y']] = df[['HourlyDryBulbTemperature_x', 'HourlyPrecipitation_x',
    'HourlyStationPressure_x', 'HourlyVisibility_x', 'HourlyWindSpeed_x', 'HourlyDryBulbTemperature_y',
    'HourlyPrecipitation_y', 'HourlyStationPressure_y', 'HourlyVisibility_y', 'HourlyWindSpeed_y']].apply(pd.to_numeric)

# Ensuring the conversion was successful.
print(df.dtypes)

# Filtering out the string type columns.
Outlier_columns = ['departure_delay', 'arrival_delay', 'delay_carrier', 'delay_weather', 'delay_national_aviation_system',
        'delay_security', 'delay_late_aircarft_arrival', 'HourlyDryBulbTemperature_x', 'HourlyPrecipitation_x',
        'HourlyStationPressure_x', 'HourlyVisibility_x', 'HourlyWindSpeed_x', 'HourlyDryBulbTemperature_y',
        'HourlyPrecipitation_y', 'HourlyStationPressure_y', 'HourlyVisibility_y', 'HourlyWindSpeed_y']

# Plotting outliers
df[Outlier_columns].plot(kind="box", subplots=True, layout=(9, 2), figsize=(30, 35));
plt.show()

# Converting float type columns to integers for simplicity.
df = df.astype({"STATION_x":'int', "HourlyDryBulbTemperature_x":'int',"HourlyPrecipitation_x":'int',
                "HourlyStationPressure_x":'int', "HourlyVisibility_x":'int', "HourlyWindSpeed_x":'int',
                "STATION_y":'int', "HourlyDryBulbTemperature_y":'int', "HourlyPrecipitation_y":'int',
                "HourlyStationPressure_y":'int', "HourlyVisibility_y":'int', "HourlyWindSpeed_y":'int'})

print(df.dtypes)

# Exporting the cleansed file to local hard-drive:
def export():
# Restrict file from duplicating.
    if os.path.exists(
        "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Cleansed_Flight_Weather.csv"):
        pass

    else:
# Export to csv
        df.to_csv(
            "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Cleansed_Flight_Weather.csv",
            index=False, encoding='utf-8-sig')

export()