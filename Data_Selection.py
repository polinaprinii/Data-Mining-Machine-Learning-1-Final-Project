# Importing supporting libraries.
import pandas as pd
import glob
import os

"""
Introducing function to prevent the file merge from re-occurring each time the code is executed.
"""
def mergefile():
# Merging all flight data from May 2019 to December 2019 from the "Flight Delays for 2019 for the USA" into one CSV file:
# Restrict file from duplicating.
    if os.path.exists("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight_Weather.csv"):
        pass
# To understand how the else works, please delete the merged file from your repository.
    else:
# Step 1: Set working directory.
        os.chdir("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA")

# Step 2: Match .csv file extension.
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# Step 3: Use pandas to concatenate in the list and export master CSV file.
# combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
# export to csv
        combined_csv.to_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight_Weather.csv",
                    index=False, encoding='utf-8-sig')

mergefile()

# Setting the merged file to a pandas dataframe:
df = pd.read_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight_Weather.csv")

print(df.shape)



