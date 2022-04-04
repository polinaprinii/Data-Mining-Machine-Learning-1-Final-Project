# Importing supporting libraries.
import pandas as pd
import glob
import os

"""
Due to the large volume of each dataset (approximate row count of 500,000).
We will look to import the 3 datasets based on seasonality (Summer, Autumn and Winter) seperately.
From there pandas will select 10,000 random rows from each dataset (as per project requirements).
Following the random selecting we will look to merge all 3 new dataframes into one master dataframe.
"""

# Import Dataframes:
Summer_df = pd.read_csv(
    "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight-06-2019.csv")
print(Summer_df.shape, "\n")

Autumn_df = pd.read_csv(
    "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight-09-2019.csv")
print(Autumn_df.shape, "\n")

Winter_df = pd.read_csv(
    "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight-12-2019.csv")
print(Winter_df.shape, "\n")

# Random selection:
random_summer_df = Summer_df.sample(n = 10000)
print(random_summer_df.shape, "\n")
print(random_summer_df.head(5), "\n")

random_autumn_df = Autumn_df.sample(n = 10000)
print(random_autumn_df.shape, "\n")
print(random_autumn_df.head(5), "\n")

random_winter_df = Winter_df.sample(n = 10000)
print(random_winter_df.shape, "\n")
print(random_winter_df.head(5), "\n")

# Merge all 3 newly created datasets with random selection
Master_df = pd.concat([random_summer_df, random_autumn_df, random_winter_df], ignore_index=True, sort=False)
print(Master_df.shape)

# Exporting file to local hard-drive:
def export():
# Restrict file from duplicating.
    if os.path.exists(
        "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight_Weather.csv"):
        pass

    else:
# Export to csv
        Master_df.to_csv(
            "/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight_Weather.csv",
            index=False, encoding='utf-8-sig')

export() # Due to laptop consitriction we will look to import the exported file following the pre-processing step.

"""
Retaining below code for leanring and exposure purposes.
"""
"""
Introducing function to prevent the file merge from re-occurring each time the code is executed.
"""
# def mergefile():
# # Merging all flight data from May 2019 to December 2019 from the "Flight Delays for 2019 for the USA" into one CSV file:
# # Restrict file from duplicating.
#     if os.path.exists("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight_Weather.csv"):
#         pass
# # To understand how the else works, please delete the merged file from your repository.
#     else:
# # Step 1: Set working directory.
#         os.chdir("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA")
#
# # Step 2: Match .csv file extension.
#         extension = 'csv'
#         all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#
# # Step 3: Use pandas to concatenate in the list and export master CSV file.
# # combine all files in the list
#         combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
# # export to csv
#         combined_csv.to_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight_Weather.csv",
#                     index=False, encoding='utf-8-sig')
#
# mergefile()
#
# # Setting the merged file to a pandas dataframe:
# df = pd.read_csv("/Users/polinaprinii/Desktop/Project Datasets/Flight Delays for 2019 for the USA/Flight_Weather.csv")
#
# print(df.shape)



