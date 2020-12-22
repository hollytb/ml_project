import os, glob
import pandas as pd

path = "/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_branch/ml_project/NON_CLICKBAIT_DATA/NY_TIMES_DATA"

all_files = glob.glob(os.path.join(path, "*.csv"))
df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged   = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv( "NY_Times_merged_2010-2020.csv")