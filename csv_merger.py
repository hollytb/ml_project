import os, glob
import pandas as pd


years=["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

for YEAR in years:
	path = "/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/Years/" + YEAR 
	all_files = glob.glob(os.path.join(path, "*.csv"))
	#print("FILES:",all_files)
	df_from_each_file = (pd.read_csv(f, sep=',', index_col=0) for f in all_files)
	df_merged   = pd.concat(df_from_each_file, ignore_index=True)
	df_merged.to_csv(YEAR + ".csv")
