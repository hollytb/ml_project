import pandas as pd
from sklearn.preprocessing import MinMaxScaler

years=[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

for YEAR in years:
	cy = '/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_'+ str(YEAR) + '.csv'

	features_csv = cy
	df = pd.read_csv(features_csv, index_col=0)
	print(df.shape)

	numb_cols = ["word_count",
	             "question_mark",
	             "exclamation_mark",
	             "start_digit",
	             "start_question",
	             "longest_word_len",
	             "avg_word_len",
	             "ratio_stopwords"]

	scaler = MinMaxScaler()
	df[numb_cols] = scaler.fit_transform(df[numb_cols])

	df.to_csv(path_or_buf=cy)
