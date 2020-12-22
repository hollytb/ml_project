key='EE4r1tU8dgaQej94KTnlJxWPglKKaz4e'
secret='AbqACp2UngzDCOWu'

import requests
import json

url = 'https://api.nytimes.com/svc/archive/v1/2020/06.json?api-key={key:AbqACp2UngzDCOWu}'

r=requests.get(url)
json_data=r.json()
from pynytimes import NYTAPI

nyt = NYTAPI("EE4r1tU8dgaQej94KTnlJxWPglKKaz4e")

import datetime

years=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

for YEAR in years:

	data = nyt.archive_metadata(date = datetime.datetime(YEAR, 6, 1))

	headlines_nyt=[]
	#for i in range(len(data)):
	for i in range(1001):

		print(i)
		headlines_nyt.append((data[i]['headline']['main'], data[i]['pub_date'], 'NY_TIMES', 0))

	import pandas as pd
	ny_times_datafinal = pd.DataFrame(headlines_nyt,columns=['text','date','source', 'value'])

	ny_times_datafinal.to_csv('nytimes_data_'+ str(YEAR) + '.csv')

