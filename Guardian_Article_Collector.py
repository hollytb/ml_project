guardian_key='a62d7d83-bdfc-4677-8d18-09ce9b10ff68'
import json
import requests

url='https://content.guardianapis.com/search?'
params = {"api-key": "a62d7d83-bdfc-4677-8d18-09ce9b10ff68","page-size": "200" ,"from-date": "2020-01-01","to-date": "2020-12-31"}

cur=0
results=[]
for pi in range(1,16):
	params["page"] = pi
	results.append(requests.get(url, params))

def guardian_call(params):
	response = requests.get(url, params)
	return json.loads(response.text)

articles = guardian_call(params)

articles['response']['results'][0]

def parse_results(results):
    parsed_result=[]
    for article in results:
        article_list = [article['webTitle'],article['webPublicationDate'], 'Guardian', 0]
        parsed_result.append(article_list)
                    
    return parsed_result

x=parse_results(articles['response']['results'])

import pandas as pd
pd.DataFrame(x).head()

columns = ['text','date', 'source', 'value']
df = pd.DataFrame(columns=columns) 
df.to_csv('guardian_data_2020.csv')

def data_save(parsed_results, csv_filename):

    existing=pd.read_csv(csv_filename,index_col=0)
    new = pd.DataFrame(parsed_results,columns=columns)
    df = pd.concat([existing,new])
    df.to_csv(csv_filename)

cur = 16
while cur < 19:
    params['page'] = cur
    results = guardian_call(params)
    parsed_results = parse_results(results['response']['results'])
    data_save(parsed_results, 'guardian_data_2020.csv')
    cur += 1


guardian_df = pd.read_csv('guardian_data_2020.csv',index_col=0)