import newspaper
from newspaper import Article
from newspaper import Source
from newspaper import news_pool
import pandas as pd

# The various News Sources we will like to web scrape from
#gamespot = newspaper.build('https://www.gamespot.com/news/', memoize_articles=False)
#bbc = newspaper.build("https://www.bbc.com/news", memoize_articles=False)
#rte = newspaper.build("https://www.rte.ie/news/", memoize_articles=False)
#cnn = newspaper.build("https://edition.cnn.com/", memoize_articles=False)
#bloomberg = newspaper.build("https://www.bloomberg.com/europe", memoize_articles=False)
#newyork = newspaper.build("https://www.nytimes.com/international/", memoize_articles=False)
#bbc1 = newspaper.build("http://news.bbc.co.uk/onthisday/low/years/2000/", memoize_articles=False)

#ny1 = newspaper.build("https://www.nytimes.com/search?dropmab=true&endDate=20001212&query=&sort=best&startDate=20000101&types=article", memoize_articles=False)
ww = newspaper.build("https://waterfordwhispersnews.com/page/6/?s=2015", memoize_articles=False)

#print('RTE', rte.size())
#print('BBC', bbc.size())
#print('cnn', cnn.size())
#print('bloomberg', bloomberg.size())
#print('newyork', newyork.size())

# Place the sources in a list
papers = [ww]

# Essentially you will be downloading 4 articles parallely per source.
# Since we have two sources, that means 8 articles are downloaded at any one time. 
# Greatly speeding up the processes.
# Once downloaded it will be stored in memory to be used in the for loop below 
# to extract the bits of data we want.
news_pool.set(papers, threads_per_source=8)

news_pool.join()

# Create our final dataframe
final_df = pd.DataFrame()

# Create a download limit per sources
# NOTE: You may not want to use a limit
#limit = 10

for source in papers:
    # temporary lists to store each element we want to extract
    list_title = []
    list_text = []
    list_source =[]
    list_date = []

    count = 0

    for article_extract in source.articles:
        article_extract.parse()

#        if count > limit: # Lets have a limit, so it doesnt take too long when you're
#            break         # running the code. NOTE: You may not want to use a limit

        # Appending the elements we want to extract
        list_title.append(article_extract.title)
        list_date.append(article_extract.publish_date)
        list_source.append(article_extract.source_url)


        # Update count
        count +=1

        print(count)

    temp_df = pd.DataFrame({'Title': list_title, 'Date': list_date, 'Source': list_source})
    # Append to the final DataFrame
    final_df = final_df.append(temp_df, ignore_index = True)
    
# From here you can export this to csv file
final_df.to_csv('aac.csv')