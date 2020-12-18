import scrapy
from pandas import read_csv
from readability.readability import Document

PATH_TO_DATA = "data/buzzfeed_urls.csv"


class HeadlineSpider(scrapy.Spider):
    name = "headline_spider"
    start_urls = read_csv(PATH_TO_DATA).iloc[:, 0].tolist()
    print(start_urls)

    def parse(self, response):
        doc = Document(response.text)
        yield {
            'full_title': doc.title(),
            'date': response.selector.xpath('//time/@datetime').getall()
        }

# TODO remove any headlines without dates from buzzfeed_raw.csv!
