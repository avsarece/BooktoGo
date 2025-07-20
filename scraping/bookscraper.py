"""Book scraping"""
import scrapy
import pandas as pd


class MyScrapper(scrapy.Spider):
    name = "myCrawler"
    custom_settings = {
        'FEED_EXPORT_ENCODING': 'utf-8'
    }
    allowed_domains = ['books.toscrape.com']
    start_urls = ['https://books.toscrape.com/catalogue/page-' + str(i) + '.html' for i in range(1, 51)]

    def parse(self, response):
        for book in response.css('article.product_pod'):
            relative_url = book.css('h3 a::attr(href)').get()
            normalized_url = relative_url.replace('../../../', 'catalogue/')
            full_url = response.urljoin(normalized_url)

            yield scrapy.Request(
                full_url,
                callback=self.parse_book_detail,
                meta={'book_url': full_url}
            )

    def parse_book_detail(self, response):
        """Parse book details. This includes book cover url and book url for recommendation interface purposes."""
        genre = response.css('ul.breadcrumb li:nth-child(3) a::text').get()
        description = response.css('div#product_description + p::text').get()
        title = response.css('h1 ::text').get()
        rating_class = response.css('p.star-rating::attr(class)').get()
        rating_str = rating_class[12:].strip()
        book_url = response.meta.get('book_url')

        dict_num = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
        rating = dict_num.get(rating_str, 0)

        table = response.css('table')
        dict_so_far2 = {}
        for tr in table.css('tr'):
            row_header = tr.css('th::text').get()
            row_value = tr.css('td::text').get()
            dict_so_far2[row_header] = row_value

        cover_relative = response.css('.item.active img::attr(src)').get()
        cover_url = response.urljoin(cover_relative) if cover_relative else None

        yield {**dict_so_far2,
               'Title': title,
               'Genre': genre,
               'Rating': rating,
               'Description': description,
               'Cover_URL': cover_url,
               'URL': book_url
               }


def create_table(filename: str) -> None:
    """
    Create book info csv table using the json file and print the total number of books in the website.
    """
    df = pd.read_json(filename)
    df_sorted1 = df.drop(columns=['Product Type', 'Price (excl. tax)', 'Number of reviews', 'Tax'], errors='ignore')

    df_sorted2 = df_sorted1.sort_values('Genre', ascending=True)
    df_sorted2.reset_index(drop=True, inplace=True)
    df_sorted2.index += 1
    df_sorted3 = df_sorted2.reindex(
        columns=['UPC', 'URL', 'Cover_URL', 'Title', 'Genre', 'Price (incl. tax)', 'Rating', 'Availability',
                 'Description'])

    df_cleaned = df_sorted3.dropna(subset=['Description'])
    df_cleaned.to_csv("book_data.csv", index=False, encoding='utf-8')
    print('Number of books:', len(df_sorted3))
