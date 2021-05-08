import scrapy
import logging
from typing import List


def scrape_article_text(sel: scrapy.Selector) -> List[str]:
    return sel.xpath(
        ".//*[not(self::script) and not(self::style) and not(ancestor-or-self::a) and not(ancestor-or-self::figure)]/text()"
    ).getall()


class YnetSpider(scrapy.Spider):
    name = "ynet"
    allowed_domains = ["ynet.co.il"]
    start_urls = ["https://www.ynet.co.il/home/0,7340,L-4269,00.html"]

    def parse(self, response):

        allowed_cat = getattr(self, "tag", None)

        for anchor in response.css("table.classMainTableBox a.indexw"):
            url = anchor.attrib["href"]
            category = anchor.css("::text").get()
            if not allowed_cat or category == allowed_cat:
                yield response.follow(url, callback=self.parse_category)

    def parse_category(self, response):
        links = response.css("a.smallheader")
        yield from response.follow_all(links, callback=self.parse_month)

    def parse_month(self, response):
        articles = response.css(
            "td.ghciArticleIndex1 table:nth-last-child(2) a.smallheader"
        )
        # yield {"link": article.get() for article in articles}
        yield from response.follow_all(articles, callback=self.parse_article)

    def parse_article(self, response):
        if response.css("div.respArticleBackground"):
            yield from self.parse_article_2020(response)
        elif response.css("div#ArticleHeaderComponent"):
            yield from self.parse_article_2021(response)
        else:
            logging.error("Unknown format for {response}")

    def parse_article_2020(self, response: scrapy.http.Response):
        header = response.css("div.respArticleBackground")
        main_title = header.css("h1.art_header_title::text").get()
        sub_title = header.css("h2.art_header_sub_title::text").get()
        author_and_date = header.css("span.art_header_footer_author")
        if len(author_and_date) != 2:
            logging.error(
                f"pre-2021 style article, couldn't parse author and date for {response}"
            )
            return
        authors, date = author_and_date[0], author_and_date[1]
        authors = authors.css("span::text").getall()
        date = date.css("span::text").get()

        text = scrape_article_text(response.css("div.art_body"))

        yield {
            "fmt": "pre2021",
            "url": response.url,
            "main_title": main_title,
            "sub_title": sub_title,
            "authors": authors,
            "date": date,
            "text": text,
        }

    def parse_article_2021(self, response: scrapy.http.Response):
        header = response.css("div#ArticleHeaderComponent")
        if not header:
            logging.error(f"Couldn't find article in {response}")
            return
        main_title = header.css("h1.mainTitle::text").get()
        sub_title = header.css("h2.subTitle::text").get()
        authors = header.css("div.authors a::text").getall()
        date = header.css("div.date::text").get()
        text = scrape_article_text(response.css("div[data-contents]"))
        yield {
            "fmt": "2021",
            "url": response.url,
            "main_title": main_title,
            "sub_title": sub_title,
            "authors": authors,
            "date": date,
            "text": text,
        }
