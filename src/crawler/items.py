from scrapy import Item, Field

class ArticleItem(Item):
    title = Field()
    content = Field()
    url = Field()
    source = Field()
    published_date = Field()
    author = Field()
    category = Field() 