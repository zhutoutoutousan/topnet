import scrapy
from datetime import datetime, timedelta
from ..items import ArticleItem
import time
from urllib.parse import urljoin, urlparse
from scrapy.exceptions import CloseSpider

class NewsSpider(scrapy.Spider):
    name = 'news'
    allowed_domains = ['apnews.com', 'bbc.com']  # Both AP News and BBC
    custom_settings = {
        'CLOSESPIDER_ITEMCOUNT': 10000,
        'CLOSESPIDER_TIMEOUT': 7200
    }
    
    def __init__(self, *args, **kwargs):
        super(NewsSpider, self).__init__(*args, **kwargs)
        self.cutoff_date = datetime.now() - timedelta(days=180)
        self.visited_urls = set()
        self.article_count = 0
        self.target_article_count = 10000
        self.start_urls = self.generate_archive_urls()
        
    def generate_archive_urls(self):
        """Generate URLs for AP News and BBC archives"""
        urls = []
        current_date = datetime.now()
        
        # Generate URLs starting from 6 months ago up to today
        dates = [(current_date - timedelta(days=i)) for i in range(180)]
        # Sort dates from oldest to newest
        dates.sort()  # This will sort in ascending order
        
        for date in dates:
            # AP News archive URLs
            urls.append(f'https://apnews.com/hub/archive?date={date.strftime("%Y-%m-%d")}')
            # BBC archive URLs - Updated format
            urls.append(f'https://www.bbc.com/pages/archive/{date.strftime("%Y/%m/%d")}')
        
        # Add section pages for broader coverage
        ap_sections = [
            'world-news',
            'us-news',
            'politics',
            'sports',
            'entertainment',
            'business',
            'technology',
            'science',
            'health',
            'oddities',
            'religion',
            'travel'
        ]
        
        bbc_sections = [
            'news',
            'sport',
            'business',
            'technology',
            'entertainment',
            'science',
            'health'
        ]
        
        for section in ap_sections:
            urls.append(f'https://apnews.com/hub/{section}')
            
        for section in bbc_sections:
            urls.append(f'https://www.bbc.com/{section}')
        
        return urls
    
    def start_requests(self):
        """Spread out initial requests"""
        for url in self.start_urls:
            if self.article_count >= self.target_article_count:
                self.logger.info(f"Reached target article count: {self.article_count}")
                return
                
            yield scrapy.Request(
                url,
                callback=self.parse,
                dont_filter=True,
                meta={'dont_retry': False, 'download_timeout': 30},
                errback=self.errback_httpbin,
                headers={'User-Agent': self.settings.get('USER_AGENT')}
            )
            time.sleep(2)  # Conservative delay between requests
    
    def errback_httpbin(self, failure):
        """Handle request failures"""
        self.logger.error(f'Request failed: {failure.value}')
    
    def parse(self, response):
        """Parse article links from news pages"""
        if self.article_count >= self.target_article_count:
            return
            
        if response.status == 429:
            self.logger.info("Received 429, backing off...")
            return
        
        # Debug info
        self.logger.info(f"Parsing page: {response.url}")
        
        article_links = set()
        domain = urlparse(response.url).netloc
        
        if 'apnews.com' in domain:
            # AP News specific selectors
            selectors = [
                'div[data-key="card-container"] a::attr(href)',
                'div[data-key="feed-card"] a::attr(href)',
                'a[data-key="story-link"]::attr(href)',
                'a[href*="/article/"]::attr(href)'
            ]
            
            for selector in selectors:
                links = response.css(selector).getall()
                self.logger.info(f"AP News - Found {len(links)} links with selector: {selector}")
                article_links.update(links)
                
            # AP News pagination
            next_page = (
                response.css('a[data-next="true"]::attr(href)').get() or
                response.css('a.next::attr(href)').get()
            )
            
        elif 'bbc.com' in domain:
            # BBC specific selectors based on actual HTML
            selectors = [
                'div.PageList-items-item a.Link::attr(href)',  # Main article links
                'div.PagePromo-media a.Link::attr(href)',  # Media article links
                'div.PagePromo a.Link::attr(href)',  # Any article in PagePromo
                'a[href*="/articles/"]::attr(href)',  # Article links
                'a[href*="/news/"]::attr(href)'  # News links
            ]
            
            for selector in selectors:
                links = response.css(selector).getall()
                self.logger.info(f"BBC - Found {len(links)} links with selector: {selector}")
                article_links.update(links)
            
            # BBC pagination - based on actual HTML
            next_page = response.css('a[data-testid="pagination-next-page"]::attr(href)').get()
        
        # Process article links
        for link in article_links:
            if self.article_count >= self.target_article_count:
                return
                
            if link and link not in self.visited_urls:
                full_url = response.urljoin(link)
                
                # Skip non-article pages and already visited URLs
                if not any(pattern in full_url for pattern in ['/article/', '/articles/', '/news/']) or full_url in self.visited_urls:
                    continue
                    
                self.visited_urls.add(full_url)
                self.logger.info(f"Found article link: {full_url}")
                
                yield scrapy.Request(
                    full_url,
                    callback=self.parse_article,
                    meta={'dont_retry': False, 'download_timeout': 30},
                    headers={'User-Agent': self.settings.get('USER_AGENT')}
                )
        
        # Follow pagination if available
        if next_page and self.article_count < self.target_article_count:
            next_url = response.urljoin(next_page)
            self.logger.info(f"Following next page: {next_url}")
            yield response.follow(
                next_url,
                callback=self.parse,
                meta={'dont_retry': False, 'download_timeout': 30},
                headers={'User-Agent': self.settings.get('USER_AGENT')}
            )
    
    def parse_article(self, response):
        """Parse articles from different sources"""
        if response.status == 429:
            self.logger.info("Received 429 on article, backing off...")
            return
        
        if self.article_count >= self.target_article_count:
            self.logger.info(f"Reached target count of {self.target_article_count} articles. Stopping spider.")
            raise CloseSpider(f"Reached target count of {self.target_article_count} articles")
        
        article = ArticleItem()
        domain = urlparse(response.url).netloc
        
        # Common fields
        article['url'] = response.url
        article['source'] = domain
        
        if 'apnews.com' in domain:
            # AP News specific parsing
            article['title'] = response.css('h1::text').get()
            content_parts = []
            content_parts.extend(response.css('.Article p::text').getall())
            content_parts.extend(response.css('.RichTextStoryBody p::text').getall())
            content_parts.extend(response.css('[data-key="article-contents"] p::text').getall())
            article['content'] = ' '.join(part.strip() for part in content_parts if part.strip())
            
            article['published_date'] = (
                response.css('time::attr(datetime)').get() or
                response.css('meta[property="article:published_time"]::attr(content)').get()
            )
            
            article['author'] = (
                response.css('.byline a::text').get() or
                response.css('.Author::text').get() or
                response.css('[data-key="byline"]::text').get() or
                'AP News'
            )
            
            article['category'] = (
                response.css('.HubLink::text').get() or
                response.css('[data-key="hub-link"]::text').get() or
                response.url.split('/hub/')[-1].split('/')[0] if '/hub/' in response.url
                else 'general'
            )
            
        elif 'bbc.com' in domain:
            # BBC specific parsing based on actual HTML structure
            article['title'] = (
                response.css('h1::text').get() or
                response.css('[data-testid="card-headline"]::text').get()
            )
            
            content_parts = []
            content_parts.extend(response.css('.Article p::text').getall())
            content_parts.extend(response.css('.ArticleBody p::text').getall())
            content_parts.extend(response.css('[data-component="text-block"] p::text').getall())
            article['content'] = ' '.join(part.strip() for part in content_parts if part.strip())
            
            # Handle BBC's Unix timestamp format
            timestamp = (
                response.css('[data-posted-date-timestamp]::attr(data-posted-date-timestamp)').get() or
                response.css('time::attr(datetime)').get()
            )
            if timestamp:
                if timestamp.isdigit():
                    # Convert Unix timestamp (milliseconds) to ISO format
                    timestamp = int(timestamp) / 1000
                    article['published_date'] = datetime.fromtimestamp(timestamp).isoformat() + 'Z'
                else:
                    article['published_date'] = timestamp
            
            article['category'] = (
                response.css('.ArticleCategory::text').get() or
                response.url.split('/')[3] if len(response.url.split('/')) > 3
                else 'general'
            )
            
            article['author'] = response.css('.ArticleAuthor::text').get() or 'BBC News'
        
        # Only yield articles with content and within date range
        if article['content'] and article['published_date']:
            try:
                # Handle multiple date formats
                date_str = article['published_date'][:19]
                for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                    try:
                        pub_date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                
                if pub_date >= self.cutoff_date:
                    self.article_count += 1
                    self.logger.info(f"Article {self.article_count}: {article['title']}")
                    yield article
                    
                    if self.article_count >= self.target_article_count:
                        self.logger.info(f"Reached target count of {self.target_article_count} articles. Stopping spider.")
                        raise CloseSpider(f"Reached target count of {self.target_article_count} articles")
                        
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Could not parse date for article: {article['url']} - Error: {e}")

    def closed(self, reason):
        """Log statistics when spider closes"""
        self.logger.info(f"Spider closed: {reason}")
        self.logger.info(f"Total URLs visited: {len(self.visited_urls)}")
        self.logger.info(f"Total articles collected: {self.article_count}") 