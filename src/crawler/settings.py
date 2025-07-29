# Scrapy settings

BOT_NAME = 'news_crawler'

SPIDER_MODULES = ['crawler.spiders']
NEWSPIDER_MODULE = 'crawler.spiders'

# Crawl responsibly by identifying yourself
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 4  # Reduced to be more conservative
CONCURRENT_REQUESTS_PER_DOMAIN = 2  # Reduced to be more conservative

# Configure a delay for requests for the same website
DOWNLOAD_DELAY = 5  # Increased delay
RANDOMIZE_DOWNLOAD_DELAY = True

# Configure timeout
DOWNLOAD_TIMEOUT = 30

# Enable cookies
COOKIES_ENABLED = True

# Configure retry middleware
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [429, 500, 502, 503, 504, 408]  # Removed 401 since it's authentication

# Enable caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # Cache for 24 hours
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [401, 403, 404, 429, 500, 502, 503, 504]
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# Configure output format
FEED_FORMAT = None
FEED_URI = None
FEED_EXPORT_ENCODING = 'utf-8'

# Add AutoThrottle
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 5  # Increased start delay
AUTOTHROTTLE_MAX_DELAY = 60  # Increased max delay
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0  # More conservative
AUTOTHROTTLE_DEBUG = True

# Enable and configure logging
LOG_LEVEL = 'DEBUG'  # Changed from INFO to DEBUG
LOG_FILE = 'crawler.log'
LOG_ENABLED = True
LOG_STDOUT = True

# Configure closespider extension
CLOSESPIDER_ITEMCOUNT = 10000  # Stop after collecting 10,000 items
CLOSESPIDER_TIMEOUT = 7200     # Stop after 2 hours (in seconds)

# Configure item pipelines
ITEM_PIPELINES = {
    'crawler.pipelines.ImmediateJsonPipeline': 300
}

# Ensure directory exists for output
import os
os.makedirs('data/raw', exist_ok=True) 