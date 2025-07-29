import json
import os
from datetime import datetime
from pathlib import Path
import logging

class ImmediateJsonPipeline:
    def __init__(self):
        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure output directory exists
        self.output_dir = Path('data/raw')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / 'articles.json'
        
        # Create empty file if it doesn't exist
        if not self.output_file.exists():
            self.output_file.touch()
            self.logger.info(f"Created new file: {self.output_file}")
            
        # Load existing URLs to avoid duplicates
        self.existing_urls = set()
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        article = json.loads(line.strip())
                        self.existing_urls.add(article['url'])
                    except json.JSONDecodeError:
                        continue
            self.logger.info(f"Loaded {len(self.existing_urls)} existing URLs")
        except FileNotFoundError:
            self.logger.warning(f"No existing file found at {self.output_file}")
        except Exception as e:
            self.logger.error(f"Error loading existing URLs: {e}")

    def process_item(self, item, spider):
        # Skip if article already exists
        if item['url'] in self.existing_urls:
            self.logger.debug(f"Skipping existing article: {item['url']}")
            return item
            
        # Add URL to tracked set
        self.existing_urls.add(item['url'])
        
        # Write article immediately
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                line = json.dumps(dict(item)) + '\n'
                f.write(line)
                self.logger.info(f"Saved new article: {item['url']}")
        except Exception as e:
            self.logger.error(f"Error saving article {item['url']}: {e}")
            
        return item

    def open_spider(self, spider):
        self.logger.info("Pipeline started")
        
    def close_spider(self, spider):
        self.logger.info(f"Pipeline closed. Total unique URLs: {len(self.existing_urls)}") 