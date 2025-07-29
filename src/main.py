import os
import pandas as pd
from crawler.spiders.news_spider import NewsSpider
from processing.text_processor import TextProcessor
from processing.embedding_generator import EmbeddingGenerator
from clustering.cluster_analyzer import ClusterAnalyzer
from visualization.dashboard import DashboardApp
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import logging
import time
import traceback
import sys

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dashboard.log')
    ]
)

logger = logging.getLogger(__name__)

def run_crawler():
    """Run the news crawler"""
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    # Get Scrapy settings
    settings = get_project_settings()
    
    # Ensure pipeline is enabled
    settings.set('ITEM_PIPELINES', {
        'crawler.pipelines.ImmediateJsonPipeline': 300
    })
    
    # Create crawler process with our settings
    process = CrawlerProcess(settings)
    
    # Configure and start the spider
    process.crawl(NewsSpider)
    process.start()

def load_data():
    """Load and preprocess the crawled data"""
    # Wait for file to be created and have content
    data_file = 'data/raw/articles.json'
    max_retries = 10
    retry_delay = 2
    
    for i in range(max_retries):
        if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
            try:
                # Load raw data
                data = pd.read_json(data_file, lines=True)
                if not data.empty:
                    break
            except (pd.errors.EmptyDataError, ValueError) as e:
                logging.warning(f"Attempt {i+1}: File exists but couldn't be read: {e}")
        
        if i < max_retries - 1:  # Don't sleep on the last iteration
            logging.info(f"Waiting for data file to be ready (attempt {i+1}/{max_retries})...")
            time.sleep(retry_delay)
    else:
        raise RuntimeError(f"Failed to load data after {max_retries} attempts")
    
    # Initialize text processor
    processor = TextProcessor()
    
    # Process all texts
    processed_texts = [processor.process_text(text) for text in data['content']]
    
    return data, processed_texts

def generate_embeddings(texts):
    """Generate embeddings for the processed texts"""
    generator = EmbeddingGenerator(model_type='sbert')
    return generator.generate_embeddings(texts)

def perform_clustering(embeddings):
    """Perform clustering on the embeddings"""
    analyzer = ClusterAnalyzer(method='hdbscan')
    
    # Reduce dimensions for visualization
    embeddings_2d = analyzer.reduce_dimensions(embeddings)
    
    # Perform clustering
    labels = analyzer.fit_predict(embeddings)
    
    # Evaluate clustering
    metrics = analyzer.evaluate_clustering(embeddings)
    print("Clustering Metrics:", metrics)
    
    return embeddings_2d, labels

def main():
    """Main execution function"""
    try:
        # Create necessary directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        try:
            # Step 1: Run crawler
            logger.info("Crawling news articles...")
            run_crawler()
            
            # Step 2: Load and process data
            logger.info("Loading and processing data...")
            data, processed_texts = load_data()
            logger.debug(f"Loaded {len(data)} articles and {len(processed_texts)} processed texts")
            
            # Step 3: Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = generate_embeddings(processed_texts)
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            
            # Step 4: Perform clustering
            logger.info("Performing clustering...")
            embeddings_2d, cluster_labels = perform_clustering(embeddings)
            logger.debug(f"2D embeddings shape: {embeddings_2d.shape}, unique clusters: {len(set(cluster_labels))}")
            
            # Save processed data
            data['cluster'] = cluster_labels
            data.to_json('data/processed/clustered_articles.json', orient='records', lines=True)
            
            # Step 5: Launch dashboard
            logger.info("Initializing dashboard...")
            try:
                dashboard = DashboardApp(
                    data=data,
                    embeddings_2d=embeddings_2d,
                    cluster_labels=cluster_labels,
                    texts=processed_texts
                )
                logger.info("Dashboard initialized, starting server...")
                dashboard.run_server(debug=True, port=8050)
            except Exception as e:
                logger.error(f"Dashboard error: {str(e)}")
                logger.error("Dashboard traceback: " + traceback.format_exc())
                raise
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            logger.error("Processing traceback: " + traceback.format_exc())
            raise
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}")
        logger.error("Main process traceback: " + traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 