import pandas as pd
import numpy as np
from visualization.dashboard import DashboardApp
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dashboard.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run just the dashboard"""
    try:
        # Load the clustered data
        logger.info("Loading clustered data...")
        data = pd.read_json('data/processed/clustered_articles.json', lines=True)
        
        # Log data statistics
        logger.debug(f"Loaded {len(data)} articles")
        logger.debug(f"Columns in data: {data.columns.tolist()}")
        logger.debug(f"Sample article content: {data['content'].iloc[0][:200] if not data.empty else 'No data'}")
        logger.debug(f"Unique clusters: {data['cluster'].unique().tolist() if 'cluster' in data.columns else 'No cluster column'}")
        
        # Create dummy embeddings for visualization (since we don't have the original embeddings)
        logger.info("Creating visualization coordinates...")
        unique_clusters = data['cluster'].unique()
        n_clusters = len(unique_clusters)
        logger.debug(f"Number of clusters: {n_clusters}")
        
        # Generate circular coordinates for each cluster
        embeddings_2d = np.zeros((len(data), 2))
        for i, cluster in enumerate(unique_clusters):
            cluster_mask = data['cluster'] == cluster
            n_points = cluster_mask.sum()
            logger.debug(f"Cluster {cluster} has {n_points} points")
            
            # Generate points in a circle for this cluster
            theta = np.linspace(0, 2*np.pi, n_points)
            radius = 2  # Adjust this for visualization
            center_x = radius * np.cos(2*np.pi * i / n_clusters)
            center_y = radius * np.sin(2*np.pi * i / n_clusters)
            
            embeddings_2d[cluster_mask, 0] = center_x + radius * 0.2 * np.cos(theta)
            embeddings_2d[cluster_mask, 1] = center_y + radius * 0.2 * np.sin(theta)
        
        # Get processed texts (we'll use content field as processed text for simplicity)
        texts = data['content'].tolist()
        logger.debug(f"Number of texts: {len(texts)}")
        logger.debug(f"Sample text length: {len(texts[0]) if texts else 0}")
        
        # Initialize and run dashboard
        logger.info("Initializing dashboard...")
        app = DashboardApp(data, embeddings_2d, data['cluster'].values, texts)
        app.run_server(debug=True, port=8050)
        
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        raise

if __name__ == "__main__":
    main() 