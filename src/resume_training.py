import os
import sys
import torch
import logging
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from topnet import IntegratedTopicNetTopNet, TopicNetTopNetTrainer, NewsDataset
from processing.text_processor import TextProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)

def resume_training():
    """Resume training from checkpoint"""
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    
    # Initialize model and trainer (same as in experiment)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    model = IntegratedTopicNetTopNet(
        vocab_size=10000,
        num_topics=20,
        semantic_graph_path=None,
        embedding_dim=300,
        hidden_dim=512,
        latent_dim=128,
        device=device
    )
    
    # Load sample data
    data_path = '../data/processed/clustered_articles.json'
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    import pandas as pd
    data = pd.read_json(data_path, lines=True)
    
    # Sample data for training
    if len(data) > 1000:
        data = data.sample(n=1000, random_state=42)
    
    # Create datasets
    from sklearn.model_selection import train_test_split
    
    # Create categories from clusters
    if 'cluster' in data.columns:
        categories = data['cluster'].tolist()
        unique_clusters = sorted(list(set(categories)))
        cluster_mapping = {cluster: i % 8 for i, cluster in enumerate(unique_clusters)}
        categories = [cluster_mapping[cluster] for cluster in categories]
    else:
        categories = [0] * len(data)
    
    # Split data
    train_articles, temp_articles, train_categories, temp_categories = train_test_split(
        data['content'].tolist(),
        categories,
        test_size=0.4,
        random_state=42,
        stratify=categories
    )
    
    val_articles, test_articles, val_categories, test_categories = train_test_split(
        temp_articles,
        temp_categories,
        test_size=0.5,
        random_state=42,
        stratify=temp_categories
    )
    
    # Create datasets
    train_dataset = NewsDataset(train_articles, train_categories)
    val_dataset = NewsDataset(val_articles, val_categories)
    test_dataset = NewsDataset(test_articles, test_categories)
    
    # Create trainer
    trainer = TopicNetTopNetTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=16,
        learning_rate=1e-4,
        device=device,
        use_wandb=False  # Disable wandb for resume
    )
    
    # Load checkpoint
    trainer.load_model(checkpoint_path)
    
    logger.info("Checkpoint loaded successfully!")
    logger.info(f"Training history: {len(trainer.train_history['epoch'])} epochs completed")
    
    # Resume training for remaining epochs
    remaining_epochs = 50 - len(trainer.train_history['epoch'])
    if remaining_epochs > 0:
        logger.info(f"Resuming training for {remaining_epochs} more epochs...")
        trainer.train(num_epochs=remaining_epochs, eval_every=2)
    else:
        logger.info("Training already completed!")
    
    # Generate final evaluation report
    logger.info("Generating final evaluation report...")
    report = trainer.generate_evaluation_report()
    
    logger.info("Training completed successfully!")
    logger.info(f"Final report: {report}")

if __name__ == "__main__":
    resume_training() 