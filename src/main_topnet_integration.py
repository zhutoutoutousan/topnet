import os
import sys
import pandas as pd
import torch
import logging
import argparse
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from topnet import IntegratedTopicNetTopNet, TopicNetTopNetTrainer, NewsDataset
from experiments.integrated_experiment import IntegratedExperiment
from processing.text_processor import TextProcessor
from visualization.dashboard import DashboardApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('topnet_integration.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TopNet-TopicNet Integration for News Analysis and Story Generation')
    
    parser.add_argument('--mode', choices=['experiment', 'train', 'demo', 'dashboard'], 
                       default='demo', help='Operation mode')
    parser.add_argument('--data_path', type=str, default='../data/processed/clustered_articles.json',
                       help='Path to news dataset')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model (for demo mode)')
    parser.add_argument('--results_dir', type=str, default='../results/topnet_integration',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_topics', type=int, default=20,
                       help='Number of topics')
    parser.add_argument('--vocab_size', type=int, default=10000,
                       help='Vocabulary size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--port', type=int, default=8051,
                       help='Port for dashboard (dashboard mode)')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> str:
    """Setup computation device"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    logger.info(f"Using device: {device}")
    return device

def load_sample_data(data_path: str, num_samples: int = 100) -> pd.DataFrame:
    """Load and prepare sample data"""
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_json(data_path, lines=True)
    
    # Sample data for demo
    if len(data) > num_samples:
        data = data.sample(n=num_samples, random_state=42)
    
    logger.info(f"Loaded {len(data)} articles")
    return data

def create_demo_model(args) -> IntegratedTopicNetTopNet:
    """Create the integrated model for demonstration"""
    logger.info("Creating integrated TopicNet-TopNet model...")
    
    model = IntegratedTopicNetTopNet(
        vocab_size=args.vocab_size,
        num_topics=args.num_topics,
        semantic_graph_path=None,  # Will be created dynamically
        embedding_dim=300,
        hidden_dim=512,
        latent_dim=128,
        device=args.device
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model

def run_experiment_mode(args):
    """Run comprehensive experiments"""
    logger.info("Running comprehensive TopNet-TopicNet integration experiments...")
    
    experiment = IntegratedExperiment(
        data_path=args.data_path,
        results_dir=args.results_dir,
        device=args.device
    )
    
    # Run complete experimental pipeline
    results = experiment.run_complete_experiment()
    
    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved to: {args.results_dir}")
    
    # Print key findings
    print("\n" + "="*60)
    print("EXPERIMENTAL RESULTS - KEY FINDINGS")
    print("="*60)
    
    for finding in results['key_findings']:
        print(f"✓ {finding}")
    
    print("\n" + "="*60)
    print("INNOVATION CONTRIBUTIONS")
    print("="*60)
    
    for contribution in results['innovation_contributions']:
        print(f"• {contribution}")
    
    return results

def run_training_mode(args):
    """Run training mode"""
    logger.info("Running training mode...")
    
    # Load data
    data = load_sample_data(args.data_path, num_samples=1000)
    
    # Prepare datasets
    from sklearn.model_selection import train_test_split
    
    train_articles, test_articles = train_test_split(
        data['content'].tolist(),
        test_size=0.2,
        random_state=42
    )
    
    val_articles, test_articles = train_test_split(
        test_articles,
        test_size=0.5,
        random_state=42
    )
    
    # Create datasets
    train_dataset = NewsDataset(train_articles[:500])
    val_dataset = NewsDataset(val_articles[:100])
    test_dataset = NewsDataset(test_articles[:100])
    
    # Create model
    model = create_demo_model(args)
    
    # Create trainer
    trainer = TopicNetTopNetTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb,
        device=args.device
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        eval_every=3,
        save_every=5,
        save_path=f"{args.results_dir}/checkpoints"
    )
    
    # Final evaluation
    evaluation_report = trainer.generate_evaluation_report()
    
    logger.info("Training completed successfully!")
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Final Topic Coherence: {evaluation_report['topic_discovery']['topic_coherence']:.4f}")
    print(f"Story Generation Quality: {evaluation_report['story_generation']['avg_story_quality']:.4f}")
    print(f"Cross-modal Alignment: {evaluation_report['cross_modal_alignment']['topic_story_alignment']}")
    
    return trainer, evaluation_report

def run_demo_mode(args):
    """Run demonstration mode"""
    logger.info("Running demonstration mode...")
    
    # Load sample data
    data = load_sample_data(args.data_path, num_samples=50)
    
    # Create model
    model = create_demo_model(args)
    
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading pre-trained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Demonstrate capabilities
    print("\n" + "="*60)
    print("TOPNET-TOPICNET INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Sample articles for demonstration
    sample_articles = data['content'].iloc[:5].tolist()
    
    processor = TextProcessor()
    
    for i, article in enumerate(sample_articles):
        print(f"\n--- Article {i+1} ---")
        print(f"Original: {article[:200]}...")
        
        # Process article
        processed = processor.process_text(article)
        
        # Simple tokenization for demo (in practice, use proper tokenizer)
        tokens = [hash(word) % args.vocab_size for word in processed.split()[:512]]
        if len(tokens) < 512:
            tokens.extend([0] * (512 - len(tokens)))
        
        article_tokens = torch.tensor([tokens], dtype=torch.long, device=args.device)
        
        with torch.no_grad():
            # Analyze news content
            results = model.analyze_news_content(
                article_tokens,
                generate_stories=True,
                story_length=150
            )
            
            # Display results
            topic_dist = results['topic_analysis']['topic_distributions'][0]
            top_topics = torch.topk(topic_dist, 3)
            
            print(f"Top Topics: {top_topics.indices.tolist()} (scores: {top_topics.values.tolist()})")
            
            if results['generated_stories']:
                story = results['generated_stories'][0]
                print(f"Generated Story Length: {story['story_length']} tokens")
                print(f"Skeleton Words: {len(story['skeleton_words'])} words")
                
            print(f"News Category: {torch.argmax(results['news_categories'][0]).item()}")
            print(f"Topic Coherence: {results['topic_coherence'][0].item():.4f}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED")
    print("="*60)
    print("Key Features Demonstrated:")
    print("✓ Semantic graph-guided topic discovery")
    print("✓ Hierarchical story generation from topics")
    print("✓ Cross-modal topic-story alignment")
    print("✓ Multi-task news analysis")
    print("✓ Real-time topic coherence assessment")

def run_dashboard_mode(args):
    """Run interactive dashboard"""
    logger.info("Starting TopNet-TopicNet integration dashboard...")
    
    # Load data
    data = load_sample_data(args.data_path, num_samples=200)
    
    # Create model
    model = create_demo_model(args)
    
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading pre-trained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create enhanced dashboard with TopNet features
    class TopNetDashboard(DashboardApp):
        def __init__(self, data, model, device, **kwargs):
            super().__init__(data, **kwargs)
            self.model = model
            self.device = device
            self.processor = TextProcessor()
        
        def analyze_article_with_topnet(self, article_text):
            """Analyze article using TopNet integration"""
            # Process article
            processed = self.processor.process_text(article_text)
            
            # Simple tokenization
            tokens = [hash(word) % 10000 for word in processed.split()[:512]]
            if len(tokens) < 512:
                tokens.extend([0] * (512 - len(tokens)))
            
            article_tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                results = self.model.analyze_news_content(
                    article_tokens,
                    generate_stories=True,
                    story_length=200
                )
            
            return results
    
    # Initialize dashboard with TopNet integration
    dashboard = TopNetDashboard(
        data=data,
        model=model,
        device=args.device,
        embeddings_2d=None,  # Will generate if needed
        cluster_labels=data.get('cluster', [0] * len(data)).tolist(),
        texts=data['content'].tolist()
    )
    
    print("\n" + "="*60)
    print("TOPNET-TOPICNET INTEGRATION DASHBOARD")
    print("="*60)
    print(f"Dashboard starting on http://localhost:{args.port}")
    print("Features available:")
    print("• Interactive topic visualization")
    print("• Real-time story generation")
    print("• Cross-modal analysis")
    print("• News categorization")
    print("• Topic coherence metrics")
    print("="*60)
    
    # Start dashboard
    dashboard.run_server(debug=True, port=args.port)

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup
    args.device = setup_device(args.device)
    
    # Create results directory
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Run based on mode
    try:
        if args.mode == 'experiment':
            results = run_experiment_mode(args)
        elif args.mode == 'train':
            trainer, evaluation = run_training_mode(args)
        elif args.mode == 'demo':
            run_demo_mode(args)
        elif args.mode == 'dashboard':
            run_dashboard_mode(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return
        
        logger.info("TopNet-TopicNet integration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in TopNet integration: {str(e)}")
        raise

if __name__ == "__main__":
    main() 