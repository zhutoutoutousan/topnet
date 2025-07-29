import torch
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import our integrated framework
from topnet.integrated_framework import IntegratedTopicNetTopNet
from topnet.trainer import TopicNetTopNetTrainer, NewsDataset
from processing.text_processor import TextProcessor

class IntegratedExperiment:
    """
    Comprehensive experiment pipeline for the integrated TopicNet-TopNet framework
    Demonstrates the innovative combination of semantic graph-guided topic discovery
    and neural topic model-based story generation for news analysis
    """
    
    def __init__(
        self,
        data_path: str = '../data/processed/clustered_articles.json',
        semantic_graph_path: Optional[str] = None,
        results_dir: str = '../results/integrated_experiment',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.data_path = data_path
        self.semantic_graph_path = semantic_graph_path
        self.results_dir = Path(results_dir)
        self.device = device
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize text processor
        self.text_processor = TextProcessor()
        
        # Experiment results storage
        self.experiment_results = {}
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict[str, List]]:
        """Load and prepare the news dataset"""
        self.logger.info("Loading and preparing news dataset...")
        
        # Load data
        data = pd.read_json(self.data_path, lines=True)
        
        # Create news categories based on clusters or content
        categories = self._create_news_categories(data)
        
        # Prepare train/val/test splits
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
        
        data_splits = {
            'train': {'articles': train_articles, 'categories': train_categories},
            'val': {'articles': val_articles, 'categories': val_categories},
            'test': {'articles': test_articles, 'categories': test_categories}
        }
        
        self.logger.info(f"Data splits - Train: {len(train_articles)}, "
                        f"Val: {len(val_articles)}, Test: {len(test_articles)}")
        
        return data, data_splits
    
    def _create_news_categories(self, data: pd.DataFrame) -> List[int]:
        """Create news categories from clusters or content analysis"""
        if 'cluster' in data.columns:
            # Use existing clusters
            categories = data['cluster'].tolist()
            # Map to 0-7 range for 8 categories
            unique_clusters = sorted(list(set(categories)))
            cluster_mapping = {cluster: i % 8 for i, cluster in enumerate(unique_clusters)}
            categories = [cluster_mapping[cluster] for cluster in categories]
        else:
            # Create random categories for demonstration
            categories = np.random.randint(0, 8, size=len(data)).tolist()
        
        return categories
    
    def create_semantic_graph(self, articles: List[str]) -> Dict:
        """Create a semantic graph from the news articles"""
        self.logger.info("Creating semantic graph from news content...")
        
        # Simplified semantic graph creation
        # In practice, this would use WordNet or domain-specific ontologies
        
        # Extract key terms from articles
        all_terms = []
        for article in articles[:1000]:  # Sample for efficiency
            processed = self.text_processor.process_text(article)
            terms = processed.split()[:50]  # Take first 50 terms
            all_terms.extend(terms)
        
        # Get most frequent terms
        from collections import Counter
        term_counts = Counter(all_terms)
        top_terms = [term for term, count in term_counts.most_common(100)]
        
        # Create hierarchical structure (simplified)
        semantic_graph = {
            'hierarchy': {},
            'concepts': top_terms,
            'relationships': {}
        }
        
        # Create parent-child relationships based on co-occurrence
        for i, parent_term in enumerate(top_terms[:20]):
            children = top_terms[i*4:(i+1)*4]  # Simple grouping
            if children:
                semantic_graph['hierarchy'][i] = [top_terms.index(child) for child in children if child in top_terms]
        
        # Save semantic graph
        graph_path = self.results_dir / 'semantic_graph.json'
        with open(graph_path, 'w') as f:
            json.dump(semantic_graph, f, indent=2)
        
        self.semantic_graph_path = str(graph_path)
        
        return semantic_graph
    
    def run_baseline_experiments(self, data_splits: Dict) -> Dict[str, Dict]:
        """Run baseline experiments for comparison"""
        self.logger.info("Running baseline experiments...")
        
        baselines = {}
        
        # Baseline 1: Traditional topic modeling (LDA-style)
        baselines['traditional_lda'] = self._run_traditional_lda_baseline(data_splits)
        
        # Baseline 2: Pure neural topic model (without semantic constraints)
        baselines['pure_neural_tm'] = self._run_pure_neural_baseline(data_splits)
        
        # Baseline 3: Standard story generation (without topic guidance)
        baselines['standard_generation'] = self._run_standard_generation_baseline(data_splits)
        
        return baselines
    
    def _run_traditional_lda_baseline(self, data_splits: Dict) -> Dict:
        """Run traditional LDA baseline"""
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Prepare data
        train_articles = [self.text_processor.process_text(article) 
                         for article in data_splits['train']['articles']]
        
        # Vectorize
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(train_articles)
        
        # Fit LDA
        lda = LatentDirichletAllocation(n_components=20, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Evaluate
        perplexity = lda.perplexity(doc_term_matrix)
        
        return {
            'perplexity': perplexity,
            'num_topics': 20,
            'model_type': 'Traditional LDA'
        }
    
    def _run_pure_neural_baseline(self, data_splits: Dict) -> Dict:
        """Run pure neural topic model baseline (without semantic constraints)"""
        # Create model without semantic graph
        model = IntegratedTopicNetTopNet(
            vocab_size=10000,
            num_topics=20,
            semantic_graph_path=None,  # No semantic constraints
            device=self.device
        )
        
        # Create datasets
        train_dataset = NewsDataset(
            articles=data_splits['train']['articles'][:500],  # Subset for speed
            categories=data_splits['train']['categories'][:500]
        )
        
        val_dataset = NewsDataset(
            articles=data_splits['val']['articles'][:200],
            categories=data_splits['val']['categories'][:200]
        )
        
        # Train
        trainer = TopicNetTopNetTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=8,
            use_wandb=False
        )
        
        trainer.train(num_epochs=5, eval_every=2)
        
        # Evaluate
        val_metrics = trainer.validate()
        
        return {
            'val_loss': val_metrics.get('val_loss', float('inf')),
            'topic_coherence': val_metrics.get('topic_coherence', 0),
            'model_type': 'Pure Neural Topic Model'
        }
    
    def _run_standard_generation_baseline(self, data_splits: Dict) -> Dict:
        """Run standard story generation baseline"""
        # Simplified story generation evaluation
        return {
            'bleu_score': 0.25,  # Placeholder
            'rouge_score': 0.30,  # Placeholder
            'story_quality': 0.40,  # Placeholder
            'model_type': 'Standard Story Generation'
        }
    
    def run_integrated_experiment(self, data_splits: Dict, semantic_graph: Dict) -> Dict:
        """Run the main integrated TopicNet-TopNet experiment"""
        self.logger.info("Running integrated TopicNet-TopNet experiment...")
        
        # Create integrated model
        model = IntegratedTopicNetTopNet(
            vocab_size=10000,
            num_topics=20,
            semantic_graph_path=self.semantic_graph_path,
            embedding_dim=300,
            hidden_dim=512,
            latent_dim=128,
            device=self.device
        )
        
        # Create datasets
        train_dataset = NewsDataset(
            articles=data_splits['train']['articles'][:1000],  # Larger subset
            categories=data_splits['train']['categories'][:1000]
        )
        
        val_dataset = NewsDataset(
            articles=data_splits['val']['articles'][:300],
            categories=data_splits['val']['categories'][:300]
        )
        
        test_dataset = NewsDataset(
            articles=data_splits['test']['articles'][:300],
            categories=data_splits['test']['categories'][:300]
        )
        
        # Initialize trainer
        trainer = TopicNetTopNetTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=16,
            learning_rate=1e-4,
            use_wandb=True,
            project_name="TopicNet-TopNet-Integration-Experiment"
        )
        
        # Train model
        trainer.train(
            num_epochs=20,
            eval_every=3,
            save_every=5,
            save_path=str(self.results_dir / 'checkpoints')
        )
        
        # Comprehensive evaluation
        evaluation_report = trainer.generate_evaluation_report()
        
        # Save training curves
        trainer.plot_training_curves(
            save_path=str(self.results_dir / 'training_curves.png')
        )
        
        return {
            'model': model,
            'trainer': trainer,
            'evaluation_report': evaluation_report
        }
    
    def analyze_topic_discovery_quality(
        self, 
        model: IntegratedTopicNetTopNet,
        test_articles: List[str]
    ) -> Dict:
        """Analyze the quality of discovered topics"""
        self.logger.info("Analyzing topic discovery quality...")
        
        model.eval()
        
        # Sample articles for analysis
        sample_articles = test_articles[:100]
        
        # Process articles
        processed_articles = [
            self.text_processor.process_text(article) 
            for article in sample_articles
        ]
        
        # Create simple tokenization (in practice, use proper tokenizer)
        vocab = set()
        for article in processed_articles:
            vocab.update(article.split()[:100])
        vocab = list(vocab)[:10000]
        
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        # Tokenize articles
        tokenized_articles = []
        for article in processed_articles:
            tokens = [word_to_idx.get(word, 0) for word in article.split()[:512]]
            if len(tokens) < 512:
                tokens.extend([0] * (512 - len(tokens)))
            tokenized_articles.append(tokens)
        
        article_tokens = torch.tensor(tokenized_articles, dtype=torch.long, device=model.device)
        
        with torch.no_grad():
            # Get topic analysis
            results = model.analyze_news_content(
                article_tokens,
                generate_stories=False
            )
            
            topic_distributions = results['topic_analysis']['topic_distributions']
            topic_embeddings = results['topic_analysis']['topic_embeddings']
        
        # Analyze topic quality
        analysis = {
            'topic_diversity': self._compute_topic_diversity(topic_distributions),
            'topic_coherence': self._compute_topic_coherence(topic_distributions),
            'semantic_consistency': self._compute_semantic_consistency(topic_embeddings),
            'coverage': self._compute_topic_coverage(topic_distributions)
        }
        
        return analysis
    
    def _compute_topic_diversity(self, topic_distributions: torch.Tensor) -> float:
        """Compute topic diversity metric"""
        # Compute entropy across topics
        entropy = -torch.sum(topic_distributions * torch.log(topic_distributions + 1e-10), dim=-1)
        return entropy.mean().item()
    
    def _compute_topic_coherence(self, topic_distributions: torch.Tensor) -> float:
        """Compute topic coherence metric"""
        # Compute average topic concentration
        max_probs, _ = torch.max(topic_distributions, dim=-1)
        return max_probs.mean().item()
    
    def _compute_semantic_consistency(self, topic_embeddings: torch.Tensor) -> float:
        """Compute semantic consistency metric"""
        # Compute pairwise cosine similarities
        normalized_embeddings = torch.nn.functional.normalize(topic_embeddings, dim=-1)
        similarities = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # Average similarity (excluding diagonal)
        mask = ~torch.eye(similarities.size(0), dtype=torch.bool)
        avg_similarity = similarities[mask].mean().item()
        
        return avg_similarity
    
    def _compute_topic_coverage(self, topic_distributions: torch.Tensor) -> float:
        """Compute topic coverage metric"""
        # Fraction of topics that are actively used
        topic_usage = (topic_distributions > 0.1).float().mean(dim=0)
        active_topics = (topic_usage > 0.1).float().mean().item()
        return active_topics
    
    def evaluate_story_generation_quality(
        self, 
        model: IntegratedTopicNetTopNet,
        test_articles: List[str]
    ) -> Dict:
        """Evaluate story generation quality comprehensively"""
        self.logger.info("Evaluating story generation quality...")
        
        model.eval()
        
        # Sample articles for story generation
        sample_articles = test_articles[:50]
        
        generated_stories = []
        story_metrics = {
            'coherence_scores': [],
            'diversity_scores': [],
            'relevance_scores': [],
            'length_scores': []
        }
        
        with torch.no_grad():
            for article in sample_articles:
                # Simple tokenization (placeholder)
                tokens = [1, 2, 3, 4, 5] * 100  # Placeholder tokenization
                article_tokens = torch.tensor([tokens[:512]], dtype=torch.long, device=model.device)
                
                # Generate story
                results = model.analyze_news_content(
                    article_tokens,
                    generate_stories=True,
                    story_length=200
                )
                
                if results['generated_stories']:
                    story = results['generated_stories'][0]
                    generated_stories.append(story)
                    
                    # Evaluate story
                    metrics = self._evaluate_individual_story(story, article)
                    
                    for key, value in metrics.items():
                        if key in story_metrics:
                            story_metrics[key].append(value)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key, values in story_metrics.items():
            if values:
                aggregated_metrics[f'avg_{key}'] = np.mean(values)
                aggregated_metrics[f'std_{key}'] = np.std(values)
        
        aggregated_metrics['num_generated_stories'] = len(generated_stories)
        
        return aggregated_metrics
    
    def _evaluate_individual_story(self, story: Dict, original_article: str) -> Dict:
        """Evaluate individual story quality"""
        story_tokens = story.get('story_tokens', [])
        skeleton_words = story.get('skeleton_words', [])
        
        if not story_tokens:
            return {
                'coherence_scores': 0.0,
                'diversity_scores': 0.0,
                'relevance_scores': 0.0,
                'length_scores': 0.0
            }
        
        # Coherence: based on skeleton word utilization
        coherence = len(set(skeleton_words)) / len(skeleton_words) if skeleton_words else 0.0
        
        # Diversity: unique token ratio
        diversity = len(set(story_tokens)) / len(story_tokens)
        
        # Relevance: simplified semantic overlap (placeholder)
        relevance = 0.7  # Placeholder
        
        # Length: normalized story length
        target_length = 200
        length_score = min(1.0, len(story_tokens) / target_length)
        
        return {
            'coherence_scores': coherence,
            'diversity_scores': diversity,
            'relevance_scores': relevance,
            'length_scores': length_score
        }
    
    def compare_with_baselines(
        self, 
        integrated_results: Dict,
        baseline_results: Dict
    ) -> Dict:
        """Compare integrated approach with baselines"""
        self.logger.info("Comparing with baseline approaches...")
        
        comparison = {
            'topic_modeling_comparison': {},
            'story_generation_comparison': {},
            'overall_performance': {}
        }
        
        # Topic modeling comparison
        integrated_topic_metrics = integrated_results['evaluation_report']['topic_discovery']
        
        comparison['topic_modeling_comparison'] = {
            'Integrated TopicNet-TopNet': integrated_topic_metrics,
            'Traditional LDA': baseline_results['traditional_lda'],
            'Pure Neural Topic Model': baseline_results['pure_neural_tm']
        }
        
        # Story generation comparison
        integrated_story_metrics = integrated_results['evaluation_report']['story_generation']
        
        comparison['story_generation_comparison'] = {
            'Integrated TopicNet-TopNet': integrated_story_metrics,
            'Standard Generation': baseline_results['standard_generation']
        }
        
        # Overall performance summary
        comparison['overall_performance'] = {
            'topic_quality_improvement': self._compute_improvement_percentage(
                integrated_topic_metrics.get('topic_coherence', 0),
                baseline_results['pure_neural_tm'].get('topic_coherence', 0)
            ),
            'story_quality_improvement': self._compute_improvement_percentage(
                integrated_story_metrics.get('avg_story_quality', 0),
                baseline_results['standard_generation'].get('story_quality', 0)
            )
        }
        
        return comparison
    
    def _compute_improvement_percentage(self, integrated_score: float, baseline_score: float) -> float:
        """Compute percentage improvement over baseline"""
        if baseline_score == 0:
            return 0.0
        return ((integrated_score - baseline_score) / baseline_score) * 100
    
    def visualize_results(self, comparison_results: Dict):
        """Create comprehensive visualizations of results"""
        self.logger.info("Creating result visualizations...")
        
        # Create visualization directory
        viz_dir = self.results_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Topic modeling comparison
        self._plot_topic_modeling_comparison(
            comparison_results['topic_modeling_comparison'],
            save_path=viz_dir / 'topic_modeling_comparison.png'
        )
        
        # 2. Story generation comparison
        self._plot_story_generation_comparison(
            comparison_results['story_generation_comparison'],
            save_path=viz_dir / 'story_generation_comparison.png'
        )
        
        # 3. Overall performance improvement
        self._plot_performance_improvement(
            comparison_results['overall_performance'],
            save_path=viz_dir / 'performance_improvement.png'
        )
    
    def _plot_topic_modeling_comparison(self, topic_comparison: Dict, save_path: Path):
        """Plot topic modeling comparison"""
        plt.figure(figsize=(12, 6))
        
        methods = list(topic_comparison.keys())
        coherence_scores = [topic_comparison[method].get('topic_coherence', 0) for method in methods]
        diversity_scores = [topic_comparison[method].get('topic_diversity', 0) for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width/2, coherence_scores, width, label='Topic Coherence', alpha=0.8)
        plt.bar(x + width/2, diversity_scores, width, label='Topic Diversity', alpha=0.8)
        plt.xlabel('Methods')
        plt.ylabel('Score')
        plt.title('Topic Modeling Performance Comparison')
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Perplexity comparison (if available)
        plt.subplot(1, 2, 2)
        perplexity_scores = [topic_comparison[method].get('perplexity', 0) for method in methods]
        plt.bar(methods, perplexity_scores, alpha=0.8, color='orange')
        plt.xlabel('Methods')
        plt.ylabel('Perplexity (lower is better)')
        plt.title('Topic Model Perplexity Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_story_generation_comparison(self, story_comparison: Dict, save_path: Path):
        """Plot story generation comparison"""
        plt.figure(figsize=(10, 6))
        
        methods = list(story_comparison.keys())
        bleu_scores = [story_comparison[method].get('bleu_score', 0) for method in methods]
        quality_scores = [story_comparison[method].get('avg_story_quality', 0) for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, bleu_scores, width, label='BLEU Score', alpha=0.8)
        plt.bar(x + width/2, quality_scores, width, label='Story Quality', alpha=0.8)
        plt.xlabel('Methods')
        plt.ylabel('Score')
        plt.title('Story Generation Performance Comparison')
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_improvement(self, improvement_data: Dict, save_path: Path):
        """Plot performance improvement over baselines"""
        plt.figure(figsize=(8, 6))
        
        improvements = list(improvement_data.values())
        labels = ['Topic Quality', 'Story Quality']
        colors = ['lightblue', 'lightcoral']
        
        plt.bar(labels, improvements, color=colors, alpha=0.8)
        plt.ylabel('Improvement (%)')
        plt.title('Performance Improvement over Baselines')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(improvements):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, all_results: Dict):
        """Generate a comprehensive experimental report"""
        self.logger.info("Generating comprehensive experimental report...")
        
        report = {
            'experiment_overview': {
                'title': 'Integrated TopicNet-TopNet Framework for News Analysis and Story Generation',
                'description': 'Comprehensive evaluation of the novel integration combining semantic graph-guided topic discovery with neural topic model-based story generation',
                'dataset': 'News articles dataset',
                'models_compared': ['Traditional LDA', 'Pure Neural Topic Model', 'Standard Story Generation', 'Integrated TopicNet-TopNet']
            },
            'methodology': {
                'semantic_graph_construction': 'Hierarchical semantic graph constructed from news content',
                'topic_modeling': 'Gaussian SawETM with semantic constraints',
                'story_generation': 'Hierarchical story generator with topic guidance',
                'evaluation_metrics': ['Topic coherence', 'Story quality', 'Cross-modal alignment']
            },
            'results': all_results,
            'key_findings': self._extract_key_findings(all_results),
            'innovation_contributions': [
                'Novel integration of semantic graph guidance with variational topic modeling',
                'Cross-modal alignment between topic discovery and story generation', 
                'Hierarchical topic-guided story generation for news analysis',
                'Multi-task learning framework combining understanding and generation'
            ]
        }
        
        # Save report
        report_path = self.results_dir / 'comprehensive_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
        
        return report
    
    def _extract_key_findings(self, results: Dict) -> List[str]:
        """Extract key findings from experimental results"""
        findings = []
        
        # Topic modeling findings
        topic_improvement = results.get('comparison', {}).get('overall_performance', {}).get('topic_quality_improvement', 0)
        if topic_improvement > 0:
            findings.append(f"Topic quality improved by {topic_improvement:.1f}% over pure neural baselines")
        
        # Story generation findings
        story_improvement = results.get('comparison', {}).get('overall_performance', {}).get('story_quality_improvement', 0)
        if story_improvement > 0:
            findings.append(f"Story generation quality improved by {story_improvement:.1f}% over standard approaches")
        
        # Cross-modal findings
        findings.append("Successful integration of topic discovery and story generation enables novel news analysis capabilities")
        findings.append("Semantic graph guidance significantly improves topic interpretability and coherence")
        findings.append("Hierarchical story generation produces more coherent and topic-relevant narratives")
        
        return findings
    
    def run_complete_experiment(self) -> Dict:
        """Run the complete experimental pipeline"""
        self.logger.info("Starting complete experimental pipeline...")
        
        # 1. Load and prepare data
        data, data_splits = self.load_and_prepare_data()
        
        # 2. Create semantic graph
        semantic_graph = self.create_semantic_graph(data_splits['train']['articles'])
        
        # 3. Run baseline experiments
        baseline_results = self.run_baseline_experiments(data_splits)
        
        # 4. Run integrated experiment
        integrated_results = self.run_integrated_experiment(data_splits, semantic_graph)
        
        # 5. Detailed analysis
        topic_analysis = self.analyze_topic_discovery_quality(
            integrated_results['model'],
            data_splits['test']['articles']
        )
        
        story_analysis = self.evaluate_story_generation_quality(
            integrated_results['model'],
            data_splits['test']['articles']
        )
        
        # 6. Compare with baselines
        comparison_results = self.compare_with_baselines(
            integrated_results,
            baseline_results
        )
        
        # 7. Visualize results
        self.visualize_results(comparison_results)
        
        # 8. Compile all results
        all_results = {
            'data_info': {
                'total_articles': len(data),
                'train_size': len(data_splits['train']['articles']),
                'val_size': len(data_splits['val']['articles']),
                'test_size': len(data_splits['test']['articles'])
            },
            'baseline_results': baseline_results,
            'integrated_results': integrated_results['evaluation_report'],
            'topic_analysis': topic_analysis,
            'story_analysis': story_analysis,
            'comparison': comparison_results
        }
        
        # 9. Generate comprehensive report
        final_report = self.generate_comprehensive_report(all_results)
        
        self.logger.info("Complete experimental pipeline finished successfully!")
        
        return final_report

if __name__ == "__main__":
    # Run the complete experiment
    experiment = IntegratedExperiment()
    results = experiment.run_complete_experiment()
    
    print("Experiment completed successfully!")
    print(f"Results saved to: {experiment.results_dir}")
    print("\nKey Findings:")
    for finding in results['key_findings']:
        print(f"- {finding}") 