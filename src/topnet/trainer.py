import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some visualizations may be limited.")
from sklearn.metrics import classification_report
import nltk
from nltk.translate.bleu_score import corpus_bleu
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available. ROUGE metrics will be disabled.")

from .integrated_framework import IntegratedTopicNetTopNet
from processing.text_processor import TextProcessor

class NewsDataset(Dataset):
    """Dataset for news articles with optional targets"""
    
    def __init__(
        self, 
        articles: List[str],
        categories: Optional[List[int]] = None,
        tokenizer: Optional[object] = None,
        max_length: int = 512
    ):
        self.articles = articles
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Initialize text processor if no tokenizer provided
        if tokenizer is None:
            self.processor = TextProcessor()
        
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = self.articles[idx]
        
        # Tokenize article
        if self.tokenizer:
            tokens = self.tokenizer.encode(article, max_length=self.max_length, truncation=True)
        else:
            # Use simple tokenization
            tokens = self.processor.tokenize(article)[:self.max_length]
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        
        result = {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'article_text': article
        }
        
        if self.categories is not None:
            result['category'] = torch.tensor(self.categories[idx], dtype=torch.long)
        
        return result

class TopicNetTopNetTrainer:
    """
    Comprehensive trainer for the integrated TopicNet-TopNet framework
    Includes multi-task learning, evaluation, and visualization
    """
    
    def __init__(
        self,
        model: IntegratedTopicNetTopNet,
        train_dataset: NewsDataset,
        val_dataset: Optional[NewsDataset] = None,
        test_dataset: Optional[NewsDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_wandb: bool = True,
        project_name: str = "TopicNet-TopNet-Integration"
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.use_wandb = use_wandb
        
        # Datasets and dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Evaluation metrics
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
        
        # Initialize wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=project_name,
                config={
                    'model': 'TopicNet-TopNet-Integrated',
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'vocab_size': model.vocab_size,
                    'num_topics': model.num_topics
                }
            )
        elif use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Continuing without wandb logging.")
            self.use_wandb = False
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'topic_coherence': [],
            'classification_accuracy': [],
            'story_quality': []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'topic_elbo': 0.0,
            'classification': 0.0,
            'hierarchy': 0.0,
            'story_generation': 0.0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch}",
            leave=False
        )
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Move to device
            article_tokens = batch['tokens'].to(self.device)
            target_categories = batch.get('category', None)
            if target_categories is not None:
                target_categories = target_categories.to(self.device)
            
            # Compute losses
            losses = self.model.compute_comprehensive_loss(
                article_tokens=article_tokens,
                target_categories=target_categories
            )
            
            # Backward pass
            total_loss = losses['total']
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update running losses
            for key, value in losses.items():
                if key in epoch_losses:
                    epoch_losses[key] += value.item()
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Scheduler step
        self.scheduler.step()
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validation step"""
        if not self.val_dataset:
            return {}
        
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'topic_elbo': 0.0,
            'classification': 0.0,
            'hierarchy': 0.0
        }
        
        topic_coherences = []
        classification_predictions = []
        classification_targets = []
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                article_tokens = batch['tokens'].to(self.device)
                target_categories = batch.get('category', None)
                if target_categories is not None:
                    target_categories = target_categories.to(self.device)
                
                # Compute losses
                losses = self.model.compute_comprehensive_loss(
                    article_tokens=article_tokens,
                    target_categories=target_categories
                )
                
                # Update running losses
                for key, value in losses.items():
                    if key in val_losses:
                        val_losses[key] += value.item()
                
                # Get model predictions for evaluation
                results = self.model.analyze_news_content(
                    article_tokens, 
                    generate_stories=False
                )
                
                # Topic coherence evaluation
                topic_coherence = self._evaluate_topic_coherence(
                    results['topic_analysis']['topic_distributions']
                )
                topic_coherences.append(topic_coherence)
                
                # Classification evaluation
                if target_categories is not None:
                    predictions = torch.argmax(results['news_categories'], dim=-1)
                    classification_predictions.extend(predictions.cpu().numpy())
                    classification_targets.extend(target_categories.cpu().numpy())
                
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Compute evaluation metrics
        val_metrics = {
            'val_loss': val_losses['total'],
            'val_topic_elbo': val_losses['topic_elbo'],
            'val_classification': val_losses['classification'],
            'val_hierarchy': val_losses['hierarchy'],
            'topic_coherence': np.mean(topic_coherences)
        }
        
        if classification_predictions:
            from sklearn.metrics import accuracy_score
            val_metrics['classification_accuracy'] = accuracy_score(
                classification_targets, 
                classification_predictions
            )
        
        return val_metrics
    
    def _evaluate_topic_coherence(self, topic_distributions: torch.Tensor) -> float:
        """Evaluate topic coherence using simplified metric"""
        # Simplified coherence: entropy of topic distributions
        # Lower entropy indicates more coherent topics
        entropy = -torch.sum(
            topic_distributions * torch.log(topic_distributions + 1e-10),
            dim=-1
        )
        return entropy.mean().item()
    
    def evaluate_story_generation(
        self, 
        num_samples: int = 100,
        reference_stories: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate story generation quality"""
        self.model.eval()
        
        generated_stories = []
        story_qualities = []
        
        with torch.no_grad():
            # Sample random articles for story generation
            sample_indices = np.random.choice(
                len(self.test_dataset or self.val_dataset), 
                size=min(num_samples, len(self.test_dataset or self.val_dataset)),
                replace=False
            )
            
            for idx in sample_indices:
                sample = (self.test_dataset or self.val_dataset)[idx]
                article_tokens = sample['tokens'].unsqueeze(0).to(self.device)
                
                # Generate story
                results = self.model.analyze_news_content(
                    article_tokens,
                    generate_stories=True,
                    story_length=200
                )
                
                if results['generated_stories']:
                    story = results['generated_stories'][0]
                    generated_stories.append(story)
                    
                    # Evaluate story quality (simplified)
                    quality_score = self._evaluate_story_quality(story)
                    story_qualities.append(quality_score)
        
        metrics = {
            'avg_story_quality': np.mean(story_qualities) if story_qualities else 0.0,
            'num_generated_stories': len(generated_stories)
        }
        
        # BLEU score evaluation if reference stories provided
        if reference_stories and generated_stories:
            bleu_scores = self._compute_bleu_scores(
                generated_stories, 
                reference_stories[:len(generated_stories)]
            )
            metrics.update(bleu_scores)
        
        return metrics
    
    def _evaluate_story_quality(self, story: Dict) -> float:
        """Evaluate individual story quality"""
        # Simplified quality metric based on story length and diversity
        story_tokens = story.get('story_tokens', [])
        
        if not story_tokens:
            return 0.0
        
        # Length penalty (stories too short or too long get penalized)
        length_score = min(1.0, len(story_tokens) / 200)
        
        # Diversity score (unique tokens ratio)
        diversity_score = len(set(story_tokens)) / len(story_tokens)
        
        # Coherence score (simplified - based on skeleton word utilization)
        skeleton_words = story.get('skeleton_words', [])
        coherence_score = len(set(skeleton_words)) / len(skeleton_words) if skeleton_words else 0.0
        
        return (length_score + diversity_score + coherence_score) / 3
    
    def _compute_bleu_scores(
        self, 
        generated_stories: List[Dict], 
        reference_stories: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU scores for generated stories"""
        candidates = []
        references = []
        
        for story_dict, ref_story in zip(generated_stories, reference_stories):
            # Convert token ids to text (simplified)
            candidate_tokens = story_dict.get('story_tokens', [])
            candidate_text = ' '.join(map(str, candidate_tokens))  # Simplified
            
            candidates.append(candidate_text.split())
            references.append([ref_story.split()])
        
        if not candidates or not references:
            return {'bleu_score': 0.0}
        
        try:
            bleu_score = corpus_bleu(references, candidates)
            return {'bleu_score': bleu_score}
        except:
            return {'bleu_score': 0.0}
    
    def train(
        self, 
        num_epochs: int = 50,
        eval_every: int = 5,
        save_every: int = 10,
        save_path: str = 'checkpoints'
    ):
        """Main training loop"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            self.logger.info(
                f"Epoch {epoch}: Train Loss = {train_metrics['total']:.4f}"
            )
            
            # Validation
            if self.val_dataset and (epoch % eval_every == 0):
                val_metrics = self.validate()
                
                self.logger.info(
                    f"Epoch {epoch}: Val Loss = {val_metrics.get('val_loss', 0):.4f}, "
                    f"Topic Coherence = {val_metrics.get('topic_coherence', 0):.4f}"
                )
                
                # Save best model
                if val_metrics.get('val_loss', float('inf')) < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.save_model(f"{save_path}/best_model.pt")
                
                # Log to wandb
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'epoch': epoch,
                        **train_metrics,
                        **val_metrics
                    })
                
                # Update history
                self.train_history['epoch'].append(epoch)
                self.train_history['train_loss'].append(train_metrics['total'])
                self.train_history['val_loss'].append(val_metrics.get('val_loss', 0))
                self.train_history['topic_coherence'].append(val_metrics.get('topic_coherence', 0))
                self.train_history['classification_accuracy'].append(
                    val_metrics.get('classification_accuracy', 0)
                )
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_model(f"{save_path}/checkpoint_epoch_{epoch}.pt")
        
        # Final evaluation
        if self.test_dataset:
            self.logger.info("Running final evaluation...")
            test_metrics = self.evaluate_story_generation()
            
            self.logger.info(f"Final Test Metrics: {test_metrics}")
            
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log(test_metrics)
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        
        self.logger.info(f"Model loaded from {path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_history['epoch'], self.train_history['train_loss'], label='Train')
        axes[0, 0].plot(self.train_history['epoch'], self.train_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Topic coherence
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['topic_coherence'])
        axes[0, 1].set_title('Topic Coherence')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Coherence Score')
        
        # Classification accuracy
        axes[1, 0].plot(self.train_history['epoch'], self.train_history['classification_accuracy'])
        axes[1, 0].set_title('Classification Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        
        # Story quality
        if self.train_history['story_quality']:
            axes[1, 1].plot(self.train_history['epoch'], self.train_history['story_quality'])
            axes[1, 1].set_title('Story Generation Quality')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Quality Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self) -> Dict[str, Union[float, List, Dict]]:
        """Generate comprehensive evaluation report"""
        self.logger.info("Generating evaluation report...")
        
        # Topic analysis evaluation
        topic_metrics = self._evaluate_topic_discovery()
        
        # Story generation evaluation
        story_metrics = self.evaluate_story_generation()
        
        # Cross-modal evaluation
        cross_modal_metrics = self._evaluate_cross_modal_alignment()
        
        report = {
            'topic_discovery': topic_metrics,
            'story_generation': story_metrics,
            'cross_modal_alignment': cross_modal_metrics,
            'training_summary': {
                'final_train_loss': self.train_history['train_loss'][-1] if self.train_history['train_loss'] else 0,
                'final_val_loss': self.train_history['val_loss'][-1] if self.train_history['val_loss'] else 0,
                'best_topic_coherence': max(self.train_history['topic_coherence']) if self.train_history['topic_coherence'] else 0,
                'best_classification_accuracy': max(self.train_history['classification_accuracy']) if self.train_history['classification_accuracy'] else 0
            }
        }
        
        return report
    
    def _evaluate_topic_discovery(self) -> Dict[str, float]:
        """Evaluate topic discovery performance"""
        # Implement topic discovery evaluation metrics
        # This could include perplexity, topic coherence, etc.
        return {
            'topic_diversity': 0.85,  # Placeholder
            'topic_coherence': 0.75,  # Placeholder
            'semantic_consistency': 0.80  # Placeholder
        }
    
    def _evaluate_cross_modal_alignment(self) -> Dict[str, float]:
        """Evaluate cross-modal alignment between topics and stories"""
        # Implement cross-modal evaluation
        return {
            'topic_story_alignment': 0.78,  # Placeholder
            'narrative_coherence': 0.82,  # Placeholder
            'semantic_preservation': 0.76  # Placeholder
        } 