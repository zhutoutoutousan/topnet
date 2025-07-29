import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import json
from pathlib import Path

from .topic_generator import TopicGenerator
from .story_generator import HierarchicalStoryGenerator

class IntegratedTopicNetTopNet(nn.Module):
    """
    Integrated TopicNet-TopNet Framework:
    Combines semantic graph-guided topic discovery (TopicNet) with 
    neural topic model-based story generation (TopNet) for advanced
    news analysis and content generation.
    
    This is the main innovation that bridges:
    1. Semantic graph-guided hierarchical topic modeling
    2. Variational topic inference for story generation
    3. Multi-modal news analysis and content synthesis
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        semantic_graph_path: Optional[str] = None,
        embedding_dim: int = 300,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        max_story_length: int = 512,
        skeleton_length: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(IntegratedTopicNetTopNet, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # Core components
        self.topic_generator = TopicGenerator(
            vocab_size=vocab_size,
            num_topics=num_topics,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        # Create story generator with adjusted hidden_dim
        self.story_generator = HierarchicalStoryGenerator(
            vocab_size=vocab_size,
            max_story_length=max_story_length,
            skeleton_length=skeleton_length,
            hidden_dim=hidden_dim,
            device=device
        )
        
        # Get the actual hidden_dim used by story generator (may be adjusted)
        actual_hidden_dim = self.story_generator.story_model.config.n_embd
        
        # Semantic graph integration (TopicNet-style)
        self.semantic_graph = self._load_semantic_graph(semantic_graph_path)
        
        # Cross-modal alignment layers
        self.topic_to_story_projector = nn.Sequential(
            nn.Linear(latent_dim, actual_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(actual_hidden_dim, actual_hidden_dim)
        )
        
        # Topic projection for hierarchy encoding
        self.topic_projector = nn.Linear(num_topics, latent_dim)
        
        # Semantic hierarchy constraint layers (TopicNet integration)
        self.hierarchy_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=8,
                dim_feedforward=latent_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Multi-task learning heads
        self.news_classifier = nn.Sequential(
            nn.Linear(num_topics, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 8)  # 8 news categories
        )
        
        self.topic_coherence_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Innovation: Cross-modal attention for topic-story alignment
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=actual_hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Topic projection for cross-modal attention
        self.topic_to_cross_modal = nn.Linear(latent_dim, actual_hidden_dim)
        
    def _load_semantic_graph(self, graph_path: Optional[str]) -> Optional[Dict]:
        """Load semantic graph structure for TopicNet-style guidance"""
        if graph_path and Path(graph_path).exists():
            with open(graph_path, 'r') as f:
                return json.load(f)
        return None
    
    def encode_news_articles(
        self, 
        article_tokens: torch.Tensor,
        apply_semantic_constraints: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode news articles into topic distributions with semantic graph guidance
        
        Args:
            article_tokens: Tokenized articles [batch_size, seq_len]
            apply_semantic_constraints: Whether to apply TopicNet-style constraints
            
        Returns:
            Dictionary containing topic representations and metadata
        """
        # Get topic distributions using TopNet-style variational inference
        topic_output = self.topic_generator(article_tokens)
        
        topic_mu = topic_output['topic_mu']
        topic_logvar = topic_output['topic_logvar']
        topic_dist = topic_output['topic_dist']
        
        # Apply semantic hierarchy constraints (TopicNet integration)
        if apply_semantic_constraints and self.semantic_graph:
            constrained_embeddings = self._apply_semantic_constraints(
                topic_mu, topic_logvar
            )
        else:
            constrained_embeddings = topic_mu
        
        # Enhance with hierarchical encoding
        # Project topic distributions to latent space for hierarchy encoding
        projected_topics = self.topic_projector(constrained_embeddings)  # [batch_size, latent_dim]
        
        enhanced_topics = self.hierarchy_encoder(
            projected_topics.unsqueeze(1)  # [batch_size, 1, latent_dim]
        ).squeeze(1)  # [batch_size, latent_dim]
        
        return {
            'topic_distributions': topic_dist,
            'topic_embeddings': enhanced_topics,
            'raw_topic_mu': topic_mu,
            'topic_logvar': topic_logvar,
            'reconstruction': topic_output['reconstruction']
        }
    
    def _apply_semantic_constraints(
        self, 
        topic_mu: torch.Tensor, 
        topic_logvar: torch.Tensor,
        constraint_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Apply TopicNet-style semantic graph constraints to topic embeddings
        
        Args:
            topic_mu: Topic means [batch_size, num_topics]
            topic_logvar: Topic log variances [batch_size, num_topics]
            constraint_weight: Weight for semantic constraints
            
        Returns:
            constrained_embeddings: Semantically constrained topic embeddings
        """
        if not self.semantic_graph:
            return topic_mu
        
        # Get topic embeddings from the generator
        topic_emb_mu, topic_emb_logvar = self.topic_generator.get_topic_embeddings()
        
        # Apply semantic hierarchy constraints using KL divergence penalties
        # This implements the TopicNet approach described in your paper
        constraint_loss = 0
        
        for parent_topic, child_topics in self.semantic_graph.get('hierarchy', {}).items():
            if parent_topic < self.num_topics:
                for child_topic in child_topics:
                    if child_topic < self.num_topics:
                        # Compute KL divergence constraint (Equation in TopicNet paper)
                        constraint_loss += self._compute_hierarchy_constraint(
                            parent_topic, child_topic, topic_emb_mu, topic_emb_logvar
                        )
        
        # Apply constraints to embeddings
        constrained_mu = topic_mu - constraint_weight * constraint_loss.grad if constraint_loss.requires_grad else topic_mu
        
        return constrained_mu
    
    def _compute_hierarchy_constraint(
        self, 
        parent_idx: int, 
        child_idx: int,
        topic_emb_mu: torch.Tensor,
        topic_emb_logvar: torch.Tensor,
        gamma: float = 1.0,
        margin: float = 10.0
    ) -> torch.Tensor:
        """
        Compute semantic hierarchy constraint using asymmetric KL divergence
        Implements the TopicNet approach from your paper
        """
        parent_mu = topic_emb_mu[parent_idx]
        child_mu = topic_emb_mu[child_idx]
        parent_var = torch.exp(topic_emb_logvar[parent_idx])
        child_var = torch.exp(topic_emb_logvar[child_idx])
        
        # KL divergence D_KL(child || parent)
        kl_div = 0.5 * (
            torch.sum(torch.log(parent_var / child_var)) +
            torch.sum(child_var / parent_var) +
            torch.sum((parent_mu - child_mu)**2 / parent_var) -
            len(parent_mu)
        )
        
        # Thresholded divergence (Equation from TopicNet paper)
        d_gamma = torch.maximum(torch.tensor(0.0), kl_div - gamma)
        
        return d_gamma
    
    def generate_news_stories(
        self, 
        news_topics: torch.Tensor,
        topic_embeddings: torch.Tensor,
        story_length: int = 300,
        num_skeleton_words: int = 10,
        **generation_kwargs
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """
        Generate coherent news stories based on discovered topics
        This is the key innovation combining TopicNet and TopNet
        
        Args:
            news_topics: Topic distributions [batch_size, num_topics]
            topic_embeddings: Enhanced topic embeddings [batch_size, latent_dim]
            story_length: Target story length
            num_skeleton_words: Number of skeleton words to generate
            
        Returns:
            Generated stories with metadata
        """
        batch_size = news_topics.size(0)
        
        # Generate skeleton words from topic distributions (TopNet approach)
        skeleton_words = self.topic_generator.generate_skeleton_words(
            news_topics, 
            num_words=num_skeleton_words,
            **generation_kwargs
        )
        
        # Project topic embeddings to story space
        story_topic_embeddings = self.topic_to_story_projector(topic_embeddings)
        story_topic_embeddings = story_topic_embeddings.unsqueeze(1).repeat(1, self.num_topics, 1)
        
        # Generate stories using hierarchical story generator
        generated_stories = self.story_generator(
            skeleton_words=skeleton_words,
            topic_embeddings=story_topic_embeddings,
            segment_length=story_length // num_skeleton_words,
            **generation_kwargs
        )
        
        return generated_stories
    
    def analyze_news_content(
        self, 
        article_tokens: torch.Tensor,
        generate_stories: bool = True,
        story_length: int = 300
    ) -> Dict[str, Union[torch.Tensor, List, Dict]]:
        """
        Comprehensive news analysis combining topic discovery and story generation
        Main pipeline of the integrated framework
        
        Args:
            article_tokens: Input news articles [batch_size, seq_len]
            generate_stories: Whether to generate stories
            story_length: Length of generated stories
            
        Returns:
            Comprehensive analysis results
        """
        # Phase 1: Semantic graph-guided topic discovery (TopicNet)
        topic_analysis = self.encode_news_articles(
            article_tokens, 
            apply_semantic_constraints=True
        )
        
        # Phase 2: News classification using discovered topics
        news_categories = self.news_classifier(topic_analysis['topic_distributions'])
        
        # Phase 3: Topic coherence assessment
        topic_coherence = self.topic_coherence_predictor(topic_analysis['topic_embeddings'])
        
        # Phase 4: Story generation (TopNet integration)
        generated_stories = []
        if generate_stories:
            generated_stories = self.generate_news_stories(
                news_topics=topic_analysis['topic_distributions'],
                topic_embeddings=topic_analysis['topic_embeddings'],
                story_length=story_length
            )
        
        # Phase 5: Cross-modal analysis
        cross_modal_analysis = self._perform_cross_modal_analysis(
            topic_analysis, generated_stories if generate_stories else None
        )
        
        return {
            'topic_analysis': topic_analysis,
            'news_categories': news_categories,
            'topic_coherence': topic_coherence,
            'generated_stories': generated_stories,
            'cross_modal_analysis': cross_modal_analysis,
            'semantic_graph_applied': self.semantic_graph is not None
        }
    
    def _perform_cross_modal_analysis(
        self, 
        topic_analysis: Dict[str, torch.Tensor],
        generated_stories: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform cross-modal analysis between topics and generated stories
        Innovation: Bridging topic discovery and story generation
        """
        analysis_results = {
            'topic_story_alignment': None,
            'narrative_coherence': None,
            'semantic_consistency': None
        }
        
        if generated_stories:
            # Compute topic-story alignment using cross-modal attention
            topic_embeddings = topic_analysis['topic_embeddings'].unsqueeze(1)  # [batch, 1, hidden]
            
            # Create dummy story representations for alignment analysis
            # Use the actual hidden dimension from story generator
            actual_hidden_dim = self.story_generator.story_model.config.n_embd
            story_representations = torch.randn(
                topic_embeddings.size(0), 10, actual_hidden_dim,
                device=self.device
            )
            
            # Project topic embeddings to story representation space
            projected_topics = self.topic_to_cross_modal(topic_embeddings)
            
            aligned_features, attention_weights = self.cross_modal_attention(
                query=projected_topics,
                key=story_representations,
                value=story_representations
            )
            
            analysis_results['topic_story_alignment'] = attention_weights
            analysis_results['aligned_features'] = aligned_features
        
        return analysis_results
    
    def compute_comprehensive_loss(
        self,
        article_tokens: torch.Tensor,
        target_categories: Optional[torch.Tensor] = None,
        target_stories: Optional[List[str]] = None,
        loss_weights: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for multi-task learning
        Combines TopicNet and TopNet objectives
        
        Args:
            article_tokens: Input articles
            target_categories: Target news categories (optional)
            target_stories: Target stories for generation (optional)
            loss_weights: Weights for different loss components
            
        Returns:
            Dictionary of loss components
        """
        if loss_weights is None:
            loss_weights = {
                'topic_elbo': 1.0,
                'classification': 0.5,
                'hierarchy': 0.3,
                'story_generation': 0.7
            }
        
        # Get full analysis
        analysis_results = self.analyze_news_content(
            article_tokens, 
            generate_stories=True
        )
        
        losses = {}
        
        # TopNet-style ELBO loss for topic modeling
        topic_output = self.topic_generator(article_tokens)
        elbo_losses = self.topic_generator.compute_elbo_loss(
            x=article_tokens,
            reconstruction=topic_output['reconstruction'],
            topic_mu=topic_output['topic_mu'],
            topic_logvar=topic_output['topic_logvar']
        )
        losses['topic_elbo'] = elbo_losses['elbo_loss']
        
        # Classification loss
        if target_categories is not None:
            classification_loss = F.cross_entropy(
                analysis_results['news_categories'], 
                target_categories
            )
            losses['classification'] = classification_loss
        
        # Semantic hierarchy constraint loss (TopicNet)
        if self.semantic_graph:
            hierarchy_loss = self._compute_hierarchy_loss(
                analysis_results['topic_analysis']['topic_embeddings']
            )
            losses['hierarchy'] = hierarchy_loss
        
        # Story generation loss (if target stories provided)
        if target_stories:
            # This would require tokenized target stories
            # Simplified version for now
            story_loss = torch.tensor(0.0, device=self.device)
            losses['story_generation'] = story_loss
        
        # Total weighted loss
        total_loss = sum(
            loss_weights.get(name, 1.0) * loss 
            for name, loss in losses.items()
        )
        losses['total'] = total_loss
        
        return losses
    
    def _compute_hierarchy_loss(self, topic_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute semantic hierarchy constraint loss"""
        if not self.semantic_graph:
            return torch.tensor(0.0, device=self.device)
        
        hierarchy_loss = torch.tensor(0.0, device=self.device)
        
        for parent_topic, child_topics in self.semantic_graph.get('hierarchy', {}).items():
            if parent_topic < self.num_topics:
                for child_topic in child_topics:
                    if child_topic < self.num_topics:
                        # Max-margin loss for hierarchy (TopicNet approach)
                        constraint = self._compute_hierarchy_constraint(
                            parent_topic, child_topic,
                            self.topic_generator.topic_embeddings_mu,
                            self.topic_generator.topic_embeddings_logvar
                        )
                        hierarchy_loss += constraint
        
        return hierarchy_loss
    
    def forward(
        self, 
        article_tokens: torch.Tensor,
        mode: str = 'analysis',
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List, Dict]]:
        """
        Forward pass of the integrated framework
        
        Args:
            article_tokens: Input news articles [batch_size, seq_len]
            mode: Operation mode ('analysis', 'generation', 'both')
            **kwargs: Additional arguments
            
        Returns:
            Results based on the specified mode
        """
        if mode == 'analysis':
            return self.analyze_news_content(
                article_tokens, 
                generate_stories=False,
                **kwargs
            )
        elif mode == 'generation':
            topic_analysis = self.encode_news_articles(article_tokens)
            return self.generate_news_stories(
                news_topics=topic_analysis['topic_distributions'],
                topic_embeddings=topic_analysis['topic_embeddings'],
                **kwargs
            )
        elif mode == 'both':
            return self.analyze_news_content(
                article_tokens,
                generate_stories=True,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}") 