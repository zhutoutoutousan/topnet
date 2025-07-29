import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Tuple, Dict, Optional, List

class TopicGenerator(nn.Module):
    """
    TopNet Topic Generator that maps short text input to topic distribution
    as described in TopNet: Learning from Neural Topic Model to Generate Long Stories
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        embedding_dim: int = 300,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        dropout: float = 0.1
    ):
        super(TopicGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder network for topic distribution inference
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Topic distribution parameters
        self.topic_mu = nn.Linear(hidden_dim, num_topics)
        self.topic_logvar = nn.Linear(hidden_dim, num_topics)
        
        # Topic embeddings (Gaussian distributed)
        self.topic_embeddings_mu = nn.Parameter(torch.randn(num_topics, latent_dim))
        self.topic_embeddings_logvar = nn.Parameter(torch.randn(num_topics, latent_dim))
        
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(num_topics, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, vocab_size)
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize topic embeddings
        nn.init.normal_(self.topic_embeddings_mu, 0, 0.1)
        nn.init.normal_(self.topic_embeddings_logvar, 0, 0.1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input text to topic distribution parameters
        
        Args:
            x: Input token ids [batch_size, seq_len]
            
        Returns:
            mu: Topic distribution means [batch_size, num_topics]
            logvar: Topic distribution log variances [batch_size, num_topics]
        """
        # Get embeddings and average over sequence length
        embeddings = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        pooled = torch.mean(embeddings, dim=1)  # [batch_size, embedding_dim]
        
        # Encode to hidden representation
        hidden = self.encoder(pooled)  # [batch_size, hidden_dim]
        
        # Get topic distribution parameters
        mu = self.topic_mu(hidden)  # [batch_size, num_topics]
        logvar = self.topic_logvar(hidden)  # [batch_size, num_topics]
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for variational inference
        
        Args:
            mu: Means [batch_size, num_topics]
            logvar: Log variances [batch_size, num_topics]
            
        Returns:
            z: Sampled latent variables [batch_size, num_topics]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode topic distribution to word probabilities
        
        Args:
            z: Topic distribution [batch_size, num_topics]
            
        Returns:
            logits: Word logits [batch_size, vocab_size]
        """
        return self.decoder(z)
    
    def get_topic_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Gaussian-distributed topic embeddings
        
        Returns:
            mu: Topic embedding means [num_topics, latent_dim]
            logvar: Topic embedding log variances [num_topics, latent_dim]
        """
        return self.topic_embeddings_mu, self.topic_embeddings_logvar
    
    def compute_topic_similarity(self, topic_i: int, topic_j: int) -> torch.Tensor:
        """
        Compute symmetric similarity between topics using expected likelihood kernel
        
        Args:
            topic_i: First topic index
            topic_j: Second topic index
            
        Returns:
            similarity: Symmetric similarity score
        """
        mu_i = self.topic_embeddings_mu[topic_i]
        mu_j = self.topic_embeddings_mu[topic_j]
        logvar_i = self.topic_embeddings_logvar[topic_i]
        logvar_j = self.topic_embeddings_logvar[topic_j]
        
        var_i = torch.exp(logvar_i)
        var_j = torch.exp(logvar_j)
        
        # Expected likelihood kernel (Equation 1 in paper)
        mu_diff = mu_i - mu_j
        var_sum = var_i + var_j
        
        # Compute multivariate Gaussian probability
        log_det = torch.sum(torch.log(var_sum))
        mahalanobis = torch.sum(mu_diff**2 / var_sum)
        
        return torch.exp(-0.5 * (log_det + mahalanobis))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of TopicGenerator
        
        Args:
            x: Input token ids [batch_size, seq_len]
            
        Returns:
            Dictionary containing:
                - topic_mu: Topic distribution means
                - topic_logvar: Topic distribution log variances  
                - topic_dist: Sampled topic distribution
                - reconstruction: Reconstructed word logits
        """
        # Encode to topic distribution
        topic_mu, topic_logvar = self.encode(x)
        
        # Sample topic distribution
        topic_dist = self.reparameterize(topic_mu, topic_logvar)
        topic_dist = F.softmax(topic_dist, dim=-1)  # Normalize to probability distribution
        
        # Decode to word probabilities
        reconstruction = self.decode(topic_dist)
        
        return {
            'topic_mu': topic_mu,
            'topic_logvar': topic_logvar,
            'topic_dist': topic_dist,
            'reconstruction': reconstruction
        }
    
    def compute_elbo_loss(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor,
        topic_mu: torch.Tensor,
        topic_logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Evidence Lower Bound (ELBO) loss
        
        Args:
            x: Input token ids [batch_size, seq_len]
            reconstruction: Reconstructed word logits [batch_size, vocab_size]
            topic_mu: Topic distribution means [batch_size, num_topics]
            topic_logvar: Topic distribution log variances [batch_size, num_topics]
            beta: KL divergence weight
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (negative log likelihood)
        # Convert input to bag-of-words representation
        bow = torch.zeros(batch_size, self.vocab_size, device=x.device)
        for i in range(batch_size):
            tokens = x[i][x[i] != 0]  # Remove padding
            for token in tokens:
                bow[i, token] += 1
        
        recon_loss = F.cross_entropy(reconstruction, bow.argmax(dim=-1), reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + topic_logvar - topic_mu.pow(2) - topic_logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()
        
        # Total ELBO loss
        elbo_loss = recon_loss + beta * kl_loss
        
        return {
            'elbo_loss': elbo_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def generate_skeleton_words(
        self, 
        topic_dist: torch.Tensor, 
        num_words: int = 10,
        temperature: float = 1.0
    ) -> List[List[int]]:
        """
        Generate skeleton words from topic distribution for story generation
        
        Args:
            topic_dist: Topic distribution [batch_size, num_topics]
            num_words: Number of skeleton words to generate
            temperature: Sampling temperature
            
        Returns:
            List of skeleton word sequences for each input
        """
        batch_size = topic_dist.size(0)
        skeleton_words = []
        
        for i in range(batch_size):
            words = []
            current_topic_dist = topic_dist[i].unsqueeze(0)
            
            for _ in range(num_words):
                # Decode current topic distribution
                word_logits = self.decode(current_topic_dist)
                word_probs = F.softmax(word_logits / temperature, dim=-1)
                
                # Sample word
                word_idx = torch.multinomial(word_probs, 1).item()
                words.append(word_idx)
                
                # Update topic distribution based on selected word (optional)
                # This creates dependency between skeleton words
                
            skeleton_words.append(words)
        
        return skeleton_words 