import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

class HierarchicalStoryGenerator(nn.Module):
    """
    Hierarchical Story Generator that uses skeleton words from TopNet
    to generate coherent long stories with topic guidance
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_story_length: int = 512,
        skeleton_length: int = 10,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(HierarchicalStoryGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_story_length = max_story_length
        self.skeleton_length = skeleton_length
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Initialize GPT-2 based language model for story generation
        # Ensure hidden_dim is divisible by num_heads
        adjusted_hidden_dim = (hidden_dim // num_heads) * num_heads
        if adjusted_hidden_dim != hidden_dim:
            print(f"Warning: Adjusted hidden_dim from {hidden_dim} to {adjusted_hidden_dim} to be divisible by {num_heads}")
        
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_story_length,
            n_embd=adjusted_hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            use_cache=True
        )
        
        self.story_model = GPT2LMHeadModel(config)
        
        # Topic-guided attention mechanism
        self.topic_attention = nn.MultiheadAttention(
            embed_dim=adjusted_hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Skeleton word embedding and position encoding
        self.skeleton_embedding = nn.Embedding(vocab_size, adjusted_hidden_dim)
        self.position_embedding = nn.Embedding(skeleton_length, adjusted_hidden_dim)
        
        # Topic-to-text alignment layer
        self.topic_projector = nn.Sequential(
            nn.Linear(adjusted_hidden_dim, adjusted_hidden_dim),
            nn.ReLU(),
            nn.Linear(adjusted_hidden_dim, adjusted_hidden_dim)
        )
        
        # Hierarchical planning layers
        self.story_planner = nn.LSTM(
            input_size=adjusted_hidden_dim,
            hidden_size=adjusted_hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Content control gates
        self.content_gate = nn.Sequential(
            nn.Linear(adjusted_hidden_dim * 2, adjusted_hidden_dim),
            nn.Tanh(),
            nn.Linear(adjusted_hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def encode_skeleton_words(
        self, 
        skeleton_words: List[List[int]], 
        topic_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode skeleton words with topic guidance
        
        Args:
            skeleton_words: List of skeleton word sequences [batch_size, skeleton_length]
            topic_embeddings: Topic embeddings [batch_size, num_topics, embed_dim]
            
        Returns:
            encoded_skeleton: Encoded skeleton representations [batch_size, skeleton_length, hidden_dim]
        """
        batch_size = len(skeleton_words)
        skeleton_tensor = torch.tensor(skeleton_words, dtype=torch.long, device=self.device)
        
        # Get skeleton word embeddings
        skeleton_embeds = self.skeleton_embedding(skeleton_tensor)  # [batch_size, skeleton_length, hidden_dim]
        
        # Add positional encoding
        positions = torch.arange(self.skeleton_length, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeds = self.position_embedding(positions)
        skeleton_embeds = skeleton_embeds + pos_embeds
        
        # Apply topic-guided attention
        # Use topic embeddings as keys and values, skeleton embeddings as queries
        topic_keys = self.topic_projector(topic_embeddings)  # [batch_size, num_topics, hidden_dim]
        
        attended_skeleton, attention_weights = self.topic_attention(
            query=skeleton_embeds,
            key=topic_keys,
            value=topic_keys
        )
        
        # Combine original skeleton with topic-attended features
        gate_input = torch.cat([skeleton_embeds, attended_skeleton], dim=-1)
        gate = self.content_gate(gate_input)
        
        encoded_skeleton = gate * skeleton_embeds + (1 - gate) * attended_skeleton
        
        return encoded_skeleton
    
    def plan_story_structure(
        self, 
        encoded_skeleton: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Plan hierarchical story structure using LSTM
        
        Args:
            encoded_skeleton: Encoded skeleton [batch_size, skeleton_length, hidden_dim]
            
        Returns:
            story_plan: Planned story structure [batch_size, skeleton_length, hidden_dim]
            plan_states: Hidden states for generation [batch_size, skeleton_length, hidden_dim]
        """
        # Use LSTM to create story plan
        story_plan, (h_n, c_n) = self.story_planner(encoded_skeleton)
        
        return story_plan, story_plan
    
    def generate_story_segment(
        self, 
        skeleton_word: int,
        story_plan: torch.Tensor,
        previous_context: torch.Tensor,
        segment_length: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a story segment around a skeleton word
        
        Args:
            skeleton_word: Current skeleton word token id
            story_plan: Story plan context [1, hidden_dim]
            previous_context: Previous generated context [1, seq_len]
            segment_length: Length of segment to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            generated_segment: Generated token sequence
            updated_context: Updated context for next generation
        """
        # Initialize with skeleton word
        if previous_context is None:
            input_ids = torch.tensor([[skeleton_word]], device=self.device)
        else:
            input_ids = torch.cat([
                previous_context, 
                torch.tensor([[skeleton_word]], device=self.device)
            ], dim=-1)
        
        generated_tokens = []
        
        for _ in range(segment_length):
            # Get model outputs
            outputs = self.story_model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get last token logits
            
            # Apply topic guidance by modifying logits
            # This is where we inject the story plan information
            guided_logits = self._apply_topic_guidance(logits, story_plan)
            
            # Apply sampling strategy
            next_token = self._sample_next_token(
                guided_logits, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            generated_tokens.append(next_token.item())
            
            # Update input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Truncate if too long
            if input_ids.size(-1) > self.max_story_length:
                input_ids = input_ids[:, -self.max_story_length:]
        
        generated_segment = torch.tensor(generated_tokens, device=self.device)
        return generated_segment, input_ids
    
    def _apply_topic_guidance(
        self, 
        logits: torch.Tensor, 
        story_plan: torch.Tensor,
        guidance_weight: float = 0.3
    ) -> torch.Tensor:
        """
        Apply topic guidance to generation logits
        
        Args:
            logits: Original logits [batch_size, vocab_size]
            story_plan: Story plan context [batch_size, hidden_dim]
            guidance_weight: Weight for topic guidance
            
        Returns:
            guided_logits: Topic-guided logits
        """
        # Project story plan to vocabulary space
        vocab_guidance = torch.matmul(
            story_plan, 
            self.story_model.transformer.wte.weight.T
        )  # [batch_size, vocab_size]
        
        # Combine original logits with topic guidance
        guided_logits = logits + guidance_weight * vocab_guidance
        
        return guided_logits
    
    def _sample_next_token(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Sample next token using various strategies
        
        Args:
            logits: Token logits [batch_size, vocab_size]
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold
            
        Returns:
            next_token: Sampled token [batch_size, 1]
        """
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def generate_full_story(
        self,
        skeleton_words: List[int],
        topic_embeddings: torch.Tensor,
        story_plan: torch.Tensor,
        segment_length: int = 50,
        **generation_kwargs
    ) -> Dict[str, Union[List[int], str]]:
        """
        Generate a complete story using skeleton words and topic guidance
        
        Args:
            skeleton_words: Skeleton word sequence
            topic_embeddings: Topic embeddings for guidance
            story_plan: Story plan from hierarchical planner
            segment_length: Length per skeleton word segment
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated story and metadata
        """
        full_story_tokens = []
        context = None
        
        for i, skeleton_word in enumerate(skeleton_words):
            # Get current story plan context
            current_plan = story_plan[0, i:i+1, :]  # [1, 1, hidden_dim]
            
            # Generate segment around this skeleton word
            segment_tokens, context = self.generate_story_segment(
                skeleton_word=skeleton_word,
                story_plan=current_plan,
                previous_context=context,
                segment_length=segment_length,
                **generation_kwargs
            )
            
            full_story_tokens.extend(segment_tokens.tolist())
        
        return {
            'story_tokens': full_story_tokens,
            'skeleton_words': skeleton_words,
            'story_length': len(full_story_tokens)
        }
    
    def forward(
        self,
        skeleton_words: List[List[int]],
        topic_embeddings: torch.Tensor,
        segment_length: int = 50,
        **generation_kwargs
    ) -> List[Dict[str, Union[List[int], str]]]:
        """
        Forward pass: generate stories for a batch of skeleton words
        
        Args:
            skeleton_words: Batch of skeleton word sequences
            topic_embeddings: Topic embeddings [batch_size, num_topics, embed_dim]
            segment_length: Length per skeleton word segment
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated stories with metadata
        """
        # Encode skeleton words with topic guidance
        encoded_skeleton = self.encode_skeleton_words(skeleton_words, topic_embeddings)
        
        # Plan story structure
        story_plan, plan_states = self.plan_story_structure(encoded_skeleton)
        
        # Generate stories for each input in batch
        generated_stories = []
        for i in range(len(skeleton_words)):
            story_result = self.generate_full_story(
                skeleton_words=skeleton_words[i],
                topic_embeddings=topic_embeddings[i:i+1],
                story_plan=story_plan[i:i+1],
                segment_length=segment_length,
                **generation_kwargs
            )
            generated_stories.append(story_result)
        
        return generated_stories 