"""
TopNet Integration Module

This module implements the integration of TopNet (neural topic model for story generation)
with the existing TopicNet (semantic graph-guided topic discovery) framework.

Key Components:
- TopicGenerator: Maps text to topic distributions using variational inference
- HierarchicalStoryGenerator: Generates coherent stories from topic distributions
- IntegratedTopicNetTopNet: Main framework combining both approaches
- TopicNetTopNetTrainer: Training pipeline for the integrated model

Innovation:
This integration represents a novel approach that combines:
1. Semantic graph-guided hierarchical topic modeling (TopicNet)
2. Variational topic inference for story generation (TopNet) 
3. Cross-modal alignment between topic discovery and story generation
4. Multi-task learning for news analysis and content synthesis
"""

from .topic_generator import TopicGenerator
from .story_generator import HierarchicalStoryGenerator
from .integrated_framework import IntegratedTopicNetTopNet
from .trainer import TopicNetTopNetTrainer, NewsDataset

__version__ = "1.0.0"
__author__ = "TopicNet-TopNet Integration Team"

__all__ = [
    "TopicGenerator",
    "HierarchicalStoryGenerator", 
    "IntegratedTopicNetTopNet",
    "TopicNetTopNetTrainer",
    "NewsDataset"
] 