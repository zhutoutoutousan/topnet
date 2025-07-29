import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

class EmbeddingGenerator:
    def __init__(self, model_type='sbert'):
        """
        Initialize the embedding generator
        Args:
            model_type: 'sbert' or 'openai'
        """
        self.model_type = model_type
        
        if model_type == 'sbert':
            self.model = SentenceTransformer('all-mpnet-base-v2')
        elif model_type == 'openai':
            load_dotenv()
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            raise ValueError("model_type must be either 'sbert' or 'openai'")
    
    def generate_embeddings(self, texts, batch_size=32):
        """
        Generate embeddings for a list of texts
        Args:
            texts: List of strings
            batch_size: Batch size for processing
        Returns:
            numpy array of embeddings
        """
        if self.model_type == 'sbert':
            return self._generate_sbert_embeddings(texts, batch_size)
        else:
            return self._generate_openai_embeddings(texts, batch_size)
    
    def _generate_sbert_embeddings(self, texts, batch_size):
        """Generate embeddings using Sentence-BERT"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)
    
    def _generate_openai_embeddings(self, texts, batch_size):
        """Generate embeddings using OpenAI API"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # OpenAI API call
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings) 