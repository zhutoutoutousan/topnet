import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler

class ClusterAnalyzer:
    def __init__(self, method='hdbscan'):
        """
        Initialize the cluster analyzer
        Args:
            method: 'hdbscan' or 'kmeans'
        """
        self.method = method
        self.scaler = StandardScaler()
        self.reducer = None
        self.clusterer = None
        self.labels_ = None
        
    def reduce_dimensions(self, embeddings, n_neighbors=15, min_dist=0.1, n_components=2):
        """
        Reduce dimensions using UMAP
        Args:
            embeddings: numpy array of embeddings
            n_neighbors: number of neighbors for UMAP
            min_dist: minimum distance for UMAP
            n_components: number of components for reduction
        Returns:
            reduced embeddings
        """
        # Scale the embeddings
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        
        # Initialize and fit UMAP
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric='cosine'
        )
        
        return self.reducer.fit_transform(scaled_embeddings)
    
    def fit_predict(self, embeddings, **kwargs):
        """
        Fit the clustering model and predict clusters
        Args:
            embeddings: numpy array of embeddings
            **kwargs: additional arguments for clustering algorithms
        Returns:
            cluster labels
        """
        if self.method == 'hdbscan':
            return self._fit_predict_hdbscan(embeddings, **kwargs)
        else:
            return self._fit_predict_kmeans(embeddings, **kwargs)
    
    def _fit_predict_hdbscan(self, embeddings, min_cluster_size=50, min_samples=5):
        """Fit and predict using HDBSCAN"""
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_epsilon=0.5
        )
        
        self.labels_ = self.clusterer.fit_predict(embeddings)
        return self.labels_
    
    def _fit_predict_kmeans(self, embeddings, n_clusters=None):
        """Fit and predict using K-Means"""
        if n_clusters is None:
            # Estimate optimal number of clusters
            n_samples = embeddings.shape[0]
            n_clusters = int(np.sqrt(n_samples/2))
        
        self.clusterer = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42
        )
        
        self.labels_ = self.clusterer.fit_predict(embeddings)
        return self.labels_
    
    def evaluate_clustering(self, embeddings):
        """
        Evaluate clustering quality using multiple metrics
        Args:
            embeddings: numpy array of embeddings
        Returns:
            dict of evaluation metrics
        """
        if self.labels_ is None:
            raise ValueError("Must run fit_predict before evaluation")
            
        # Filter out noise points for HDBSCAN
        if self.method == 'hdbscan':
            mask = self.labels_ != -1
            embeddings = embeddings[mask]
            labels = self.labels_[mask]
        else:
            labels = self.labels_
            
        # Calculate metrics
        metrics = {
            'silhouette': silhouette_score(embeddings, labels),
            'davies_bouldin': davies_bouldin_score(embeddings, labels),
            'calinski_harabasz': calinski_harabasz_score(embeddings, labels)
        }
        
        return metrics 