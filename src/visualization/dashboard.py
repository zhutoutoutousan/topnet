import dash
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import logging
import sys
import time
from datetime import timedelta
from textblob import TextBlob
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.util import ngrams
import colorcet as cc
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Initialize stop words
STOP_WORDS = set(stopwords.words('english'))

# Add news sources to extended stop words
NEWS_SOURCES = {
    'bbc', 'cnn', 'reuters', 'ap', 'afp', 'guardian', 
    'yahoo', 'msn', 'huffington'
}

# Add more tech/business specific stop words but keep meaningful terms
DOMAIN_STOP_WORDS = {
    'according', 'says', 'said', 'told', 'announced',
    'million', 'billion', 'percent', 'percentage',
    'update', 'updated', 'latest', 'earlier', 'later',
    'first', 'second', 'third', 'fourth', 'fifth',
    'next', 'last', 'previous', 'currently', 'recently'
}

# Extended stop words including common reporting verbs and general terms
EXTENDED_STOP_WORDS = STOP_WORDS.union(NEWS_SOURCES).union(DOMAIN_STOP_WORDS).union({
    # Common reporting verbs
    'said', 'says', 'told', 'reported', 'announced', 'claimed', 'stated',
    'added', 'noted', 'mentioned', 'explained', 'described', 'confirmed',
    
    # Common temporal words
    'today', 'yesterday', 'tomorrow', 'week', 'month', 'year', 'time',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
    
    # Common quantities
    'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third',
    'many', 'several', 'some', 'few', 'much', 'more', 'most',
    
    # Common actions
    'made', 'make', 'making', 'take', 'taking', 'took', 'get', 'getting',
    'got', 'going', 'went', 'gone', 'come', 'coming', 'came',
    
    # Common adjectives
    'new', 'old', 'good', 'bad', 'big', 'small', 'high', 'low',
    'many', 'much', 'several', 'various', 'different',
    
    # Other common words
    'people', 'thing', 'way', 'day', 'man', 'men', 'woman', 'women',
    'time', 'year', 'work', 'life', 'world', 'part', 'case',
    'problem', 'fact', 'point', 'government', 'number', 'group',
    'company', 'question', 'example'
})

def _extract_topics_worker(text, top_n=5, min_word_length=4):
    """Worker function for topic extraction"""
    logger.debug(f"Starting topic extraction with text sample: {text[:200]}...")
    
    # Tokenize and clean
    words = word_tokenize(text.lower())
    logger.debug(f"Total words after tokenization: {len(words)}")
    
    # Get bigrams and trigrams
    bigrams = list(ngrams(words, 2))
    trigrams = list(ngrams(words, 3))
    logger.debug(f"Found {len(bigrams)} bigrams and {len(trigrams)} trigrams")
    
    # Create phrases with proper filtering
    bigram_phrases = [
        f"{w1}_{w2}" for w1, w2 in bigrams 
        if all(len(w) >= min_word_length and w.isalnum() for w in [w1, w2])
        and not all(w in EXTENDED_STOP_WORDS for w in [w1, w2])
    ]
    
    trigram_phrases = [
        f"{w1}_{w2}_{w3}" for w1, w2, w3 in trigrams 
        if all(len(w) >= min_word_length and w.isalnum() for w in [w1, w2, w3])
        and not all(w in EXTENDED_STOP_WORDS for w in [w1, w2, w3])
    ]
    
    logger.debug(f"After filtering: {len(bigram_phrases)} bigram phrases, {len(trigram_phrases)} trigram phrases")
    
    # Get single words with POS tagging
    tagged_words = pos_tag(words)
    logger.debug(f"Tagged words sample: {tagged_words[:10]}")
    
    # Keep only relevant nouns, adjectives, and verbs
    important_words = [
        word for word, tag in tagged_words
        if word.isalnum()
        and len(word) >= min_word_length
        and word not in EXTENDED_STOP_WORDS
        and (
            tag.startswith(('NN', 'NNP', 'NNPS')) or  # Nouns
            (tag.startswith('JJ') and not word.endswith('ed')) or  # Adjectives (excluding past participles)
            (tag.startswith('VB') and tag not in ['VBD', 'VBN'])  # Verbs (excluding past tense)
        )
    ]
    
    logger.debug(f"Found {len(important_words)} important words")
    if len(important_words) > 0:
        logger.debug(f"Sample important words: {important_words[:10]}")
    
    # Combine all terms with weights
    all_terms = (
        important_words +  # Weight: 1
        bigram_phrases * 2 +  # Weight: 2
        trigram_phrases * 3  # Weight: 3
    )
    
    # Count frequencies
    term_freq = Counter(all_terms)
    logger.debug(f"Total unique terms: {len(term_freq)}")
    if len(term_freq) > 0:
        logger.debug(f"Top 10 terms before filtering: {dict(term_freq.most_common(10))}")
    
    # Prioritize meaningful terms - Optimized version
    scored_terms = {}
    proper_nouns = {word for word, tag in tagged_words if tag.startswith('NNP')}
    
    for term, freq in term_freq.most_common(1000):  # Only process top 1000 terms
        score = freq
        # Boost score for proper nouns
        if term in proper_nouns:
            score *= 1.5
        # Boost score for compound terms
        if '_' in term:
            score *= (1 + 0.5 * term.count('_'))  # More boost for longer compounds
        scored_terms[term] = score
    
    # Get top terms efficiently
    filtered_terms = []
    seen_roots = set()
    
    for term, _ in sorted(scored_terms.items(), key=lambda x: x[1], reverse=True):
        if len(filtered_terms) >= top_n:
            break
            
        # Get the root form of the term
        root = term.split('_')[0]
        
        # Skip if we already have a term with this root
        if root in seen_roots:
            continue
            
        # Add the term if it's meaningful
        if (
            not any(stop in term.lower() for stop in EXTENDED_STOP_WORDS)
            and not any(src in term.lower() for src in NEWS_SOURCES)
            and len(term) >= min_word_length
        ):
            filtered_terms.append(term)
            seen_roots.add(root)
    
    return filtered_terms if filtered_terms else ["No significant topics found"]

def extract_topics(text, top_n=5, min_word_length=4, timeout_seconds=30):
    """Extract meaningful topics from text using POS tagging and n-gram analysis with timeout"""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_extract_topics_worker, text, top_n, min_word_length)
            return future.result(timeout=timeout_seconds)
    except TimeoutError:
        logger.warning(f"Topic extraction timed out after {timeout_seconds} seconds")
        return ["Topic extraction timed out"]
    except Exception as e:
        logger.error(f"Error in topic extraction: {str(e)}")
        return ["Error in topic extraction"]

def format_topic(topic):
    """Format a topic by replacing underscores with spaces and capitalizing"""
    words = topic.replace('_', ' ').split()
    return ' '.join(word.capitalize() for word in words)

class DashboardApp:
    def __init__(self, data, embeddings_2d, cluster_labels, texts):
        """Initialize the dashboard"""
        logger.info("Starting dashboard initialization...")
        start_time = time.time()
        
        logger.info("Setting up initial data...")
        self.data = data
        self.embeddings_2d = embeddings_2d
        self.cluster_labels = cluster_labels
        self.texts = texts
        
        # Generate t-SNE visualization
        logger.info("Generating t-SNE visualization...")
        self.tsne = TSNE(n_components=2, random_state=42)
        self.embeddings_tsne = self.tsne.fit_transform(embeddings_2d)
        
        logger.info(f"Data setup completed in {time.time() - start_time:.2f} seconds")
        
        # Initialize Dash app
        logger.info("Creating Dash app instance...")
        self.app = dash.Dash(__name__)
        logger.info(f"Dash app created in {time.time() - start_time:.2f} seconds")
        
        logger.info("Computing cluster metrics...")
        metric_start = time.time()
        self._compute_cluster_metrics()
        logger.info(f"Cluster metrics computed in {time.time() - metric_start:.2f} seconds")
        
        logger.info("Setting up dashboard layout...")
        layout_start = time.time()
        self.setup_layout()
        logger.info(f"Layout setup completed in {time.time() - layout_start:.2f} seconds")
        
        logger.info("Setting up callbacks...")
        callback_start = time.time()
        self.setup_callbacks()
        logger.info(f"Callbacks setup completed in {time.time() - callback_start:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Dashboard initialization completed in {total_time:.2f} seconds")

    def _get_cluster_label(self, cluster):
        """Generate a descriptive label for a cluster based on top keywords"""
        cluster_mask = (self.cluster_labels == cluster)
        cluster_texts = ' '.join(np.array(self.texts)[cluster_mask])
        
        logger.debug(f"Processing cluster {cluster} with {cluster_mask.sum()} articles")
        logger.debug(f"Sample text from cluster: {cluster_texts[:200]}...")
        
        # Extract meaningful topics
        top_topics = extract_topics(cluster_texts, top_n=5)
        
        # Format topics
        formatted_topics = [format_topic(topic) for topic in top_topics]
        
        return f"Cluster {cluster}: {', '.join(formatted_topics)}"

    def _compute_cluster_metrics(self):
        """Compute various metrics for each cluster"""
        try:
            # Fix datetime parsing to handle ISO8601 format with UTC timezone
            logger.debug("Parsing dates...")
            self.data['published_date'] = pd.to_datetime(self.data['published_date'], format='ISO8601', utc=True)
            
            # Get temporal distribution per cluster - Optimized version
            logger.debug("Computing temporal distribution...")
            start_time = time.time()
            
            # First group by date to reduce the data size
            self.data['date'] = self.data['published_date'].dt.date
            daily_counts = self.data.groupby(['cluster', 'date']).size().reset_index(name='count')
            
            # Convert back to datetime for Plotly
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            self.temporal_dist = daily_counts.rename(columns={'date': 'published_date', 'count': 'title'})
            
            logger.debug(f"Temporal distribution computed in {time.time() - start_time:.2f} seconds")
            
            # Compute word frequencies per cluster
            logger.debug("Computing word frequencies...")
            self.cluster_words = {}
            self.cluster_labels_desc = {}
            
            # Log cluster statistics
            unique_clusters = set(self.cluster_labels)
            logger.debug(f"Found {len(unique_clusters)} unique clusters")
            for cluster in unique_clusters:
                cluster_size = np.sum(self.cluster_labels == cluster)
                logger.debug(f"Cluster {cluster} has {cluster_size} articles")
            
            for cluster in unique_clusters:
                cluster_mask = (self.cluster_labels == cluster)
                cluster_texts = ' '.join(np.array(self.texts)[cluster_mask])
                
                logger.debug(f"\nProcessing cluster {cluster}:")
                logger.debug(f"Number of articles: {cluster_mask.sum()}")
                logger.debug(f"Total text length: {len(cluster_texts)}")
                logger.debug(f"Sample text: {cluster_texts[:200]}...")
                
                # Extract topics and compute frequencies
                topics = extract_topics(cluster_texts, top_n=50)  # Get more topics for word cloud
                
                # Create frequency dict with formatted topics
                self.cluster_words[cluster] = {
                    format_topic(topic): freq 
                    for topic, freq in Counter(topics).items()
                }
                
                # Generate descriptive label
                self.cluster_labels_desc[cluster] = self._get_cluster_label(cluster)
                
                logger.debug(f"Cluster {cluster} processed with {len(topics)} topics")
                logger.debug(f"Topics found: {topics[:10]}")
                
        except Exception as e:
            logger.error(f"Error in compute_cluster_metrics: {str(e)}")
            raise

    def generate_wordcloud(self, cluster):
        """Generate word cloud for a cluster"""
        try:
            # Get words and frequencies for the cluster
            if cluster not in self.cluster_words:
                return ""
                
            word_freq = self.cluster_words[cluster]
            if not word_freq:
                return ""
                
            # Create and configure word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                prefer_horizontal=0.7
            )
            
            # Generate word cloud
            wordcloud.generate_from_frequencies(word_freq)
            
            # Convert to image
            img = wordcloud.to_image()
            
            # Save to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Convert to base64
            return 'data:image/png;base64,' + base64.b64encode(img_bytes.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error generating word cloud: {str(e)}")
            return ""

    def generate_cluster_summary(self, cluster):
        """Generate a human-readable summary of cluster insights"""
        try:
            cluster_mask = (self.cluster_labels == cluster)
            cluster_data = self.data[cluster_mask]
            
            # Basic statistics
            total_articles = len(cluster_data)
            date_range = cluster_data['published_date'].max() - cluster_data['published_date'].min()
            date_range_days = date_range.days
            
            # Most frequent words (excluding common stop words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            word_freq = Counter(word.lower() for word in ' '.join(np.array(self.texts)[cluster_mask]).split() 
                              if word.lower() not in stop_words and len(word) > 3)
            top_words = [word for word, _ in word_freq.most_common(5)]
            
            # Source distribution
            sources = cluster_data['source'].value_counts()
            
            # Time analysis
            daily_counts = cluster_data.groupby(cluster_data['published_date'].dt.date).size()
            peak_date = daily_counts.idxmax()
            peak_articles = daily_counts.max()
            
            # Create summary
            summary = [
                html.Div([
                    html.H4(f"Cluster {cluster} Analysis", className='cluster-title'),
                    
                    # Overview section
                    html.Div([
                        html.H5("📊 Overview"),
                        html.P([
                            f"This cluster contains {total_articles} articles ",
                            f"spanning {date_range_days} days."
                        ])
                    ], className='insight-section'),
                    
                    # Key Topics section
                    html.Div([
                        html.H5("🎯 Key Topics"),
                        html.P("Most frequent topics in this cluster:"),
                        html.Ul([html.Li(word.title()) for word in top_words])
                    ], className='insight-section'),
                    
                    # Timeline Insights
                    html.Div([
                        html.H5("📅 Timeline Insights"),
                        html.P([
                            f"Peak activity: {peak_date} ",
                            f"with {peak_articles} articles"
                        ])
                    ], className='insight-section'),
                    
                    # Source Distribution
                    html.Div([
                        html.H5("📰 Source Distribution"),
                        html.Ul([
                            html.Li(f"{source}: {count} articles")
                            for source, count in sources.items()
                        ])
                    ], className='insight-section'),
                    
                    # Sample Articles
                    html.Div([
                        html.H5("📑 Sample Articles"),
                        html.Div([
                            html.Div([
                                html.H6(row['title'], style={'fontWeight': 'bold'}),
                                html.P(f"Source: {row['source']} | Date: {row['published_date'].strftime('%Y-%m-%d')}")
                            ]) for _, row in cluster_data.head(3).iterrows()
                        ])
                    ], className='insight-section')
                ], className='cluster-insights')
            ]
            
            return summary
        except Exception as e:
            logger.error(f"Error generating cluster summary: {str(e)}")
            return [html.P("Error generating cluster summary")]

    def _create_scatter_plot(self, embeddings, title, customdata=None):
        """Create a scatter plot for the embeddings"""
        # Create a mapping of cluster labels to colors
        unique_clusters = np.unique(self.cluster_labels)
        colors = px.colors.qualitative.Set3[:len(unique_clusters)]
        color_map = {str(cluster): color for cluster, color in zip(unique_clusters, colors)}
        
        # Create hover text with safety checks
        hover_text = []
        for i, label in enumerate(self.cluster_labels):
            try:
                # Get title safely
                article_title = str(self.data['title'].iloc[i]) if 'title' in self.data.columns else 'No title'
                article_title = (article_title[:50] + '...') if len(article_title) > 50 else article_title
                
                # Get source safely
                source = str(self.data['source'].iloc[i]) if 'source' in self.data.columns else 'Unknown source'
                
                # Get date safely
                try:
                    date = self.data['published_date'].iloc[i].strftime('%Y-%m-%d') if 'published_date' in self.data.columns else 'No date'
                except (AttributeError, ValueError):
                    date = 'No date'
                
                text = [
                    f"Cluster {label}",
                    f"Title: {article_title}",
                    f"Source: {source}",
                    f"Date: {date}"
                ]
            except Exception as e:
                logger.warning(f"Error creating hover text for point {i}: {str(e)}")
                text = [f"Cluster {label}", "Details not available"]
            
            hover_text.append("<br>".join(text))
        
        # Create the scatter plot
        fig = go.Figure()
        
        for cluster in unique_clusters:
            mask = self.cluster_labels == cluster
            fig.add_trace(go.Scatter(
                x=embeddings[mask, 0],
                y=embeddings[mask, 1],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(
                    size=8,
                    color=color_map[str(cluster)],
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=np.array(hover_text)[mask],
                hoverinfo='text',
                customdata=np.full(np.sum(mask), cluster)  # Store numeric cluster ID
            ))
        
        fig.update_layout(
            title=f"{title} Visualization",
            xaxis_title=f"{title} Dimension 1",
            yaxis_title=f"{title} Dimension 2",
            showlegend=True,
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        return fig

    def _create_temporal_distribution(self):
        """Create temporal distribution plot"""
        try:
            fig = go.Figure()
            
            # Create a mapping of cluster labels to colors
            unique_clusters = np.unique(self.cluster_labels)
            colors = px.colors.qualitative.Set3[:len(unique_clusters)]
            color_map = {str(cluster): color for cluster, color in zip(unique_clusters, colors)}
            
            # Check if temporal_dist exists and has required columns
            if hasattr(self, 'temporal_dist') and isinstance(self.temporal_dist, pd.DataFrame):
                required_cols = ['cluster', 'published_date', 'title']
                if all(col in self.temporal_dist.columns for col in required_cols):
                    for cluster in sorted(unique_clusters):
                        cluster_data = self.temporal_dist[self.temporal_dist['cluster'] == cluster]
                        if not cluster_data.empty:
                            fig.add_trace(go.Scatter(
                                x=cluster_data['published_date'],
                                y=cluster_data['title'],
                                name=f'Cluster {cluster}',
                                mode='lines+markers',
                                line=dict(color=color_map[str(cluster)]),
                                hovertemplate=(
                                    "Date: %{x}<br>" +
                                    "Articles: %{y}<br>" +
                                    "<extra></extra>"
                                )
                            ))
                else:
                    logger.warning("Missing required columns in temporal_dist")
            else:
                logger.warning("temporal_dist not available")
            
            fig.update_layout(
                title="Articles Over Time by Cluster",
                xaxis_title="Date",
                yaxis_title="Number of Articles",
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating temporal distribution plot: {str(e)}")
            # Return an empty figure with an error message
            fig = go.Figure()
            fig.add_annotation(
                text="Error creating temporal distribution plot",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig

    def setup_layout(self):
        """Set up the dashboard layout"""
        logger.debug("Creating UMAP scatter plot...")
        umap_scatter = self._create_scatter_plot(
            self.embeddings_2d, 
            'UMAP', 
            customdata=self.cluster_labels
        )
        
        logger.debug("Creating t-SNE scatter plot...")
        tsne_scatter = self._create_scatter_plot(
            self.embeddings_tsne, 
            't-SNE',
            customdata=self.cluster_labels
        )
        
        logger.debug("Creating temporal distribution plot...")
        temporal_dist = self._create_temporal_distribution()
        
        self.app.layout = html.Div([
            html.H1("News Article Clustering Dashboard"),
            
            html.Div([
                html.Div([
                    html.H3("UMAP Visualization"),
                    dcc.Graph(
                        id='umap-scatter',
                        figure=umap_scatter
                    )
                ], className='six columns'),
                
                html.Div([
                    html.H3("t-SNE Visualization"),
                    dcc.Graph(
                        id='tsne-scatter',
                        figure=tsne_scatter
                    )
                ], className='six columns'),
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.H3("Cluster Information"),
                    html.Div(id='cluster-info')
                ], className='six columns'),
                
                html.Div([
                    html.H3("Word Cloud"),
                    html.Img(id='wordcloud-img', style={'width': '100%'})
                ], className='six columns'),
            ], className='row'),
            
            html.Div([
                html.H3("Temporal Distribution"),
                dcc.Graph(
                    id='temporal-dist',
                    figure=temporal_dist
                )
            ], className='row')
        ])

    def setup_callbacks(self):
        """Set up interactive callbacks"""
        @self.app.callback(
            [Output('cluster-info', 'children'),
             Output('wordcloud-img', 'src')],
            [Input('umap-scatter', 'clickData'),
             Input('tsne-scatter', 'clickData')]
        )
        def update_cluster_info(umap_click, tsne_click):
            try:
                ctx = dash.callback_context
                if not ctx.triggered:
                    return "Click a point to see cluster insights", ""
                
                # Get the clicked point data from whichever plot was clicked
                click_data = umap_click if ctx.triggered[0]['prop_id'].startswith('umap') else tsne_click
                if not click_data:
                    return "Click a point to see cluster insights", ""
                
                # Extract cluster from customdata
                point = click_data['points'][0]
                cluster = point['customdata']  # This should be the numeric cluster ID
                
                if not isinstance(cluster, (int, np.integer)):
                    # If it's not already a number, try to extract it from the label
                    try:
                        # Try to extract cluster number from the label (e.g., "Cluster -1: ...")
                        cluster = int(str(cluster).split(':')[0].replace('Cluster ', '').strip())
                    except (ValueError, IndexError, AttributeError) as e:
                        logger.error(f"Error parsing cluster number: {str(e)}")
                        return "Error: Could not determine cluster", ""
                
                # Generate cluster summary and word cloud
                summary = self.generate_cluster_summary(cluster)
                wordcloud = self.generate_wordcloud(cluster)
                
                return summary, wordcloud
            except Exception as e:
                logger.error(f"Error in callback: {str(e)}")
                return f"Error: {str(e)}", ""

    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        try:
            logger.info(f"Starting dashboard server on http://127.0.0.1:{port}")
            self.app.run_server(debug=debug, port=port)
        except Exception as e:
            logger.error(f"Error starting dashboard server: {str(e)}")
            raise 