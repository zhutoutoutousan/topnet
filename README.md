# Semantic Clustering for Topic Discovery

This project implements a complete pipeline for discovering topics in news articles using semantic clustering. It includes web crawling, text processing, embedding generation, clustering, and interactive visualization.

## Features

- Web crawler for news articles (Reuters, AP News)
- Text preprocessing pipeline with NLTK and spaCy
- Embedding generation using Sentence-BERT or OpenAI
- Clustering using HDBSCAN or K-Means
- Interactive visualization dashboard with Plotly Dash
- Comprehensive evaluation metrics

## Project Structure

```
.
├── data/
│   ├── raw/         # Raw crawled data
│   └── processed/   # Processed and clustered data
├── src/
│   ├── crawler/     # Web crawler implementation
│   ├── processing/  # Text processing and embedding
│   ├── clustering/  # Clustering algorithms
│   ├── visualization/ # Dashboard implementation
│   └── main.py      # Main execution script
├── tests/           # Test files
├── EXPERIMENT.md    # Experimental design
├── REQUIREMENT.md   # Project requirements
└── requirements.txt # Python dependencies
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. (Optional) Set up OpenAI API key if using OpenAI embeddings:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

1. Run the complete pipeline:
   ```bash
   python src/main.py
   ```

   This will:
   - Crawl news articles if not already present
   - Process the texts
   - Generate embeddings
   - Perform clustering
   - Launch the visualization dashboard

2. Access the dashboard:
   - Open a web browser
   - Navigate to `http://localhost:8050`

## Dashboard Features

- Interactive scatter plot of document clusters
- Cluster size distribution
- Top keywords per cluster
- Time distribution of articles
- Export capabilities (PNG, HTML, data)

## Evaluation Metrics

The clustering quality is evaluated using:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 