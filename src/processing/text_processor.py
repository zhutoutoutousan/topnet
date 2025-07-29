import re
import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Initialize components
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_html(self, text):
        """Remove HTML tags"""
        return BeautifulSoup(text, 'html.parser').get_text()
    
    def normalize_text(self, text):
        """Normalize text by converting to lowercase and removing special characters"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stop words"""
        words = word_tokenize(text)
        return ' '.join([word for word in words if word not in self.stop_words])
    
    def lemmatize_text(self, text):
        """Lemmatize text"""
        words = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])
    
    def extract_entities(self, text):
        """Extract named entities"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def tokenize(self, text):
        """Tokenize text into word indices for model input"""
        # Simple tokenization for demo purposes
        # In production, use proper tokenizer like BERT tokenizer
        words = word_tokenize(text.lower())
        # Create a simple vocabulary mapping
        vocab = {}
        token_ids = []
        
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
            token_ids.append(vocab[word])
        
        return token_ids
    
    def process_text(self, text, include_ner=False):
        """Complete text processing pipeline"""
        # Clean HTML
        text = self.clean_html(text)
        
        # Normalize text
        text = self.normalize_text(text)
        
        # Remove stopwords
        text = self.remove_stopwords(text)
        
        # Lemmatize
        text = self.lemmatize_text(text)
        
        if include_ner:
            # Extract named entities
            entities = self.extract_entities(text)
            return text, entities
        
        return text 