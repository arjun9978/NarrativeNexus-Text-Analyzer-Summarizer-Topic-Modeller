"""
LDA Topic Model Training Script

This script trains an LDA model on your dataset and saves it for inference.
You MUST run this before the FastAPI server can perform topic prediction.

Usage:
    python -m src.training.train_lda --input data/processed/corpus.txt --num-topics 10
"""
import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split

# Optional: Gensim alternative (commented out)
# from gensim import corpora
# from gensim.models import LdaModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LDATrainer:
    def __init__(self, n_topics: int = 10, max_iter: int = 50):
        """
        Initialize LDA trainer
        
        Args:
            n_topics: Number of topics to discover
            max_iter: Maximum iterations for LDA training
        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        
        # Vectorizer configuration
        self.vectorizer = CountVectorizer(
            max_df=0.95,  # Ignore terms appearing in >95% of documents
            min_df=2,      # Ignore terms appearing in <2 documents
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        
        # LDA model
        self.lda_model = None
        self.topic_names = {}
    
    def load_corpus(self, filepath: str) -> list:
        """
        Load text corpus from file
        Expected format: one document per line
        """
        logger.info(f"Loading corpus from {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Corpus file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def train(self, documents: list):
        """
        Train LDA model on document corpus
        """
        logger.info(f"Training LDA model with {self.n_topics} topics...")
        
        # Vectorize documents
        logger.info("Vectorizing documents...")
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        logger.info(f"Document-term matrix shape: {doc_term_matrix.shape}")
        
        # Train LDA
        logger.info("Running LDA training...")
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method='online',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.lda_model.fit(doc_term_matrix)
        
        logger.info("✓ LDA training complete!")
        
        # Compute perplexity
        perplexity = self.lda_model.perplexity(doc_term_matrix)
        logger.info(f"Model perplexity: {perplexity:.2f}")
        
        # Display topics
        self._display_topics()
    
    def _display_topics(self, n_top_words: int = 10):
        """
        Display top words for each topic
        """
        logger.info("\n" + "="*80)
        logger.info("DISCOVERED TOPICS")
        logger.info("="*80)
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_word_indices = topic.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            
            logger.info(f"\nTopic {topic_idx}:")
            logger.info(f"  {', '.join(top_words)}")
            
            # Auto-generate topic name from top 3 words
            self.topic_names[topic_idx] = f"{top_words[0]}_{top_words[1]}_{top_words[2]}"
        
        logger.info("\n" + "="*80)
    
    def save_model(self, output_path: str = "models/lda_model.pkl"):
        """
        Save trained model, vectorizer, and topic names
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        model_data = {
            'model': self.lda_model,
            'vectorizer': self.vectorizer,
            'topic_names': self.topic_names,
            'n_topics': self.n_topics
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Model saved to {output_path}")
    
    def evaluate(self, documents: list):
        """
        Evaluate model on held-out test set
        """
        # Split data
        train_docs, test_docs = train_test_split(documents, test_size=0.2, random_state=42)
        
        # Vectorize
        train_matrix = self.vectorizer.fit_transform(train_docs)
        test_matrix = self.vectorizer.transform(test_docs)
        
        # Train
        self.lda_model.fit(train_matrix)
        
        # Evaluate
        train_perplexity = self.lda_model.perplexity(train_matrix)
        test_perplexity = self.lda_model.perplexity(test_matrix)
        
        logger.info(f"Train perplexity: {train_perplexity:.2f}")
        logger.info(f"Test perplexity: {test_perplexity:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Train LDA topic model")
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/corpus.txt',
        help='Path to input corpus file (one document per line)'
    )
    parser.add_argument(
        '--num-topics',
        type=int,
        default=10,
        help='Number of topics to discover'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=50,
        help='Maximum LDA iterations'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/lda_model.pkl',
        help='Path to save trained model'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LDATrainer(n_topics=args.num_topics, max_iter=args.max_iter)
    
    # Load corpus
    documents = trainer.load_corpus(args.input)
    
    # Train model
    trainer.train(documents)
    
    # Save model
    trainer.save_model(args.output)
    
    logger.info("\n✅ Training complete! You can now run the FastAPI server.")

if __name__ == "__main__":
    main()
