"""
Train LDA Topic Model on 20 Newsgroups Dataset
This script trains an LDA model on the 20 Newsgroups dataset and saves it for inference.
"""

import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the 20 newsgroups categories
CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'
]

def train_lda_on_20newsgroups(n_topics=20, max_iter=100):
    """
    Train LDA model on 20 Newsgroups dataset
    
    Args:
        n_topics: Number of topics (should match 20 newsgroups categories)
        max_iter: Maximum number of iterations for LDA
    """
    logger.info("Training LDA model on 20 Newsgroups dataset...")
    logger.info("Fetching 20 Newsgroups dataset...")
    
    # Fetch training data (remove headers, footers, quotes for cleaner data)
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    logger.info(f"Loaded {len(newsgroups_train.data)} documents")
    logger.info(f"Number of categories: {len(newsgroups_train.target_names)}")
    
    # Create vectorizer
    logger.info("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_df=0.95,  # Ignore terms that appear in >95% of documents
        min_df=2,     # Ignore terms that appear in <2 documents
        max_features=5000,  # Limit to top 5000 features
        stop_words='english',
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    
    # Fit and transform the documents
    logger.info("Vectorizing documents...")
    doc_term_matrix = vectorizer.fit_transform(newsgroups_train.data)
    logger.info(f"Document-term matrix shape: {doc_term_matrix.shape}")
    
    # Train LDA model
    logger.info(f"Training LDA model with {n_topics} topics...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method='online',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    lda_model.fit(doc_term_matrix)
    
    # Calculate perplexity (lower is better)
    perplexity = lda_model.perplexity(doc_term_matrix)
    logger.info(f"Model perplexity: {perplexity:.2f}")
    
    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    
    logger.info("\n" + "="*80)
    logger.info("TOP WORDS FOR EACH TOPIC:")
    logger.info("="*80)
    
    topic_names = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        
        # Create topic name from top 3 words
        topic_name = '_'.join(top_words[:3])
        topic_names.append(topic_name)
        
        logger.info(f"\nTopic {topic_idx}: {topic_name}")
        logger.info(f"Top words: {', '.join(top_words)}")
    
    # Map topics to original newsgroup categories (approximate)
    # We'll create a mapping based on topic distribution
    logger.info("\n" + "="*80)
    logger.info("MAPPING TOPICS TO NEWSGROUP CATEGORIES:")
    logger.info("="*80)
    
    # Transform documents to get topic distributions
    topic_distributions = lda_model.transform(doc_term_matrix)
    
    # For each newsgroup category, find the dominant topic
    category_topic_mapping = {}
    for cat_idx, category in enumerate(newsgroups_train.target_names):
        # Get documents in this category
        cat_docs = newsgroups_train.target == cat_idx
        cat_topic_dist = topic_distributions[cat_docs].mean(axis=0)
        dominant_topic = cat_topic_dist.argmax()
        category_topic_mapping[category] = {
            'topic_id': int(dominant_topic),
            'topic_name': topic_names[dominant_topic],
            'confidence': float(cat_topic_dist[dominant_topic])
        }
        logger.info(f"{category} -> Topic {dominant_topic}: {topic_names[dominant_topic]}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    
    logger.info("\nSaving models...")
    with open('models/lda_model.pkl', 'wb') as f:
        pickle.dump(lda_model, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save topic names and category mapping
    metadata = {
        'topic_names': topic_names,
        'n_topics': n_topics,
        'perplexity': perplexity,
        'feature_names': feature_names.tolist(),
        'categories': CATEGORIES,
        'category_topic_mapping': category_topic_mapping
    }
    
    with open('models/lda_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"✓ LDA model saved to models/lda_model.pkl")
    logger.info(f"✓ Vectorizer saved to models/vectorizer.pkl")
    logger.info(f"✓ Metadata saved to models/lda_metadata.pkl")
    
    return lda_model, vectorizer, metadata

if __name__ == "__main__":
    logger.info("Training LDA model on 20 Newsgroups dataset...")
    lda_model, vectorizer, metadata = train_lda_on_20newsgroups(n_topics=20, max_iter=50)
    logger.info("\n✅ Training complete!")
