"""
Retrain with BETTER LDA parameters + save category-based classifier
This combines LDA topics with actual category labels for accurate classification
"""

import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
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

def train_improved_model():
    """Train both LDA (for topic discovery) AND classifier (for accurate prediction)"""
    
    logger.info("="*80)
    logger.info("TRAINING IMPROVED 20 NEWSGROUPS MODEL")
    logger.info("="*80)
    
    # Fetch training data
    logger.info("Fetching 20 Newsgroups dataset...")
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    logger.info(f"Loaded {len(newsgroups_train.data)} documents")
    logger.info(f"Categories: {len(CATEGORIES)}")
    
    # Create TF-IDF vectorizer
    logger.info("\nCreating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_df=0.90,
        min_df=3,
        max_features=8000,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True  # Better for classification
    )
    
    doc_term_matrix = vectorizer.fit_transform(newsgroups_train.data)
    logger.info(f"Document-term matrix shape: {doc_term_matrix.shape}")
    
    # Train LDA for topic modeling (still useful for exploration)
    logger.info("\nTraining LDA model (20 topics)...")
    lda_model = LatentDirichletAllocation(
        n_components=20,
        max_iter=50,
        learning_method='online',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    lda_model.fit(doc_term_matrix)
    
    # Train Naive Bayes classifier for ACCURATE category prediction
    logger.info("\nTraining Naive Bayes classifier for category prediction...")
    classifier = MultinomialNB(alpha=0.01)
    classifier.fit(doc_term_matrix, newsgroups_train.target)
    
    # Calculate accuracy on training data
    train_accuracy = classifier.score(doc_term_matrix, newsgroups_train.target)
    logger.info(f"Training accuracy: {train_accuracy*100:.2f}%")
    
    # Test on test set
    logger.info("\nTesting on test set...")
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    test_matrix = vectorizer.transform(newsgroups_test.data)
    test_accuracy = classifier.score(test_matrix, newsgroups_test.target)
    logger.info(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    
    logger.info("\nSaving models...")
    
    # Save LDA model
    with open('models/lda_model.pkl', 'wb') as f:
        pickle.dump(lda_model, f)
    logger.info("✓ LDA model saved to models/lda_model.pkl")
    
    # Save vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info("✓ Vectorizer saved to models/vectorizer.pkl")
    
    # Save classifier
    with open('models/classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    logger.info("✓ Classifier saved to models/classifier.pkl")
    
    # Save metadata
    metadata = {
        'categories': CATEGORIES,
        'num_categories': len(CATEGORIES),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'num_topics': 20
    }
    
    with open('models/lda_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    logger.info("✓ Metadata saved to models/lda_metadata.pkl")
    
    logger.info("\n" + "="*80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Train Accuracy: {train_accuracy*100:.2f}%")
    logger.info(f"Test Accuracy: {test_accuracy*100:.2f}%")
    logger.info("\nThe model will now:")
    logger.info("  1. Use Naive Bayes for ACCURATE category prediction")
    logger.info("  2. Use LDA for topic probabilities (exploratory)")
    logger.info("  3. Return confidence scores for all 20 categories")
    logger.info("="*80)

if __name__ == '__main__':
    train_improved_model()
