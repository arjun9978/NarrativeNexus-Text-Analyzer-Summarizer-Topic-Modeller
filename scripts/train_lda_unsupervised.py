"""
Train REAL UNSUPERVISED LDA Topic Model on 20 Newsgroups
LDA discovers latent topics WITHOUT using category labels
Then we map discovered topics to category names for better interpretability
"""
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORIES = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
    'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
    'sci.space', 'soc.religion.christian', 'talk.politics.guns',
    'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
]

def train_real_lda():
    """
    Train LDA UNSUPERVISED - discovers topics from text ONLY, no labels used
    After training, we analyze which topics align with which categories (for naming)
    """
    logger.info("=" * 80)
    logger.info("TRAINING REAL UNSUPERVISED LDA TOPIC MODEL")
    logger.info("=" * 80)
    
    # Fetch training data (NO LABELS USED IN TRAINING!)
    logger.info("Fetching 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    logger.info(f"Loaded {len(newsgroups.data)} documents")
    logger.info("⚠️  LDA will IGNORE category labels - pure unsupervised learning!")
    
    # Create CountVectorizer (LDA works better with counts, not TF-IDF)
    logger.info("Creating CountVectorizer for LDA...")
    lda_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=5,
        max_features=3000,
        stop_words='english'
    )
    
    doc_term_matrix = lda_vectorizer.fit_transform(newsgroups.data)
    logger.info(f"Document-term matrix shape: {doc_term_matrix.shape}")
    
    # Train UNSUPERVISED LDA model (no labels used here!)
    logger.info(f"Training UNSUPERVISED LDA with 20 topics...")
    logger.info("⚠️  LDA will discover topics based ONLY on word co-occurrence patterns!")
    lda_model = LatentDirichletAllocation(
        n_components=20,  # Discover 20 latent topics
        max_iter=100,     # More iterations for better convergence
        learning_method='online',
        learning_offset=50.,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    lda_model.fit(doc_term_matrix)
    perplexity = lda_model.perplexity(doc_term_matrix)
    logger.info(f"Model perplexity: {perplexity:.2f}")
    
    # Get feature names
    feature_names = lda_vectorizer.get_feature_names_out()
    
    # Extract topic keywords (what LDA discovered)
    logger.info("\n" + "=" * 80)
    logger.info("TOPICS DISCOVERED BY LDA (UNSUPERVISED):")
    logger.info("=" * 80)
    
    topic_keywords = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-15:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_keywords.append(top_words)
        
        # Create descriptive name from top 3 words
        topic_name = '_'.join(top_words[:3])
        logger.info(f"\nTopic {topic_idx}: {topic_name}")
        logger.info(f"  Top words: {', '.join(top_words[:15])}")
    
    # AFTER training (post-hoc analysis): Map discovered topics to category names
    # This is ONLY for interpretability - LDA didn't use these labels during training!
    logger.info("\n" + "=" * 80)
    logger.info("POST-HOC ANALYSIS: Which categories align with discovered topics?")
    logger.info("(This mapping is for naming only - NOT used during LDA training!)")
    logger.info("=" * 80)
    
    # Get topic distribution for all documents
    doc_topic_dist = lda_model.transform(doc_term_matrix)
    
    # For each category, find which LDA topic is most dominant
    topic_to_category = {}
    category_to_topic = {}
    
    for cat_idx, category in enumerate(newsgroups.target_names):
        # Get documents in this category
        cat_mask = newsgroups.target == cat_idx
        cat_topic_dist = doc_topic_dist[cat_mask].mean(axis=0)
        
        # Find which LDA topic appears most in this category
        dominant_topic = cat_topic_dist.argmax()
        confidence = cat_topic_dist[dominant_topic]
        
        # Don't overwrite if topic already mapped to another category with higher confidence
        if dominant_topic not in topic_to_category:
            topic_to_category[dominant_topic] = category
        
        category_to_topic[category] = {
            'topic_id': int(dominant_topic),
            'confidence': float(confidence),
            'keywords': topic_keywords[dominant_topic][:5]
        }
        
        logger.info(f"{category:30s} → Topic {dominant_topic:2d} ({confidence:.1%}) | {', '.join(topic_keywords[dominant_topic][:3])}")
    
    # Create human-readable topic names
    topic_names = []
    for topic_idx in range(20):
        if topic_idx in topic_to_category:
            # Use category name if one aligned strongly
            topic_names.append(topic_to_category[topic_idx])
        else:
            # Use keyword-based name for unmapped topics
            topic_names.append('_'.join(topic_keywords[topic_idx][:3]))
    
    logger.info("\n📝 Note: These category names are for human readability.")
    logger.info("   LDA discovered topics purely from word patterns, no labels used!")
    
    # Save models
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODELS:")
    logger.info("=" * 80)
    
    # Save LDA model
    with open('models/lda_model.pkl', 'wb') as f:
        pickle.dump(lda_model, f)
    logger.info("✓ LDA model saved to models/lda_model.pkl")
    
    # Save LDA vectorizer (separate from classification vectorizer!)
    with open('models/lda_vectorizer.pkl', 'wb') as f:
        pickle.dump(lda_vectorizer, f)
    logger.info("✓ LDA vectorizer saved to models/lda_vectorizer.pkl")
    
    # Update metadata with LDA topic names
    try:
        with open('models/lda_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = {'categories': CATEGORIES}
    
    metadata['topic_names'] = topic_names
    metadata['lda_topic_keywords'] = topic_keywords
    metadata['category_to_topic'] = category_to_topic
    metadata['perplexity'] = perplexity
    
    with open('models/lda_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    logger.info("✓ Metadata updated with LDA topic names")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ UNSUPERVISED LDA TOPIC MODELING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Perplexity: {perplexity:.2f} (lower is better)")
    logger.info(f"Topics discovered: 20")
    logger.info(f"Topics aligned with categories: {len(topic_to_category)}/20")
    logger.info("\n🎯 Now you have BOTH approaches:")
    logger.info("  1. Topic CLASSIFICATION (Naive Bayes) - predicts category labels")
    logger.info("  2. Topic MODELING (LDA) - discovers latent topics unsupervised")
    logger.info("\n💡 LDA shows topic DISTRIBUTIONS (e.g., 60% topic A + 30% topic B)")
    logger.info("   Classification shows single CATEGORY (e.g., 'comp.graphics')")
    logger.info("=" * 80)

if __name__ == "__main__":
    train_real_lda()
