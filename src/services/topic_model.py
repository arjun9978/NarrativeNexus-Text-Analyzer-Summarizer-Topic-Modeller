"""
Topic Modeling Service - Dual approach with Classification AND LDA Topic Modeling
Provides both supervised classification and unsupervised topic discovery
"""
import pickle
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class TopicModelService:
    def __init__(self):
        # Classification models (Naive Bayes)
        self.classifier = None
        self.classifier_vectorizer = None
        self.categories = []
        
        # LDA topic modeling
        self.lda_model = None
        self.lda_vectorizer = None
        self.lda_topic_names = []
        
        self.load_models()
    
    def load_models(self):
        """Load both classification and LDA modeling models"""
        model_dir = "models"
        
        # ===== CLASSIFICATION MODELS (Naive Bayes) =====
        classifier_path = os.path.join(model_dir, "classifier.pkl")
        if os.path.exists(classifier_path):
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            logger.info(f"✓ Classification model loaded from {classifier_path}")
        else:
            logger.warning("⚠️  Classifier not found! Run: python scripts/retrain_better_model.py")
        
        # Load classifier vectorizer
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.classifier_vectorizer = pickle.load(f)
            logger.info(f"✓ Classification vectorizer loaded")
        
        # Load metadata with categories
        metadata_path = os.path.join(model_dir, "lda_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.categories = metadata.get('categories', [])
            logger.info(f"✓ Metadata loaded - {len(self.categories)} categories")
        
        # ===== LDA TOPIC MODELING =====
        lda_model_path = os.path.join(model_dir, "lda_model.pkl")
        if os.path.exists(lda_model_path):
            with open(lda_model_path, 'rb') as f:
                self.lda_model = pickle.load(f)
            logger.info(f"✓ LDA topic model loaded from {lda_model_path}")
            
            # Load LDA vectorizer (separate from classifier vectorizer)
            lda_vectorizer_path = os.path.join(model_dir, "lda_vectorizer.pkl")
            if os.path.exists(lda_vectorizer_path):
                with open(lda_vectorizer_path, 'rb') as f:
                    self.lda_vectorizer = pickle.load(f)
                logger.info(f"✓ LDA vectorizer loaded")
            
            # Load LDA topic names
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.lda_topic_names = metadata.get('topic_names', [])
                    if self.lda_topic_names:
                        logger.info(f"✓ LDA topics loaded - {len(self.lda_topic_names)} topics")
        else:
            logger.info("ℹ️  LDA model not found (optional feature)")
    
    def classify_topic(self, text: str) -> dict:
        """
        CLASSIFICATION: Predict which of 20 predefined categories text belongs to
        Uses Naive Bayes classifier - ACCURATE and FAST
        Returns: category name with probabilities for all 20 categories
        """
        if not self.classifier or not self.classifier_vectorizer:
            return {
                "topic_id": 0,
                "topic_name": "Unknown (Model not trained)",
                "confidence": 0.0,
                "probabilities": {}
            }
        
        try:
            text_vector = self.classifier_vectorizer.transform([text])
            proba = self.classifier.predict_proba(text_vector)[0]
            
            predicted_idx = np.argmax(proba)
            predicted_category = self.categories[predicted_idx]
            confidence = float(proba[predicted_idx])
            
            category_probs = {
                self.categories[i]: float(proba[i]) 
                for i in range(len(self.categories))
            }
            sorted_probs = dict(sorted(category_probs.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "topic_id": int(predicted_idx),
                "topic_name": predicted_category,
                "confidence": confidence,
                "probabilities": sorted_probs
            }
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return {
                "topic_id": 0,
                "topic_name": "Error during classification",
                "confidence": 0.0,
                "probabilities": {}
            }
    
    def discover_topics_lda(self, text: str) -> dict:
        """
        TOPIC MODELING (LDA): Discover latent topics in text
        Uses Latent Dirichlet Allocation - UNSUPERVISED topic discovery
        Returns: dominant topic with probability distribution across all discovered topics
        """
        if not self.lda_model or not self.lda_vectorizer:
            return {
                "topic_id": 0,
                "topic_name": "LDA model not available",
                "confidence": 0.0,
                "probabilities": {},
                "method": "lda_unavailable"
            }
        
        try:
            # Transform text using LDA vectorizer
            text_vector = self.lda_vectorizer.transform([text])
            
            # Get topic distribution
            topic_dist = self.lda_model.transform(text_vector)[0]
            
            # Find dominant topic
            dominant_idx = np.argmax(topic_dist)
            confidence = float(topic_dist[dominant_idx])
            
            # Map to category names using learned mapping
            if self.lda_topic_names and dominant_idx < len(self.lda_topic_names):
                topic_name = self.lda_topic_names[dominant_idx]
            else:
                topic_name = f"Topic_{dominant_idx}"
            
            # Create probability distribution for all topics
            topic_probs = {}
            for i in range(len(topic_dist)):
                if self.lda_topic_names and i < len(self.lda_topic_names):
                    name = self.lda_topic_names[i]
                else:
                    name = f"Topic_{i}"
                topic_probs[name] = float(topic_dist[i])
            
            sorted_probs = dict(sorted(topic_probs.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "topic_id": int(dominant_idx),
                "topic_name": topic_name,
                "confidence": confidence,
                "probabilities": sorted_probs,
                "method": "lda"
            }
            
        except Exception as e:
            logger.error(f"Error in LDA topic modeling: {e}")
            return {
                "topic_id": 0,
                "topic_name": "Error during topic modeling",
                "confidence": 0.0,
                "probabilities": {},
                "method": "lda_error"
            }

# Singleton instance
topic_service = TopicModelService()

def get_topic(text: str) -> dict:
    """Get topic classification (Naive Bayes) - backwards compatible"""
    return topic_service.classify_topic(text)

def classify_topic(text: str) -> dict:
    """Get topic classification using Naive Bayes classifier"""
    return topic_service.classify_topic(text)

def discover_topics_lda(text: str) -> dict:
    """Discover topics using LDA topic modeling"""
    return topic_service.discover_topics_lda(text)
