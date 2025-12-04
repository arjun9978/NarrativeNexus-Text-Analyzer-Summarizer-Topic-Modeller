"""
Topic Modeling Service - Uses Naive Bayes classifier for ACCURATE 20 Newsgroups prediction
"""
import pickle
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class TopicModelService:
    def __init__(self):
        self.classifier = None
        self.vectorizer = None
        self.categories = []
        self.load_models()
    
    def load_models(self):
        """Load classifier, vectorizer, and metadata"""
        model_dir = "models"
        
        # Load classifier
        classifier_path = os.path.join(model_dir, "classifier.pkl")
        if os.path.exists(classifier_path):
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            logger.info(f"✓ Classifier loaded from {classifier_path}")
        else:
            logger.warning("⚠️  Classifier not found! Run: python scripts/retrain_better_model.py")
        
        # Load vectorizer
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"✓ Vectorizer loaded from {vectorizer_path}")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "lda_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.categories = metadata.get('categories', [])
            logger.info(f"✓ Metadata loaded - {len(self.categories)} categories")
    
    def predict_topic(self, text: str) -> dict:
        """
        Predict the newsgroup category for given text
        Returns category name with confidence scores for ALL 20 categories
        """
        if not self.classifier or not self.vectorizer:
            return {
                "topic_id": 0,
                "topic_name": "Unknown (Model not trained)",
                "confidence": 0.0,
                "probabilities": {}
            }
        
        try:
            # Vectorize the text
            text_vector = self.vectorizer.transform([text])
            
            # Get probabilities for all categories
            proba = self.classifier.predict_proba(text_vector)[0]
            
            # Get predicted category
            predicted_idx = np.argmax(proba)
            predicted_category = self.categories[predicted_idx]
            confidence = float(proba[predicted_idx])
            
            # Create probabilities dictionary for ALL 20 categories (sorted by confidence)
            category_probs = {
                self.categories[i]: float(proba[i]) 
                for i in range(len(self.categories))
            }
            
            # Sort by probability (highest first)
            sorted_probs = dict(sorted(category_probs.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "topic_id": int(predicted_idx),
                "topic_name": predicted_category,
                "confidence": confidence,
                "probabilities": sorted_probs  # ALL 20 categories, sorted by confidence
            }
            
        except Exception as e:
            logger.error(f"Error predicting topic: {e}")
            return {
                "topic_id": 0,
                "topic_name": "Error during prediction",
                "confidence": 0.0,
                "probabilities": {}
            }

# Singleton instance
topic_service = TopicModelService()

def get_topic(text: str) -> dict:
    """Get topic classification for text"""
    return topic_service.predict_topic(text)
