"""
Sentiment Analysis Service - Transformer-based with VADER fallback
"""
import logging
from typing import Dict
from transformers import pipeline
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class SentimentService:
    def __init__(self, use_transformer: bool = True):
        """
        Initialize sentiment analyzer
        Defaults to transformer-based, with VADER as fallback
        """
        self.use_transformer = use_transformer
        
        # Initialize VADER (always available as fallback)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Try to load transformer model
        if use_transformer:
            try:
                logger.info("Loading transformer sentiment model...")
                device = 0 if torch.cuda.is_available() else -1
                
                # Use fine-tuned RoBERTa for sentiment (offline mode - uses cached models)
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=device,
                    local_files_only=True  # Use cached models, no internet needed
                )
                logger.info(f"Sentiment model loaded on {'GPU' if device == 0 else 'CPU'}")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}. Using VADER.")
                self.transformer_pipeline = None
        else:
            self.transformer_pipeline = None
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of text
        Returns label (positive/negative/neutral) and confidence scores
        """
        # Try transformer first
        if self.transformer_pipeline is not None:
            return self._analyze_transformer(text)
        else:
            return self._analyze_vader(text)
    
    def _analyze_transformer(self, text: str) -> Dict[str, any]:
        """
        Transformer-based sentiment analysis (RoBERTa)
        """
        try:
            # Truncate if too long
            max_length = 512
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])
            
            # Get predictions for all labels
            all_results = self.transformer_pipeline(text, top_k=None)[0]
            
            # Find the top prediction
            top_result = max(all_results, key=lambda x: x['score'])
            label = top_result['label'].lower()
            score = top_result['score']
            
            # Normalize labels (twitter-roberta uses: negative, neutral, positive)
            if 'positive' in label or label == 'pos':
                sentiment_label = 'positive'
            elif 'negative' in label or label == 'neg':
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            # Build detailed score breakdown
            scores = {}
            for item in all_results:
                label_name = item['label'].lower()
                if 'positive' in label_name:
                    scores['positive'] = round(item['score'], 4)
                elif 'negative' in label_name:
                    scores['negative'] = round(item['score'], 4)
                else:
                    scores['neutral'] = round(item['score'], 4)
            
            return {
                "label": sentiment_label,
                "confidence": round(score, 4),
                "scores": scores,  # NEW: All sentiment scores
                "method": "transformer",
                "model": "roberta-base-sentiment",
                "text_length": len(text.split()),
                "analysis": self._get_sentiment_description(sentiment_label, score)
            }
        
        except Exception as e:
            logger.error(f"Transformer sentiment error: {e}. Falling back to VADER.")
            return self._analyze_vader(text)
    
    def _analyze_vader(self, text: str) -> Dict[str, any]:
        """
        VADER rule-based sentiment analysis (fallback)
        """
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine label based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                "label": label,
                "confidence": round(abs(compound), 4),
                "scores": {
                    "positive": round(scores['pos'], 4),
                    "neutral": round(scores['neu'], 4),
                    "negative": round(scores['neg'], 4),
                    "compound": round(compound, 4)
                },
                "method": "vader",
                "model": "rule-based"
            }
        
        except Exception as e:
            logger.error(f"VADER sentiment error: {e}")
            return {
                "label": "neutral",
                "confidence": 0.0,
                "error": str(e),
                "method": "vader"
            }
    
    def _get_sentiment_description(self, label: str, confidence: float) -> str:
        """Generate human-readable sentiment analysis"""
        strength = "strongly" if confidence > 0.8 else "moderately" if confidence > 0.6 else "slightly"
        
        descriptions = {
            "positive": f"This text expresses {strength} positive sentiment.",
            "negative": f"This text expresses {strength} negative sentiment.",
            "neutral": f"This text is {strength} neutral/objective in tone."
        }
        
        return descriptions.get(label, "Sentiment could not be determined.")
