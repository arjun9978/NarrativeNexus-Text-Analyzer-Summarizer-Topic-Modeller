"""
Summarization Service - Extractive + Abstractive
"""
import logging
from typing import Optional
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

class SummarizerService:
    def __init__(self, language: str = "english"):
        """
        Initialize both extractive and abstractive summarizers
        """
        self.language = language
        
        # Extractive summarizer (TextRank)
        self.stemmer = Stemmer(language)
        self.extractive_summarizer = TextRankSummarizer(self.stemmer)
        self.extractive_summarizer.stop_words = get_stop_words(language)
        
        # Abstractive summarizer (Hugging Face transformer)
        logger.info("Loading abstractive summarizer model...")
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Use BART for abstractive summarization
            self.abstractive_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
            logger.info(f"Abstractive model loaded on {'GPU' if device == 0 else 'CPU'}")
        except Exception as e:
            logger.warning(f"Could not load BART model: {e}. Using fallback.")
            self.abstractive_pipeline = None
    
    def summarize_extractive(self, text: str, sentence_count: int = 3) -> str:
        """
        Extractive summarization using TextRank algorithm
        Selects most important sentences from original text
        """
        try:
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            summary_sentences = self.extractive_summarizer(parser.document, sentence_count)
            
            # Join sentences
            summary = " ".join([str(sentence) for sentence in summary_sentences])
            
            if not summary:
                # Fallback: return first few sentences
                sentences = text.split('. ')
                summary = '. '.join(sentences[:sentence_count]) + '.'
            
            return summary
        
        except Exception as e:
            logger.error(f"Extractive summarization error: {e}")
            # Fallback: return first 200 chars
            return text[:200] + "..." if len(text) > 200 else text
    
    def summarize_abstractive(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Abstractive summarization using transformer model (BART)
        Generates new sentences that capture key information
        """
        if self.abstractive_pipeline is None:
            logger.warning("Abstractive model not available, using extractive fallback")
            return self.summarize_extractive(text, sentence_count=2)
        
        try:
            # Truncate very long texts (BART has max token limit)
            max_input_length = 1024
            if len(text.split()) > max_input_length:
                text = ' '.join(text.split()[:max_input_length])
            
            # Generate summary
            result = self.abstractive_pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            summary = result[0]['summary_text']
            return summary
        
        except Exception as e:
            logger.error(f"Abstractive summarization error: {e}")
            # Fallback to extractive
            return self.summarize_extractive(text, sentence_count=2)
    
    def summarize_hybrid(self, text: str) -> dict:
        """
        Returns both extractive and abstractive summaries
        """
        return {
            "extractive": self.summarize_extractive(text),
            "abstractive": self.summarize_abstractive(text)
        }
