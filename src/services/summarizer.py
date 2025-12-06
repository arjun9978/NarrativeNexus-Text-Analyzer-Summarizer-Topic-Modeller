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
            # Use T5 for TRUE abstractive summarization with paraphrasing
            self.abstractive_pipeline = pipeline(
                "summarization",
                model="t5-base",  # T5 is trained for text-to-text generation
                device=device
            )
            logger.info(f"Abstractive model (T5) loaded on {'GPU' if device == 0 else 'CPU'}")
        except Exception as e:
            logger.error(f"Could not load T5 model: {e}")
            self.abstractive_pipeline = None
    
    def summarize_extractive(self, text: str, sentence_count: int = 3) -> str:
        """
        Extractive summarization using TextRank algorithm
        Selects most important sentences from original text
        """
        try:
            # Count actual sentences in text
            sentences_in_text = text.count('.') + text.count('!') + text.count('?')
            
            # Adjust sentence count to be reasonable (max 70% of original)
            if sentences_in_text <= 3:
                # Very short text - return 2 sentences max
                adjusted_count = min(2, sentences_in_text)
            else:
                # Longer text - request fewer than total
                adjusted_count = min(sentence_count, max(2, int(sentences_in_text * 0.6)))
            
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            summary_sentences = self.extractive_summarizer(parser.document, adjusted_count)
            
            # Join sentences
            summary = " ".join([str(sentence) for sentence in summary_sentences])
            
            if not summary or len(summary) >= len(text) * 0.95:
                # If summary is almost the same as original, force reduction
                sentences = text.split('. ')
                summary = '. '.join(sentences[:max(1, len(sentences) // 2)]) + '.'
            
            return summary
        
        except Exception as e:
            logger.error(f"Extractive summarization error: {e}")
            # Fallback: return first half of sentences
            sentences = text.split('. ')
            return '. '.join(sentences[:max(1, len(sentences) // 2)]) + '.'
    
    def summarize_abstractive(self, text: str, max_length: int = 200, min_length: int = 50) -> str:
        """
        Abstractive summarization using transformer model (T5)
        Generates compressed summaries with intelligent sentence selection
        """
        if self.abstractive_pipeline is None:
            logger.warning("Abstractive model not available, using extractive fallback")
            return self.summarize_extractive(text, sentence_count=2)
        
        try:
            # Truncate very long texts (T5 has max token limit)
            max_input_length = 512
            words = text.split()
            if len(words) > max_input_length:
                text = ' '.join(words[:max_input_length])
            
            # Calculate dynamic lengths - force much shorter output
            input_word_count = len(text.split())
            
            # CRITICAL: Force abstractive to be MUCH shorter than extractive
            max_length = min(64, max(25, int(input_word_count * 0.40)))
            min_length = min(15, max(10, int(max_length * 0.30)))
            
            # Add T5 prefix for summarization task
            text_with_prefix = "summarize: " + text
            
            # Generate summary with T5 - use sampling for variation
            result = self.abstractive_pipeline(
                text_with_prefix,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,            # Enable sampling for variation
                temperature=0.8,           # Higher temp for more creativity
                top_k=40,                  # Top-k sampling
                top_p=0.90,                # Nucleus sampling
                truncation=True,
                num_beams=1                # Disable beam search when sampling
            )
            
            summary = result[0]['summary_text']
            
            # Ensure summary is actually shorter
            if len(summary) >= len(text) * 0.80:
                logger.warning("T5 summary too long, forcing compression")
                sentences = summary.split('. ')
                summary = '. '.join(sentences[:max(1, len(sentences) // 2)]) + '.'
            
            return summary
        
        except Exception as e:
            logger.error(f"Abstractive summarization error: {e}")
            # Fallback to extractive with fewer sentences
            return self.summarize_extractive(text, sentence_count=2)
    
    def summarize_hybrid(self, text: str) -> dict:
        """
        Returns both extractive and abstractive summaries
        """
        return {
            "extractive": self.summarize_extractive(text),
            "abstractive": self.summarize_abstractive(text)
        }
