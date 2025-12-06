"""
FastAPI main application for NarrativeNexus
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import services (will be created)
from .services.summarizer import SummarizerService
from .services.sentiment import SentimentService
from .services.topic_model import TopicModelService

# Initialize FastAPI app
app = FastAPI(
    title="NarrativeNexus API",
    description="Dynamic Text Analysis Platform - Summarization, Sentiment Analysis, Topic Modeling",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Input text to analyze")

class AnalysisResponse(BaseModel):
    summary_extractive: str
    summary_abstractive: str
    sentiment: dict
    topic: dict

# Global service instances (loaded on startup)
summarizer_service = None
sentiment_service = None
topic_service = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global summarizer_service, sentiment_service, topic_service
    logger.info("Loading models...")
    
    try:
        summarizer_service = SummarizerService()
        logger.info("✓ Summarizer loaded")
        
        sentiment_service = SentimentService()
        logger.info("✓ Sentiment analyzer loaded")
        
        topic_service = TopicModelService()
        logger.info("✓ Topic model loaded")
        
        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NarrativeNexus API",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text input - returns summary (extractive + abstractive), sentiment, and topic
    """
    try:
        text = request.text
        
        # Run all analyses
        logger.info(f"Analyzing text (length: {len(text)} chars)")
        
        # Summarization (both extractive and abstractive)
        summary_extractive = summarizer_service.summarize_extractive(text)
        summary_abstractive = summarizer_service.summarize_abstractive(text)
        
        # Sentiment analysis
        sentiment_result = sentiment_service.analyze(text)
        
        # Topic analysis: both supervised classification and unsupervised LDA modeling
        classification = topic_service.classify_topic(text)
        lda_modeling = topic_service.discover_topics_lda(text)
        topic_result = {
            "classification": classification,
            "lda": lda_modeling
        }
        
        return AnalysisResponse(
            summary_extractive=summary_extractive,
            summary_abstractive=summary_abstractive,
            sentiment=sentiment_result,
            topic=topic_result
        )
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a .txt file for analysis
    """
    try:
        # Validate file type
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are supported")
        
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        if len(text) < 10:
            raise HTTPException(status_code=400, detail="Text too short (minimum 10 characters)")
        
        logger.info(f"Processing uploaded file: {file.filename} ({len(text)} chars)")
        
        # Run all analyses
        summary_extractive = summarizer_service.summarize_extractive(text)
        summary_abstractive = summarizer_service.summarize_abstractive(text)
        sentiment_result = sentiment_service.analyze(text)
        classification = topic_service.classify_topic(text)
        lda_modeling = topic_service.discover_topics_lda(text)
        topic_result = {
            "classification": classification,
            "lda": lda_modeling
        }
        
        return JSONResponse({
            "filename": file.filename,
            "text_length": len(text),
            "summary_extractive": summary_extractive,
            "summary_abstractive": summary_abstractive,
            "sentiment": sentiment_result,
            "topic": topic_result
        })
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding error. Ensure file is UTF-8 encoded.")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "ok",
        "models": {
            "summarizer": summarizer_service is not None,
            "sentiment": sentiment_service is not None,
            "topic": topic_service is not None
        }
    }

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
