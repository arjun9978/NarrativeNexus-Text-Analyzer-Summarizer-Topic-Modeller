# NarrativeNexus: The Dynamic Text Analysis Platform

## 1. Introduction

The goal of this project is to develop a dynamic text analysis platform that can accept various types of text data, perform comprehensive analysis including summarization, sentiment detection, and topic classification, delivering actionable insights to users.

The platform is designed to efficiently process diverse text inputs—whether articles, reports, reviews, or social media content—by performing three core analytical tasks:

1. **Text Summarization**: Extract key information and generate concise summaries using both extractive and abstractive methods
2. **Sentiment Analysis**: Assess the emotional tone of the text with high accuracy using transformer-based models
3. **Topic Classification**: Identify the primary subject matter from 20 predefined categories using supervised machine learning

This comprehensive solution leverages state-of-the-art deep learning models and natural language processing algorithms to provide users with multi-faceted insights from text data. The platform saves time, enhances decision-making, and delivers real value for researchers, businesses, content analysts, and anyone working with large volumes of text.

Beyond basic analysis, the system provides confidence scores, probability distributions, and detailed breakdowns of results, enabling users to make informed decisions based on the extracted insights.

---

## 2. Methodology

### 2.1 Data Collection and Input Handling

**Data Sources**: The platform accepts text data through multiple input methods:
- Direct text input via web interface
- File upload (`.txt` format)
- API endpoints for programmatic access

**Input Module**: A user-friendly Flask-based web interface allows users to:
- Type or paste text directly into a textarea
- Upload `.txt` files via drag-and-drop or file selection
- Submit text for analysis through RESTful API endpoints

**Backend API**: FastAPI framework handles all requests with automatic validation, ensuring data consistency and proper error handling.

### 2.2 Data Preprocessing

**Text Cleaning Pipeline**:
- Remove special characters, excessive whitespace, and URLs
- Normalize text encoding to UTF-8
- Handle punctuation appropriately based on the analysis task
- Remove English stopwords for topic classification

**Tokenization**:
- Sentence tokenization using NLTK's Punkt tokenizer
- Word tokenization for vectorization
- Subword tokenization for transformer models (BART, RoBERTa)

**Normalization**:
- Case normalization where appropriate
- Handling of contractions and special linguistic patterns

### 2.3 Topic Classification Implementation

**Algorithm Selection**: Multinomial Naive Bayes classifier was chosen for topic classification due to its speed, accuracy, and interpretability on text data.

**Dataset**: The model is trained on the **20 Newsgroups dataset**, consisting of approximately 18,000 newsgroup documents partitioned across 20 different categories:

- `alt.atheism`
- `comp.graphics`
- `comp.os.ms-windows.misc`
- `comp.sys.ibm.pc.hardware`
- `comp.sys.mac.hardware`
- `comp.windows.x`
- `misc.forsale`
- `rec.autos`
- `rec.motorcycles`
- `rec.sport.baseball`
- `rec.sport.hockey`
- `sci.crypt`
- `sci.electronics`
- `sci.med`
- `sci.space`
- `soc.religion.christian`
- `talk.politics.guns`
- `talk.politics.mideast`
- `talk.politics.misc`
- `talk.religion.misc`

**Model Training Process**:
1. Download and preprocess 20 Newsgroups corpus (headers, footers, quotes removed)
2. Create TF-IDF vectorizer with 5000 features, including unigrams and bigrams
3. Train Multinomial Naive Bayes classifier
4. Achieve 65% test accuracy across 20 categories
5. Save trained model, vectorizer, and metadata for inference

**Inference**: When new text is submitted, it is vectorized using the same TF-IDF transformation and the classifier predicts probabilities for all 20 categories.

### 2.4 Sentiment Analysis

**Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` - A fine-tuned RoBERTa transformer model

**Architecture**: RoBERTa (Robustly Optimized BERT Approach) with 125 million parameters, specifically fine-tuned on Twitter sentiment data for robust performance on diverse text types.

**Sentiment Detection Process**:
1. Tokenize input text using RoBERTa tokenizer
2. Generate contextual embeddings through transformer layers
3. Pass through classification head to produce sentiment logits
4. Apply softmax to obtain probability distribution over sentiment classes
5. Return dominant sentiment label with confidence score and detailed probabilities

**Categories**: 
- **Positive**: Text expressing favorable opinions, satisfaction, or optimism
- **Negative**: Text expressing unfavorable opinions, dissatisfaction, or pessimism  
- **Neutral**: Text with balanced or objective tone

**Output**: Sentiment label, confidence score (0-1), all three class probabilities, and model identifier

### 2.5 Summarization Techniques

The platform implements both extractive and abstractive summarization to provide complementary perspectives on the text:

#### Extractive Summarization (TextRank Algorithm)
- **Method**: Graph-based ranking algorithm similar to PageRank
- **Library**: Sumy
- **Process**:
  1. Split text into sentences
  2. Create similarity matrix based on sentence embeddings
  3. Apply PageRank algorithm to identify most important sentences
  4. Select top-ranked sentences to form summary
- **Advantages**: Fast, preserves original phrasing, deterministic
- **Use Case**: When exact quotations are needed

#### Abstractive Summarization (T5 Transformer)
- **Model**: `t5-base` (220 million parameters)
- **Architecture**: Text-to-Text Transfer Transformer (T5) - unified framework for all NLP tasks
- **Pre-training**: Trained on C4 dataset with text-to-text paradigm for true generation
- **Process**:
  1. Prefix input with "summarize: " for task specification
  2. Tokenize and encode through T5 encoder layers
  3. Decode with sampling (temperature=0.7) for creative paraphrasing
  4. Use top-k and nucleus sampling for diverse outputs
- **Advantages**: Generates truly paraphrased summaries with natural language variation; uses sampling for creative rewording
- **Use Case**: When you need genuinely different wording and creative summarization

**Output**: Both summaries are returned together, allowing users to compare extractive (original key sentences) and abstractive (T5-generated paraphrased version).

### 2.6 API Architecture and User Interface

**Backend Framework**: FastAPI
- High-performance asynchronous Python web framework
- Automatic interactive API documentation (Swagger UI)
- Request/response validation using Pydantic models
- Built-in error handling and HTTP exception management

**API Endpoints**:
- `GET /` - Service information and welcome message
- `GET /health` - Health check returning model availability status
- `POST /analyze` - Analyze text provided in JSON payload
- `POST /upload` - Upload and analyze `.txt` file
- `GET /docs` - Interactive Swagger UI for API testing

**Frontend Interface**: Flask Web Application
- Premium cyberpunk-themed dark UI with black and neon green aesthetics
- Tab-based input (type text or upload file)
- Drag-and-drop file upload support
- Real-time character counting
- Beautiful results visualization showing all three analyses
- Animated confidence bars with shimmer effects
- Print-friendly results page

**Response Format** (JSON):
```json
{
  "summary_extractive": "Key sentences from original text...",
  "summary_abstractive": "AI-generated paraphrased summary...",
  "sentiment": {
    "label": "positive",
    "confidence": 0.92,
    "all_scores": {
      "positive": 0.92,
      "neutral": 0.06,
      "negative": 0.02
    },
    "method": "transformer",
    "model": "roberta-base-sentiment"
  },
  "topic": {
    "category": "comp.sys.mac.hardware",
    "confidence": 0.81,
    "all_probabilities": {
      "comp.sys.mac.hardware": 0.81,
      "comp.graphics": 0.05,
      ...
    }
  }
}
```

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │    Flask Web UI (Port 5000) - Premium Dark Theme    │   │
│  │  • Text Input  • File Upload  • Results Display     │   │
│  │  • Cyberpunk aesthetics with animations             │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           │ HTTP POST                       │
│                           ▼                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FASTAPI BACKEND (Port 8000)                │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Endpoints: /analyze | /upload |              │  │
│  │  • Request Validation  • Model Loading  • Routing   │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│     ┌─────────────────────┼─────────────────────┐          │
│     │                     │                     │          │
│     ▼                     ▼                     ▼          │
│  ┌──────────┐      ┌─────────────┐      ┌──────────────┐  │
│  │Summarizer│      │  Sentiment  │      │Topic Classify│  │
│  │ Service  │      │   Service   │      │   Service    │  │
│  └──────────┘      └─────────────┘      └──────────────┘  │
│       │                   │                     │          │
└───────┼───────────────────┼─────────────────────┼──────────┘
        │                   │                     │
        ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING MODELS                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ SUMMARIZATION                                       │   │
│  │  • TextRank (Extractive) - Graph-based ranking      │   │
│  │  • T5 (Abstractive) - t5-base                       │   │
│  │    220M parameters, Sampling-based Paraphrasing     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ SENTIMENT ANALYSIS                                  │   │
│  │  • RoBERTa - cardiffnlp/twitter-roberta-base        │   │
│  │    125M parameters, Fine-tuned Transformer          │   │
│  │    Output: Positive/Negative/Neutral + All Scores   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ TOPIC CLASSIFICATION                                │   │
│  │  • Multinomial Naive Bayes Classifier               │   │
│  │  • Trained on 20 Newsgroups Dataset (65% accuracy)  │   │
│  │  • TF-IDF Vectorizer (5000 features)                │   │
│  │  • 20 Categories from news, tech, sports, etc.      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    SAVED MODELS (Disk)                      │
│                                                             │
│  • models/classifier.pkl          (Naive Bayes 500KB)       │
│  • models/vectorizer.pkl          (TF-IDF vectorizer)       │
│  • models/lda_metadata.pkl        (20 categories mapping)   │
│  • ~/.cache/huggingface/          (T5 & RoBERTa 1.5GB)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Expected Deliverables

✅ **A Fully Functional Dynamic Text Analysis Platform** capable of:
- Accepting text input via web interface or API
- Processing `.txt` file uploads
- Performing three types of analysis simultaneously

✅ **Trained Topic Classification Model**:
- Multinomial Naive Bayes trained on 20 Newsgroups dataset
- 65% test accuracy across 20 categories
- Saved model artifacts for inference

✅ **Pre-trained Transformer Models**:
- T5 for abstractive summarization (220M parameters, sampling-based paraphrasing)
- RoBERTa for sentiment analysis (125M parameters)
- Automatic model downloading from Hugging Face Hub

✅ **Interactive Web Interface**:
- Flask-based premium dark theme UI
- Text input and file upload
- Real-time analysis with animated loading
- Beautiful results visualization with glow effects
- Print-friendly output

✅ **RESTful API with Documentation**:
- FastAPI backend with 4 endpoints
- Automatic Swagger UI documentation
- Request validation and error handling
- JSON response format

✅ **Comprehensive Documentation**:
- This README with methodology and usage instructions
- API documentation via Swagger UI
- Code comments and docstrings
- Quick start guide

---

## 5. Project Timeline (8 Weeks)

### Week 1-2: Research & Planning
- ✅ Literature review on text analysis techniques
- ✅ Technology stack selection (FastAPI, Flask, Transformers)
- ✅ Dataset identification (20 Newsgroups for topic modeling)
- ✅ Architecture design and API planning

### Week 3-4: Model Development
- ✅ Implement TextRank extractive summarization
- ✅ Integrate BART transformer for abstractive summarization
- ✅ Integrate RoBERTa for sentiment analysis
- ✅ Train Multinomial Naive Bayes on 20 Newsgroups dataset
- ✅ Model evaluation and performance tuning

### Week 5-6: Backend & API Development
- ✅ FastAPI application setup with 4 endpoints
- ✅ Service layer for each ML model
- ✅ Request/response validation
- ✅ Error handling and logging
- ✅ API testing and documentation (Swagger)

### Week 7: Frontend Development
- ✅ Flask web application setup
- ✅ Premium dark theme UI with cyberpunk aesthetics
- ✅ Text input and file upload functionality
- ✅ Results visualization with animations
- ✅ Responsive design for mobile/desktop

### Week 8: Testing, Documentation & Deployment
- ✅ End-to-end testing of all features
- ✅ Performance optimization (model loading, caching)
- ✅ Comprehensive documentation (README, API docs)
- ✅ Quick start guide and troubleshooting
- ✅ Git repository setup and code push

---

## 6. Installation and Setup

### Prerequisites
- **Python**: 3.12 or higher
- **RAM**: 4GB minimum (8GB recommended for transformer models)
- **Storage**: ~3GB for model downloads
- **Operating System**: Windows, Linux, or macOS

### Step-by-Step Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/springboardmentor443m-coder/NarrativeNexus-Dynamic-Text-Analysis.git
cd NarrativeNexus-Dynamic-Text-Analysis
git checkout ArjunPratap_Infosys
```

#### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `fastapi==0.115.12` - Backend API framework
- `uvicorn==0.34.2` - ASGI server
- `flask==3.1.2` - Web UI framework
- `transformers==4.48.3` - Hugging Face models
- `torch==2.6.0` - PyTorch deep learning
- `scikit-learn==1.7.2` - Naive Bayes and TF-IDF
- `nltk==3.9.2` - Text preprocessing
- `sumy==0.11.0` - Extractive summarization

#### Step 4: Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### Step 5: Train Topic Classification Model (Optional - Already Trained)
```bash
python scripts/retrain_better_model.py
```

**What this does**:
1. Downloads 20 Newsgroups dataset (~14 MB)
2. Trains Multinomial Naive Bayes classifier
3. Saves model, vectorizer, and metadata to `models/` directory

**Expected time**: 2-3 minutes

**Output files**:
- `models/classifier.pkl` - Trained classifier (500KB)
- `models/vectorizer.pkl` - TF-IDF vectorizer
- `models/lda_metadata.pkl` - Category names and mappings

---

## 7. Running the Application

### Option 1: Using Batch Script (Recommended for Windows)

```bash
# Double-click or run:
START_SERVERS.bat
```

This automatically:
1. Starts FastAPI backend on port 8000
2. Starts Flask UI on port 5000
3. Opens browser to http://127.0.0.1:5000

### Option 2: Manual Start (All Platforms)

#### Terminal 1: Start FastAPI Backend
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# Start FastAPI server
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

**Wait for**: `"All models loaded successfully!"`

**Models loaded on startup**:
- ✓ Summarizer (TextRank + T5)
- ✓ Sentiment analyzer (RoBERTa)
- ✓ Topic classifier (Naive Bayes)

#### Terminal 2: Start Flask Web UI
```bash
# Activate virtual environment (in new terminal)
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# Start Flask application
python ui/app.py
```

#### Access the Application
1. Open browser to: **http://127.0.0.1:5000**
2. Choose input method:
   - **Type Text**: Paste or type your text directly
   - **Upload File**: Drag-and-drop or select a `.txt` file
3. Click "Analyze Text"
4. View comprehensive results:
   - Extractive summary (key sentences)
   - Abstractive summary (AI-generated)
   - Sentiment analysis (label + all probabilities)
   - Topic classification (category + all 20 probabilities)

### Option 3: Using API Directly (For Developers)

#### Access Swagger Documentation
Navigate to: **http://127.0.0.1:8000/docs**

Interactive API documentation with test interface.

#### Example API Calls

**1. Health Check**
```bash
curl http://127.0.0.1:8000/health
```

**2. Analyze Text**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

**3. Upload File**
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@article.txt"
```

---

## 8. Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Backend Framework** | FastAPI | 0.115.12 | High-performance async API |
| **Web UI Framework** | Flask | 3.1.2 | User interface |
| **ASGI Server** | Uvicorn | 0.34.2 | Production server |
| **Transformer Library** | Hugging Face Transformers | 4.48.3 | BART & RoBERTa models |
| **Deep Learning** | PyTorch | 2.6.0 | Neural network backend |
| **Machine Learning** | Scikit-learn | 1.7.2 | Naive Bayes & TF-IDF |
| **NLP Processing** | NLTK | 3.9.2 | Tokenization & preprocessing |
| **Extractive Summarization** | Sumy | 0.11.0 | TextRank algorithm |
| **HTTP Client** | Requests | 2.32.3 | API communication |
| **Language** | Python | 3.12 | Programming language |

---

## 9. Model Performance

### Summarization
| Method | Speed (CPU) | Quality | Preserves Original |
|--------|-------------|---------|-------------------|
| **Extractive (TextRank)** | ~100ms | Good | ✓ Yes |
| **Abstractive (T5)** | ~3 seconds | Excellent | ✗ Sampling-based Paraphrasing |

### Sentiment Analysis
- **Model**: RoBERTa-base fine-tuned on Twitter sentiment
- **Accuracy**: ~89% on benchmark datasets
- **Inference Time**: ~500ms per text (CPU)
- **Output**: 3 classes (Positive/Negative/Neutral) with all probabilities

### Topic Classification
- **Dataset**: 20 Newsgroups (18,000 documents, 20 categories)
- **Algorithm**: Multinomial Naive Bayes with TF-IDF
- **Test Accuracy**: 65% (13 correct out of 20 on average)
- **Inference Time**: ~100ms per text (CPU)
- **Output**: Top category with all 20 probabilities

**Total Analysis Time**: ~4 seconds per text (CPU) | ~1 second (GPU)

---

## 10. Use Cases

- **Content Analysis**: Analyze articles, blog posts, and reports
- **Social Media Monitoring**: Process tweets, reviews, and comments
- **Research**: Extract themes from academic papers or documents
- **Business Intelligence**: Analyze customer feedback and reviews
- **News Aggregation**: Summarize and categorize news articles
- **Education**: Summarize educational content and identify topics

---

## 11. Future Enhancements

### Short-term (Next 1-3 months)
- [ ] Add BERT-based topic classifier for 90%+ accuracy
- [ ] GPU acceleration support for faster inference
- [ ] Batch processing for multiple files
- [ ] Export results to PDF/CSV/JSON
- [ ] User authentication and analysis history

### Medium-term (3-6 months)
- [ ] Support for additional file formats (PDF, DOCX, HTML)
- [ ] Custom topic model training on user datasets
- [ ] Multilingual support (Spanish, French, German)
- [ ] Real-time streaming analysis
- [ ] Named Entity Recognition (NER)

### Long-term (6-12 months)
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Mobile app (iOS/Android)
- [ ] Advanced visualizations (word clouds, topic networks)
- [ ] Custom model fine-tuning interface
- [ ] Integration with popular CMS platforms
- [ ] Webhook support for automated pipelines

---

## 12. Troubleshooting

### Issue: Models not loading
**Solution**: Models download automatically on first run. Check internet connection.
```bash
# Manually download T5 and RoBERTa
python -c "from transformers import pipeline; pipeline('summarization', model='t5-base'); pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')"
```

### Issue: Port already in use
**Solution**: Stop existing process or use different port:
```bash
# FastAPI on different port
uvicorn src.main:app --host 127.0.0.1 --port 8001

# Flask on different port (edit ui/app.py)
app.run(port=5001)
```

### Issue: Slow inference on CPU
**Solutions**:
- Install GPU-enabled PyTorch for 10x faster transformer inference
- Reduce max_length in T5 summarization
- Use smaller models like t5-small (requires code modification)

### Issue: Topic classification inaccurate
**Expected**: 65% accuracy means 1 in 3 texts may be misclassified. This is normal for 20 categories with traditional ML.
**Solution**: For better accuracy (90%+), implement BERT-based classifier (future enhancement).

### Issue: Out of memory errors
**Solution**: 
- Close other applications
- Reduce BART max_length from 1024 to 512
- Use CPU-only mode (default)

---

## 13. Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 14. License

This project is created for educational purposes as part of an academic assignment.

---

## 15. Acknowledgments

- **20 Newsgroups Dataset**: UCI Machine Learning Repository
- **Hugging Face**: Pre-trained T5 and RoBERTa models
- **Scikit-learn**: Machine learning algorithms and datasets
- **FastAPI & Flask**: Web framework communities
- **PyTorch**: Deep learning framework

---

## 16. Contact & Support

- **GitHub**: https://github.com/springboardmentor443m-coder/NarrativeNexus-Dynamic-Text-Analysis
- **Branch**: ArjunPratap_Infosys
- **Email**: arjunbrt1303@gmail.com

---
