"""
Sample Dataset Preparation Script

This script helps you prepare a corpus for LDA topic modeling.
Modify according to your dataset format.
"""
import os
import re
import pandas as pd
from pathlib import Path

def clean_text(text: str) -> str:
    """
    Basic text cleaning
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters (keep only letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lowercase
    text = text.lower()
    
    return text

def prepare_corpus_from_csv(csv_file: str, text_column: str, output_file: str):
    """
    Prepare corpus from CSV file
    
    Args:
        csv_file: Path to input CSV
        text_column: Name of column containing text
        output_file: Path to save processed corpus
    """
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Found {len(df)} documents")
    print(f"Processing column: {text_column}")
    
    # Clean texts
    documents = []
    for idx, text in enumerate(df[text_column]):
        if pd.notna(text):
            cleaned = clean_text(str(text))
            if len(cleaned.split()) > 10:  # Filter very short docs
                documents.append(cleaned)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} documents...")
    
    print(f"\n✓ Kept {len(documents)} valid documents")
    
    # Save to file (one document per line)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + '\n')
    
    print(f"✓ Saved to {output_file}")

def prepare_corpus_from_txt_files(directory: str, output_file: str):
    """
    Prepare corpus from directory of .txt files
    """
    print(f"Loading .txt files from {directory}...")
    
    documents = []
    txt_files = list(Path(directory).glob('*.txt'))
    
    print(f"Found {len(txt_files)} .txt files")
    
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            cleaned = clean_text(text)
            if len(cleaned.split()) > 10:
                documents.append(cleaned)
    
    print(f"✓ Processed {len(documents)} documents")
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + '\n')
    
    print(f"✓ Saved to {output_file}")

def create_sample_corpus(output_file: str = "data/processed/corpus.txt"):
    """
    Create a minimal sample corpus for testing
    (REPLACE WITH YOUR REAL DATA)
    """
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data",
        "Natural language processing helps computers understand and generate human language",
        "Deep learning uses neural networks with multiple layers to process complex patterns",
        "Python is a popular programming language for data science and machine learning projects",
        "Climate change poses significant challenges to ecosystems and human societies worldwide",
        "Renewable energy sources like solar and wind are becoming more economically viable",
        "Electric vehicles are transforming the automotive industry and reducing carbon emissions",
        "Healthcare systems are adopting digital technologies to improve patient outcomes",
        "Telemedicine allows remote consultation between doctors and patients using technology",
        "Blockchain technology provides secure and transparent transaction recording systems",
        "Cryptocurrency markets have seen significant volatility in recent years",
        "Financial technology startups are disrupting traditional banking services",
        "Social media platforms influence public opinion and political discourse",
        "Data privacy concerns are growing as companies collect more personal information",
        "Cybersecurity threats are evolving as hackers develop sophisticated attack methods"
    ]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in sample_docs:
            f.write(doc.lower() + '\n')
    
    print(f"✓ Created sample corpus at {output_file}")
    print(f"  {len(sample_docs)} sample documents")
    print("\n⚠️  IMPORTANT: Replace this with your real dataset!")

if __name__ == "__main__":
    # Example 1: Create sample corpus for testing
    create_sample_corpus("data/processed/corpus.txt")
    
    # Example 2: From CSV (uncomment and modify)
    # prepare_corpus_from_csv(
    #     csv_file="data/raw/your_data.csv",
    #     text_column="text",
    #     output_file="data/processed/corpus.txt"
    # )
    
    # Example 3: From directory of .txt files (uncomment and modify)
    # prepare_corpus_from_txt_files(
    #     directory="data/raw/documents/",
    #     output_file="data/processed/corpus.txt"
    # )
