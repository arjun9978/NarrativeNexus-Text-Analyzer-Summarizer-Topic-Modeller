"""
Flask UI for NarrativeNexus
Provides web interface for text analysis
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'narrativenexus-secret-key-2024'  # Change in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'txt'}

# FastAPI backend URL
FASTAPI_URL = "http://127.0.0.1:8000"

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle text analysis request"""
    try:
        # Check if text input or file upload
        text_input = request.form.get('text')
        file = request.files.get('file')
        
        if not text_input and not file:
            flash('Please provide text or upload a file', 'error')
            return redirect(url_for('index'))
        
        # Handle file upload
        if file and file.filename:
            if not allowed_file(file.filename):
                flash('Only .txt files are allowed', 'error')
                return redirect(url_for('index'))
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                text_input = f.read()
            
            # Clean up uploaded file
            os.remove(filepath)
        
        if not text_input or len(text_input.strip()) == 0:
            flash('Text cannot be empty', 'error')
            return redirect(url_for('index'))
        
        # Call FastAPI backend
        response = requests.post(
            f"{FASTAPI_URL}/analyze",
            json={"text": text_input},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return render_template('results.html', 
                                 text=text_input,
                                 result=result)
        else:
            flash(f'Error from API: {response.text}', 'error')
            return redirect(url_for('index'))
            
    except requests.exceptions.ConnectionError:
        flash('Cannot connect to analysis server. Make sure FastAPI is running on port 8000.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/health')
def health():
    """Check health of both Flask UI and FastAPI backend"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        backend_status = response.json() if response.status_code == 200 else {"status": "error"}
    except:
        backend_status = {"status": "offline"}
    
    return jsonify({
        "ui": "ok",
        "backend": backend_status
    })

if __name__ == '__main__':
    print("="*60)
    print("NarrativeNexus UI Starting...")
    print("="*60)
    print(f"UI URL: http://127.0.0.1:5000")
    print(f"FastAPI Backend: {FASTAPI_URL}")
    print("="*60)
    print("⚠️  Make sure FastAPI server is running on port 8000!")
    print("   Start it with: uvicorn src.main:app --host 127.0.0.1 --port 8000")
    print("="*60)
    
    app.run(host='127.0.0.1', port=5000, debug=True)
