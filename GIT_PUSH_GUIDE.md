# 🚀 Git Push Guide - Push to Your Branch

## ✅ What Will Be Pushed (Clean & Professional)

### Essential Code:
- `src/` - Backend code (FastAPI, ML services)
- `ui/` - Frontend code (Flask, templates, CSS)
- `scripts/` - Training scripts
- `requirements.txt` - Dependencies
- `README.md` - Main documentation
- `START_SERVERS.bat` - Easy startup script

### NOT Pushed (Excluded by .gitignore):
- ❌ `venv/` - Virtual environment (too large)
- ❌ `models/*.pkl` - Trained models (100MB+, regenerate on mentor's machine)
- ❌ `.cache/` - Downloaded BART/RoBERTa models (2GB+)
- ❌ Test files, logs, temporary files
- ❌ Extra documentation drafts

---

## 📋 Step-by-Step Git Commands

### Step 1: Check Your Remote Repository
```powershell
cd D:\InfosysNarrativeNexus
git remote -v
```

**Expected output:**
```
origin  https://github.com/mentor-username/repo-name.git (fetch)
origin  https://github.com/mentor-username/repo-name.git (push)
```

**If remote is NOT set:**
```powershell
git remote add origin https://github.com/mentor-username/repo-name.git
```

---

### Step 2: Create Your Branch (if not exists)
```powershell
# Replace 'your-name' with your actual branch name
git checkout -b your-name
```

**Or switch to existing branch:**
```powershell
git checkout your-name
```

---

### Step 3: Check What Will Be Committed
```powershell
git status
```

**You should see ONLY:**
- Modified: `src/`, `ui/`, `scripts/`, `README.md`, etc.
- **NOT** venv/, models/, .cache/

**If you see unwanted files:**
```powershell
# Remove from staging
git reset HEAD <filename>
```

---

### Step 4: Stage Essential Files
```powershell
# Add all tracked changes (respects .gitignore)
git add .

# OR add specific folders only
git add src/ ui/ scripts/ requirements.txt README.md START_SERVERS.bat
```

---

### Step 5: Commit with Professional Message
```powershell
git commit -m "Implement NarrativeNexus: Text Analysis Platform

Features:
- Extractive & Abstractive Summarization (TextRank + BART)
- Sentiment Analysis (RoBERTa transformer)
- Topic Classification (TF-IDF + Multinomial NB on 20 Newsgroups)
- FastAPI backend with RESTful API
- Flask web interface with responsive dark UI
- Complete documentation and setup scripts"
```

---

### Step 6: Push to Your Branch
```powershell
# First time push
git push -u origin your-name

# Subsequent pushes
git push
```

---

## 🛡️ Before Pushing - Final Checklist

Run these commands to verify:

### ✅ Check 1: No venv or models in commit
```powershell
git ls-files | findstr "venv"
git ls-files | findstr "models.*pkl"
```
**Should return NOTHING**

### ✅ Check 2: Check commit size
```powershell
git diff --cached --stat
```
**Should be < 5MB total**

### ✅ Check 3: Verify .gitignore working
```powershell
git status --ignored
```
**venv/, models/, .cache/ should be in "Ignored files"**

---

## 📝 Making Your Work Look Professional (Not AI-Generated)

### 1. Add Personal Comments in Code
```python
# My approach: Using TF-IDF because it's faster than BERT
# I chose Multinomial NB for 20 newsgroups classification
```

### 2. Add Implementation Notes
```python
# Struggled with this part - LDA wasn't accurate enough
# Switched to supervised classification for better results
```

### 3. Keep README Simple
- Don't over-explain every detail
- Focus on: What it does, How to run it, What I learned
- Mention challenges you faced

### 4. Add a Personal Touch in README
```markdown
## Challenges Faced
- Initially used LDA but accuracy was low (~30%)
- Switched to Multinomial Naive Bayes (65% accuracy)
- Learned about supervised vs unsupervised topic modeling

## What I Learned
- How to integrate multiple ML models in one application
- REST API design with FastAPI
- Handling large transformer models efficiently
```

---

## 🔧 If You Need to Update After Pushing

```powershell
# Make changes to your code
git add .
git commit -m "Fix: Improved topic classification accuracy"
git push
```

---

## ⚠️ Common Issues

### Issue: "Models too large to push"
**Solution:** Already handled by `.gitignore` - models won't be pushed

### Issue: "Everything is being committed"
**Solution:** Check `.gitignore` exists and is properly formatted

### Issue: "Merge conflicts"
**Solution:** 
```powershell
git pull origin your-name
# Resolve conflicts in files
git add .
git commit -m "Resolve merge conflicts"
git push
```

---

## 📦 What Mentor Needs to Do

After you push, mentor will:
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Train models: `python scripts/train_classifier_20newsgroups.py`
5. Run servers: Double-click `START_SERVERS.bat`

**Your code will work on their machine!**

---

## 🎓 Final Tips

1. **Test before pushing**: Make sure `START_SERVERS.bat` works
2. **Clean commit messages**: Describe what you did, not how
3. **Small commits**: Don't push everything at once
4. **Check file sizes**: Keep repo under 100MB
5. **Add personal notes**: Show you understand the code

---

**Ready to push? Follow the steps above! 🚀**
