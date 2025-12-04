# 🚀 Deploy NarrativeNexus to Vercel

## Step-by-Step Deployment

### 1. Install Vercel CLI (One-time Setup)
```bash
# Install globally
npm install -g vercel

# Or use without installing
npx vercel
```

### 2. Login to Vercel
```bash
vercel login
```
**Enter your email** when prompted

### 3. Deploy from Project Root
```bash
cd D:\InfosysNarrativeNexus

# First deployment
vercel

# Follow prompts:
# - Set up and deploy? Y
# - Which scope? (select your account)
# - Link to existing project? N
# - Project name? narrative-nexus (or your choice)
# - Directory? ./ (press Enter)
# - Override settings? N
```

### 4. Production Deployment
```bash
vercel --prod
```

### 5. Get Your Live URL
After deployment, Vercel will give you a URL like:
```
https://narrative-nexus-abc123.vercel.app
```

---

## ⚠️ IMPORTANT: Vercel Limitations

### What WON'T Work on Free Vercel:
- ❌ **BART Model** (406MB - too large, 50MB limit)
- ❌ **RoBERTa Model** (125MB - too large)
- ❌ **Flask UI** (Vercel is for API only)
- ❌ **Model files** (Git LFS or cloud storage needed)

### What WILL Work:
- ✅ **FastAPI Backend** (API endpoints only)
- ✅ **TF-IDF + Naive Bayes** (small models, ~500KB)
- ✅ **Extractive summarization** (no large models)

---

## 🎯 Recommended Deployment Strategy

### Option A: Deploy to Render/Railway (RECOMMENDED)
**Why?**: No size limits, can run full Flask + FastAPI + all models

**Steps for Render.com**:
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect your GitHub repo
5. Select branch: `ArjunPratap_Infosys`
6. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.main:app --host 0.0.0.0 --port 8000`
   - **Plan**: Free (512MB RAM)
7. Click "Create Web Service"
8. Wait 5-10 minutes for deployment
9. Get your URL: `https://narrative-nexus.onrender.com`

**For Flask UI (Separate Service)**:
1. Create another web service
2. **Start Command**: `python ui/app.py`
3. Environment variable: `FASTAPI_URL=https://your-backend.onrender.com`

### Option B: Deploy to Hugging Face Spaces (BEST FOR DEMOS)
**Why?**: Free, supports large models, interactive demos

**Steps**:
1. Go to https://huggingface.co/spaces
2. Create new Space
3. Select "Gradio" or "Streamlit"
4. Upload your code
5. Models cache automatically
6. Get shareable link

### Option C: Keep Vercel (API Only)
**If you want Vercel anyway** (limited functionality):

1. Remove large model dependencies
2. Deploy API-only version
3. Use lightweight models only
4. Frontend hosted elsewhere (Netlify/Vercel Pages)

---

## 📝 After Deployment

Once deployed, add to README:

```markdown
## 🌐 Live Demo

[![Deploy](https://img.shields.io/badge/Live-Demo-blue)](https://your-app.onrender.com)

**Try it now**: [https://your-app.onrender.com](https://your-app.onrender.com)
```

---

## 🔧 Quick Deploy Commands

### Render.com (RECOMMENDED):
```bash
# Already on GitHub - just connect via Render dashboard
# No CLI needed!
```

### Vercel (Limited):
```bash
vercel --prod
```

### Hugging Face Spaces:
```bash
# Use web interface to upload
# Or git push to HF repo
```

Choose **Render.com** for the easiest full deployment! 🎉
