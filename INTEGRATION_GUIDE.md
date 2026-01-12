# Integration Guide: Connecting LRC API to Your Portfolio

This guide shows how to integrate the LRC ML Demo API with your Next.js portfolio site.

## ✅ What's Been Built (Backend - This Repo)

### Core Files Created:
1. **`api_lib.py`** - Pure function interface wrapping your ML code
   - `run_pipeline(config)` - Main training function
   - `get_dataset_info()` - Dataset metadata
   - Returns JSON-serializable data (no print statements)

2. **`api_server.py`** - FastAPI server with endpoints:
   - `GET /health` - Health check
   - `GET /datasets` - Dataset information
   - `POST /run` - Train and evaluate models
   - `GET /docs` - Interactive API documentation

3. **`test_api.py`** - Test script to verify everything works

4. **Deployment Files:**
   - `Dockerfile` - Container definition
   - `render.yaml` - Render.com configuration
   - `.dockerignore` - Files to exclude from Docker builds
   - `API_README.md` - Complete API documentation

### Test Results:
✅ All tests passed!
- Breast Cancer + Logistic: 96.49% accuracy, 0.9987 AUROC (1.6s)
- Wine + Multiclass Logistic: 100% accuracy (0.03s)
- Polynomial features: Working correctly

## 🚀 Quick Start (Local Testing)

### 1. Install API Dependencies

You need to add these packages:

```bash
pip install fastapi uvicorn pydantic
```

Or update your `requirements.txt` to include:
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
```

### 2. Start the API Server

```bash
# Option A: Direct Python
python api_server.py

# Option B: With uvicorn (recommended for development)
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Run an experiment
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "breast_cancer",
    "model": "logistic",
    "use_feature_selection": true,
    "polynomial_degree": 1
  }'
```

Or visit: **http://localhost:8000/docs** for interactive testing

## 📡 API Contract (Frontend ↔ Backend)

### Request Format

```typescript
// POST /run
{
  dataset: 'breast_cancer' | 'wine',
  model: 'linear' | 'logistic' | 'multiclass_logistic',
  use_feature_selection: boolean,
  polynomial_degree: 1 | 2 | 3
}
```

### Response Format

```typescript
{
  success: boolean,
  metrics: {
    accuracy: number,      // 0.0 to 1.0
    auroc: number         // 0.0 to 1.0 (binary only)
  },
  coefficients: number[] | number[][],  // Model weights
  feature_names: string[],              // Selected feature names
  selected_features_count: number,
  total_features_count: number,
  loss_history: number[],               // Training loss per iteration
  confusion_matrix: number[][],         // 2x2 or 3x3 matrix
  roc_curve: {                         // Binary classification only
    fpr: number[],
    tpr: number[]
  } | null,
  training_time_seconds: number,
  convergence_iterations: number | null,
  config: object,                       // Echoed config
  error?: string                        // Only if success=false
}
```

## 🎨 Frontend Implementation (Next.js Side)

### Environment Variable

In your Next.js project, create/update `.env.local`:

```bash
# Local development
NEXT_PUBLIC_LRC_API_URL=http://localhost:8000

# Production (after deployment)
NEXT_PUBLIC_LRC_API_URL=https://your-api.onrender.com
```

### React Component Example

```typescript
'use client'

import { useState } from 'react'

const API_URL = process.env.NEXT_PUBLIC_LRC_API_URL

export default function MLPlayground() {
  const [config, setConfig] = useState({
    dataset: 'breast_cancer',
    model: 'logistic',
    use_feature_selection: true,
    polynomial_degree: 1
  })
  
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  
  const runExperiment = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      
      const data = await response.json()
      
      if (data.success) {
        setResults(data)
      } else {
        console.error('Training failed:', data.error)
      }
    } catch (error) {
      console.error('API call failed:', error)
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div>
      {/* Your UI here */}
      <button onClick={runExperiment} disabled={loading}>
        {loading ? 'Training...' : 'Run Experiment'}
      </button>
      
      {results && (
        <div>
          <p>Accuracy: {(results.metrics.accuracy * 100).toFixed(2)}%</p>
          <p>Training Time: {results.training_time_seconds}s</p>
        </div>
      )}
    </div>
  )
}
```

## 🌐 Deployment

### Option 1: Render.com (Easiest)

1. Push this repo to GitHub
2. Go to https://dashboard.render.com
3. Click "New +" → "Web Service"
4. Connect your GitHub repo
5. Render auto-detects Python and uses `render.yaml`
6. Click "Create Web Service"
7. Wait ~5 minutes for deployment
8. Get your URL: `https://your-service.onrender.com`

**Free tier includes:**
- 750 hours/month free
- Auto-deploy on git push
- Automatic HTTPS
- Spins down after 15 min inactivity (first request slower)

### Option 2: Fly.io

```bash
fly launch
fly deploy
```

### Option 3: Railway

1. Go to railway.app
2. "New Project" → "Deploy from GitHub"
3. Select this repo
4. Auto-deploys

### Important: Update CORS

After deployment, update `api_server.py` line ~28:

```python
allow_origins=[
    "http://localhost:3000",
    "https://your-portfolio-domain.com",  # ADD YOUR DOMAIN
    "https://*.vercel.app",
]
```

## 🧪 Testing Production Deployment

```bash
# Test health
curl https://your-api.onrender.com/health

# Test training
curl -X POST https://your-api.onrender.com/run \
  -H "Content-Type: application/json" \
  -d '{"dataset":"breast_cancer","model":"logistic","use_feature_selection":true,"polynomial_degree":1}'
```

## 📊 Performance Notes

- **First request**: ~2-3 seconds (dataset loading)
- **Subsequent requests**: ~0.1-0.5 seconds (datasets cached)
- **Polynomial degree 2-3**: ~1-2 seconds extra
- **Free tier cold start**: ~10-30 seconds (after inactivity)

## 🔧 Troubleshooting

### CORS Errors
- Update `allow_origins` in `api_server.py` with your frontend URL
- Redeploy the API

### Slow Responses
- First request is always slower (normal)
- Consider paid tier for always-on instances
- Call `/warmup` endpoint after deployment

### Dataset Loading Fails
- Check server logs on Render dashboard
- Datasets download from UCI on first request
- They're cached after that

## 📁 File Structure (This Repo)

```
LRC/
├── api_lib.py              ← Core API logic (you built this)
├── api_server.py           ← FastAPI server (you built this)
├── test_api.py             ← Test script (you built this)
├── API_README.md           ← API documentation
├── INTEGRATION_GUIDE.md    ← This file
├── Dockerfile              ← Container config
├── render.yaml             ← Render deployment config
├── requirements.txt        ← Dependencies (add FastAPI!)
├── models/                 ← Your existing ML models
├── preprocessing/          ← Your existing data loaders
├── utils/                  ← Your existing utilities
├── main.py                 ← Still works! (terminal experiments)
└── run_all_experiments.py  ← Still works! (full suite)
```

## ✅ Verification Checklist

Before going live:

- [ ] API starts locally: `python api_server.py`
- [ ] Health check works: `curl localhost:8000/health`
- [ ] Test script passes: `python test_api.py`
- [ ] Added FastAPI to `requirements.txt`
- [ ] Pushed code to GitHub
- [ ] Deployed to Render/Fly/Railway
- [ ] Updated CORS origins with your domain
- [ ] Tested production endpoint
- [ ] Added API URL to Next.js `.env.local`
- [ ] Frontend can call `/run` successfully

## 🎯 Next Steps (Frontend Side)

In your portfolio repo:
1. Create `components/MLPlayground.tsx`
2. Add dataset selector, model picker, toggles
3. Add visualization components (charts, metrics cards)
4. Integrate into your project page
5. Style to match your portfolio theme

## 📞 Support

If something doesn't work:
1. Check API logs on Render dashboard
2. Test with `/docs` endpoint
3. Verify CORS settings
4. Check `test_api.py` locally

---

**You're all set!** The backend API is ready. Now build your frontend UI to call it! 🚀
