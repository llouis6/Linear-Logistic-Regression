# LRC ML Demo API

Backend API for the Linear & Logistic Regression from Scratch project.

## Quick Start (Local Development)

```bash
# Install dependencies (including API dependencies)
pip install fastapi uvicorn pydantic

# Or install all from requirements.txt
pip install -r requirements.txt

# Run the API server
python api_server.py

# Or with uvicorn directly
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "datasets_loaded": ["breast_cancer", "wine"]
}
```

### GET /datasets
Get information about available datasets.

**Response:**
```json
{
  "datasets": [
    {
      "id": "breast_cancer",
      "name": "Breast Cancer Wisconsin (Diagnostic)",
      "type": "binary",
      "samples": 569,
      "features": 30,
      "description": "...",
      "target_classes": ["Benign", "Malignant"]
    },
    ...
  ]
}
```

### POST /run
Train and evaluate a model.

**Request Body:**
```json
{
  "dataset": "breast_cancer",
  "model": "logistic",
  "use_feature_selection": true,
  "polynomial_degree": 1
}
```

**Parameters:**
- `dataset`: `"breast_cancer"` or `"wine"`
- `model`: `"linear"`, `"logistic"`, or `"multiclass_logistic"`
- `use_feature_selection`: boolean (default: true)
- `polynomial_degree`: 1-3 (default: 1)

**Response:**
```json
{
  "success": true,
  "metrics": {
    "accuracy": 0.9561,
    "auroc": 0.9971
  },
  "coefficients": [...],
  "feature_names": [...],
  "selected_features_count": 15,
  "total_features_count": 30,
  "loss_history": [0.693, 0.421, ...],
  "confusion_matrix": [[35, 2], [3, 74]],
  "roc_curve": {
    "fpr": [0.0, 0.0, 0.054, ...],
    "tpr": [0.0, 0.013, 0.987, ...]
  },
  "training_time_seconds": 0.125,
  "convergence_iterations": 287,
  "config": {...}
}
```

### GET /docs
Interactive API documentation (Swagger UI).

## Deployment

### Option 1: Render.com (Recommended)

1. Push code to GitHub
2. Go to https://dashboard.render.com
3. Click "New +" → "Blueprint"
4. Connect your repo
5. Render will use `render.yaml` for configuration
6. Deploy automatically

**Free tier includes:**
- Automatic HTTPS
- Auto-deploy on git push
- 750 hours/month

### Option 2: Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch (creates fly.toml)
fly launch

# Deploy
fly deploy
```

### Option 3: Railway

1. Go to https://railway.app
2. "New Project" → "Deploy from GitHub repo"
3. Select your LRC repo
4. Railway auto-detects Python and deploys

### Option 4: Docker

```bash
# Build image
docker build -t lrc-api .

# Run container
docker run -p 8000:8000 lrc-api

# Or use docker-compose
docker-compose up
```

## Configuration

### CORS Origins

Update `api_server.py` line ~28 to add your frontend domain:

```python
allow_origins=[
    "http://localhost:3000",
    "https://your-portfolio-domain.com",
    "https://*.vercel.app",
]
```

### Environment Variables

Create `.env` file (see `.env.example`):

```bash
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## Testing the API

### Using curl:

```bash
# Health check
curl http://localhost:8000/health

# Run experiment
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "breast_cancer",
    "model": "logistic",
    "use_feature_selection": true,
    "polynomial_degree": 1
  }'
```

### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/run",
    json={
        "dataset": "breast_cancer",
        "model": "logistic",
        "use_feature_selection": True,
        "polynomial_degree": 1
    }
)

result = response.json()
print(f"Accuracy: {result['metrics']['accuracy']:.2%}")
print(f"AUROC: {result['metrics']['auroc']:.4f}")
```

### Using the interactive docs:

Go to http://localhost:8000/docs and use the "Try it out" feature.

## Performance Notes

- First request may be slower (~2-3s) due to dataset loading
- Subsequent requests are fast (~0.1-0.5s)
- Datasets are cached in memory for quick access
- Polynomial degree 2-3 increases compute time

## Connecting to Frontend

In your Next.js project, set environment variable:

```bash
# .env.local
NEXT_PUBLIC_LRC_API_URL=https://your-api-url.onrender.com
```

Then in your React component:

```typescript
const API_URL = process.env.NEXT_PUBLIC_LRC_API_URL

const response = await fetch(`${API_URL}/run`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(config)
})

const results = await response.json()
```

## Troubleshooting

**Dataset loading fails:**
- Check internet connection (UCI repository download)
- Datasets are cached after first load
- Use `/warmup` endpoint after deployment

**CORS errors:**
- Update `allow_origins` in `api_server.py`
- Ensure frontend domain is whitelisted

**Slow responses:**
- First request loads datasets (normal)
- Consider upgrading from free tier
- Use `/warmup` endpoint proactively

## Project Structure

```
LRC/
├── api_lib.py           # Core API logic, wraps ML code
├── api_server.py        # FastAPI application
├── models/              # ML model implementations
├── preprocessing/       # Dataset loaders
├── utils/               # Evaluation utilities
├── Dockerfile           # Container definition
├── render.yaml          # Render.com config
└── requirements.txt     # Python dependencies
```
