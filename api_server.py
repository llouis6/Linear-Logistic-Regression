"""
FastAPI Server for LRC ML Demo
Exposes existing ML functionality through REST API.
Deploy to Render, Fly.io, Railway, etc.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
import uvicorn

from api_lib import run_pipeline, get_dataset_info, get_dataset_cache

# Create FastAPI app
app = FastAPI(
    title="LRC ML Demo API",
    description="From-scratch Linear & Logistic Regression implementation - Backend API",
    version="1.0.0"
)

# CORS middleware - allows frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",          # Local Next.js dev
        "http://localhost:3001",          # Alternative local port
        "https://*.vercel.app",           # Vercel preview deployments
        "https://yourdomain.com",         # Your production domain (update this!)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class RunRequest(BaseModel):
    """Request body for /run endpoint."""
    dataset: Literal["breast_cancer", "wine"] = Field(
        ...,
        description="Dataset to use for training"
    )
    model: Literal["linear", "logistic", "multiclass_logistic"] = Field(
        ...,
        description="Model type to train"
    )
    use_feature_selection: bool = Field(
        default=True,
        description="Whether to apply feature selection (mean threshold)"
    )
    polynomial_degree: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Polynomial degree for feature transformation (1=none, 2=quadratic, 3=cubic)"
    )


class MetricsResponse(BaseModel):
    """Metrics from model evaluation."""
    accuracy: float
    auroc: float


class RunResponse(BaseModel):
    """Response from /run endpoint."""
    success: bool
    metrics: Optional[MetricsResponse] = None
    coefficients: Optional[List] = None
    feature_names: Optional[List[str]] = None
    selected_features_count: Optional[int] = None
    total_features_count: Optional[int] = None
    loss_history: Optional[List[float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    roc_curve: Optional[Dict[str, List[float]]] = None
    training_time_seconds: Optional[float] = None
    convergence_iterations: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    datasets_loaded: List[str]


class DatasetInfo(BaseModel):
    """Information about a dataset."""
    id: str
    name: str
    type: str
    samples: int
    features: int
    classes: Optional[int] = None
    description: str
    target_classes: List[str]


class DatasetsResponse(BaseModel):
    """Response from /datasets endpoint."""
    datasets: List[DatasetInfo]


# Endpoints
@app.get("/", tags=["Info"])
def root():
    """Root endpoint - API information."""
    return {
        "name": "LRC ML Demo API",
        "description": "Backend API for Linear & Logistic Regression from Scratch",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "run_experiment": "/run (POST)",
            "datasets_info": "/datasets",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns status and information about loaded datasets.
    """
    cache = get_dataset_cache()
    loaded_datasets = list(cache._datasets.keys())
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "datasets_loaded": loaded_datasets
    }


@app.get("/datasets", response_model=DatasetsResponse, tags=["Info"])
def list_datasets():
    """
    List available datasets with metadata.
    """
    return get_dataset_info()


@app.post("/run", response_model=RunResponse, tags=["ML"])
async def run_experiment(request: RunRequest):
    """
    Train and evaluate a model with given configuration.
    
    This endpoint:
    1. Loads the specified dataset
    2. Optionally applies feature selection
    3. Optionally applies polynomial feature transformation
    4. Trains the specified model
    5. Evaluates on test set
    6. Returns metrics, coefficients, and visualization data
    
    Returns:
        JSON with metrics, loss history, confusion matrix, ROC data, etc.
    """
    try:
        # Run the pipeline
        result = run_pipeline(request.dict())
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Unknown error occurred')
            )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/warmup", tags=["Health"])
def warmup():
    """
    Warmup endpoint - pre-loads datasets into cache.
    Call this after deployment to ensure fast first request.
    """
    try:
        cache = get_dataset_cache()
        # Try to load both datasets
        cache.get_dataset('breast_cancer')
        cache.get_dataset('wine')
        
        return {
            "status": "warmed_up",
            "datasets_loaded": ["breast_cancer", "wine"]
        }
    except Exception as e:
        return {
            "status": "partial_warmup",
            "error": str(e)
        }


# Main entry point for local development
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting LRC ML Demo API Server")
    print("=" * 80)
    print("\nEndpoints:")
    print("  - Health Check:  http://localhost:8000/health")
    print("  - Run Experiment: http://localhost:8000/run (POST)")
    print("  - Datasets Info: http://localhost:8000/datasets")
    print("  - API Docs:      http://localhost:8000/docs")
    print("\nWarming up datasets...")
    
    # Pre-load datasets
    cache = get_dataset_cache()
    
    print("✅ Server ready!\n")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
