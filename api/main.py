"""
FastAPI Backend for Pump Anomaly Detection
===========================================

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import json
import logging
from datetime import datetime
import uvicorn
from src.config import config
from src.inference_engine import get_detector
from src.utils import validate_audio_format, handle_error, logger

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)

# Global detector instance 
detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown
    """
    global detector
    
    # Startup
    try:
        # Load detector
        detector = get_detector()
        
        if detector is None:
            raise RuntimeError("Detector failed to initialize (detector is None)")
        
        # Validate
        is_valid, issues = detector.validate_setup()
        
        if not is_valid:
            logger.error("Detector validation failed!")
            for issue in issues:
                logger.error(f"  - {issue}")
            raise RuntimeError("Detector validation failed")
        
        if issues:  # Warnings only
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        logger.info("✅ API server ready")
        logger.info(f"   Listening on: http://{config.API_HOST}:{config.API_PORT}")
        
    
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield  # Server is running
    
    # Shutdown
    logger.info("API server shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Pump Anomaly Detection API",
    description="REST API for pump sound anomaly detection with explainability",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOW_ORIGINS,
    allow_credentials=config.ALLOW_CREDENTIALS,
    allow_methods=config.ALLOW_METHODS,
    allow_headers=config.ALLOW_HEADERS
)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint
    """
    return {
        "name": "Pump Anomaly Detection API",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "model_info": "/model-info",
            "training_report": "/training-report",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status of the API
    """
    try:
        # Check if detector is loaded
        if detector is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "message": "Detector not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Validate detector
        is_valid, issues = detector.validate_setup()
        
        return {
            "status": "healthy" if is_valid else "degraded",
            "detector_loaded": True,
            "validation_passed": is_valid,
            "issues": issues if issues else None,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=handle_error(e, "Health check")
        )


@app.post("/predict", tags=["Prediction"])
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict pump anomaly from uploaded audio file
    """
    start_time = datetime.now()

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Check file extension
    if not validate_audio_format(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Expected: {config.ALLOWED_EXTENSIONS}"
        )
    
    # Check file size
    file_size_mb = 0
    temp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file_path = temp_file.name
            
            # Read and save file
            content = await file.read()
            file_size_mb = len(content) / (1024 * 1024)
            
            # Check size limit
            if file_size_mb > config.MAX_UPLOAD_SIZE_MB:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large: {file_size_mb:.2f}MB (max: {config.MAX_UPLOAD_SIZE_MB}MB)"
                )
            
            temp_file.write(content)
        
        logger.info(f"Processing upload: {file.filename} ({file_size_mb:.2f}MB)")
        
        # Make prediction
        if detector is None:
            raise HTTPException(
                status_code=503,
                detail="Detector is not initialized (detector is None)"
            )
        
        result = detector.predict_from_file(temp_file_path)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add metadata
        result['api_metadata'] = {
            'processing_time_seconds': processing_time,
            'file_size_mb': file_size_mb,
            'timestamp': start_time.isoformat()
        }
        
        logger.info(f"Prediction completed in {processing_time:.2f}s: {result['prediction']}")
        
        return result
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=handle_error(e, "Prediction")
        )
    
    finally:
        # Clean up temporary file
        if temp_file_path and Path(temp_file_path).exists():
            try:
                Path(temp_file_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """
    Get model information and metadata
    
    Returns:
        Model metadata and configuration
    """
    try:
        if detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Detector not initialized"
            )
        
        model_info = detector.get_model_info()
        return model_info
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=handle_error(e, "Model info retrieval")
        )


@app.get("/training-report", tags=["Model"])
async def get_training_report():
    """
    Get complete training report
    
    Returns:
        Training report JSON
    """
    try:
        report_path = config.TRAINING_REPORT_PATH
        
        if not report_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training report not found: {report_path}"
            )
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        return report
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to load training report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=handle_error(e, "Training report retrieval")
        )


@app.get("/config", tags=["Configuration"])
async def get_config():
    """
    Get API configuration
    
    Returns:
        Current configuration
    """
    return {
        "api_version": "1.0",
        "max_upload_size_mb": config.MAX_UPLOAD_SIZE_MB,
        "allowed_extensions": list(config.ALLOWED_EXTENSIONS),
        "expected_features": config.EXPECTED_N_FEATURES,
        "sample_rate": config.SAMPLE_RATE or "native",
        "class_names": config.CLASS_NAMES
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": True,
            "message": "Endpoint not found",
            "path": str(request.url)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "detail": str(exc)
        }
    )


# Run with: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
if __name__ == "__main__":
    
    print("=" * 80)
    print("Starting Pump Anomaly Detection API Server")
    print("=" * 80)
    print(f"Docs: http://localhost:{config.API_PORT}/docs")
    print(f"Health: http://localhost:{config.API_PORT}/health")
    print("=" * 80)
    print("\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower()
    )