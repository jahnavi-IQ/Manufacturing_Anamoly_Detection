"""
AWS Lambda Handler for Pump Anomaly Detection API
==================================================

This file is the entry point for AWS Lambda.
It receives audio files via API Gateway and returns predictions.

Usage:
    Deploy this file to AWS Lambda
    Set Handler to: lambda_handler.lambda_handler
    Set Runtime to: Python 3.11
"""

import json
import base64
import tempfile
from pathlib import Path
import boto3
import logging
import os

# Import from src (relative imports work in Lambda)
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.inference_engine import get_detector
from src.utils import logger, ModelLoadingError

# Configure logging
logger.setLevel(logging.INFO)

# Global detector instance (persists across Lambda invocations)
detector = None


def init_detector():
    """Initialize detector on first Lambda invocation"""
    global detector
    
    if detector is None:
        try:
            logger.info("🔧 Initializing detector on first invocation...")
            detector = get_detector()
            
            # Validate
            is_valid, issues = detector.validate_setup()
            
            if not is_valid:
                logger.error("❌ Detector validation failed!")
                for issue in issues:
                    logger.error(f"   {issue}")
                raise RuntimeError("Detector validation failed")
            
            logger.info("✅ Detector ready for predictions")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize detector: {e}")
            raise
    
    return detector


def lambda_handler(event, context):
    """
    AWS Lambda handler for /predict endpoint
    
    Args:
        event: API Gateway event containing:
            - body: base64-encoded audio file
            - headers: HTTP headers
        context: Lambda context object
    
    Returns:
        dict: JSON response with prediction results
    """
    
    logger.info("=" * 80)
    logger.info("🎯 LAMBDA INVOCATION STARTED")
    logger.info("=" * 80)
    
    try:
        # Step 1: Initialize detector
        logger.info("Step 1/4: Initializing detector...")
        detector = init_detector()
        
        # Step 2: Parse request
        logger.info("Step 2/4: Parsing request...")
        
        # Extract audio data from request
        if not event.get('body'):
            return error_response(400, "No audio file provided in request body")
        
        # Handle base64-encoded or binary audio
        try:
            if isinstance(event['body'], str):
                audio_data = base64.b64decode(event['body'])
            else:
                audio_data = event['body']
        except Exception as e:
            logger.error(f"Failed to decode audio: {e}")
            return error_response(400, "Invalid audio format. Expected base64-encoded WAV file.")
        
        # Validate file size
        file_size_mb = len(audio_data) / (1024 * 1024)
        if file_size_mb > 10:  # 10 MB limit
            logger.error(f"File too large: {file_size_mb:.2f}MB")
            return error_response(413, f"File too large: {file_size_mb:.2f}MB (max: 10MB)")
        
        # Step 3: Save to temporary file and predict
        logger.info("Step 3/4: Processing audio...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        try:
            # Make prediction
            result = detector.predict_from_file(tmp_path)
            
            logger.info(f"✅ Prediction complete: {result['prediction']}")
            
            # Step 4: Format response
            logger.info("Step 4/4: Formatting response...")
            
            return success_response(result)
            
        finally:
            # Clean up temporary file
            try:
                Path(tmp_path).unlink()
                logger.debug(f"Cleaned up temp file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")
    
    except ModelLoadingError as e:
        logger.error(f"❌ Model error: {e}")
        return error_response(503, f"Model initialization failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        return error_response(500, f"Internal server error: {str(e)}")
    
    finally:
        logger.info("=" * 80)
        logger.info("🏁 LAMBDA INVOCATION COMPLETED")
        logger.info("=" * 80)


def health_handler(event, context):
    """
    Health check endpoint - called by GET /health
    """
    try:
        logger.info("🏥 Health check requested")
        
        detector = init_detector()
        is_valid, issues = detector.validate_setup()
        
        if is_valid:
            logger.info("✅ Health check: HEALTHY")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'healthy',
                    'detector_loaded': True,
                    'message': 'Pump anomaly detector is ready'
                }),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            }
        else:
            logger.warning(f"⚠️ Health check: DEGRADED - {issues}")
            return {
                'statusCode': 503,
                'body': json.dumps({
                    'status': 'degraded',
                    'detector_loaded': True,
                    'issues': issues
                }),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            }
    
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            'statusCode': 503,
            'body': json.dumps({
                'status': 'unhealthy',
                'error': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }


def model_info_handler(event, context):
    """
    Model info endpoint - called by GET /model-info
    """
    try:
        logger.info("📋 Model info requested")
        
        detector = init_detector()
        model_info = detector.get_model_info()
        
        return {
            'statusCode': 200,
            'body': json.dumps(model_info, indent=2),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to get model info: {e}")
        return error_response(500, f"Failed to get model info: {str(e)}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def success_response(data):
    """Format successful response"""
    return {
        'statusCode': 200,
        'body': json.dumps(data, indent=2, default=str),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }
    }


def error_response(status_code, message):
    """Format error response"""
    return {
        'statusCode': status_code,
        'body': json.dumps({
            'error': True,
            'message': message,
            'status_code': status_code
        }),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }
    }


# For local testing
if __name__ == "__main__":
    print("Testing Lambda handler locally...")
    
    # Simulate API Gateway event
    test_event = {
        'body': base64.b64encode(open('test_audio.wav', 'rb').read()).decode()
    }
    
    context = None
    response = lambda_handler(test_event, context)
    
    print(json.dumps(response, indent=2))