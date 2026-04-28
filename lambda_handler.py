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
            logger.info("Initializing detector on first invocation...")
            detector = get_detector()

            # Validate
            is_valid, issues = detector.validate_setup()

            if not is_valid:
                logger.error("Detector validation failed!")
                for issue in issues:
                    logger.error(f"   {issue}")
                raise RuntimeError("Detector validation failed")

            logger.info("Detector ready for predictions")

        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise

    return detector


# ============================================================================
# MULTIPART PARSER  <-- NEW FUNCTION ADDED
# ============================================================================

def _extract_from_multipart(raw_bytes, content_type):
    """
    Extract WAV audio bytes from a multipart/form-data body.
    Pure Python - no external libraries needed.
    """
    try:
        # Extract boundary from Content-Type
        # e.g. "multipart/form-data; boundary=----WebKitFormBoundaryXYZ"
        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[len("boundary="):].strip().strip('"')
                break

        if not boundary:
            logger.error("Could not find multipart boundary")
            return None

        boundary_bytes = ("--" + boundary).encode()
        parts = raw_bytes.split(boundary_bytes)

        for part in parts:
            if b"Content-Disposition" not in part:
                continue

            # Find the double CRLF separating headers from body
            sep = b"\r\n\r\n"
            idx = part.find(sep)
            if idx == -1:
                sep = b"\n\n"
                idx = part.find(sep)
            if idx == -1:
                continue

            headers_raw = part[:idx].lower()
            body_part = part[idx + len(sep):]

            # Strip trailing boundary markers
            if body_part.endswith(b"\r\n"):
                body_part = body_part[:-2]
            if body_part.endswith(b"--"):
                body_part = body_part[:-2]

            # Return if this part is a file/audio field
            if b"filename" in headers_raw or b"audio" in headers_raw or b"wav" in headers_raw:
                logger.info(f"Extracted audio from multipart: {len(body_part)} bytes")
                return body_part

        logger.error("No audio file field found in multipart body")
        return None

    except Exception as e:
        logger.error(f"Multipart parse error: {e}")
        return None


# ============================================================================
# MAIN HANDLER
# ============================================================================

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
    logger.info("LAMBDA INVOCATION STARTED")
    logger.info("=" * 80)

    try:
        # ── CORS preflight  <-- NEW: handles browser OPTIONS request
        if event.get("httpMethod") == "OPTIONS":
            return _cors_preflight_response()

        # Step 1: Initialize detector
        logger.info("Step 1/4: Initializing detector...")
        detector = init_detector()

        # Step 2: Parse request
        logger.info("Step 2/4: Parsing request...")

        if not event.get("body"):
            return error_response(400, "No audio file provided in request body")

        body = event.get("body", "")
        is_b64 = event.get("isBase64Encoded", False)

        # Read Content-Type header (case-insensitive)  <-- NEW: detect multipart
        content_type = ""
        for k in event.get("headers", {}):
            if k.lower() == "content-type":
                content_type = event["headers"][k]
                break
        logger.info(f"Content-Type: {content_type}")
        logger.info(f"isBase64Encoded: {is_b64}")

        # Decode body to raw bytes  <-- NEW: handles all 3 cases correctly
        try:
            if is_b64:
                # API Gateway base64-encoded a binary body for us
                raw_bytes = base64.b64decode(body)
            elif isinstance(body, str):
                # Plain string body — treat as base64 (old app.py path / fallback)
                raw_bytes = base64.b64decode(body)
            else:
                raw_bytes = body
        except Exception as e:
            logger.error(f"Failed to decode body: {e}")
            return error_response(400, "Could not decode request body. Expected base64-encoded WAV.")

        # Extract actual WAV bytes  <-- NEW: strips multipart wrapper if present
        if "multipart/form-data" in content_type:
            audio_data = _extract_from_multipart(raw_bytes, content_type)
            if audio_data is None:
                logger.warning("Multipart parse failed, falling back to raw bytes")
                audio_data = raw_bytes
        else:
            audio_data = raw_bytes

        if not audio_data:
            return error_response(400, "Could not extract audio data from request")

        # Validate file size
        file_size_mb = len(audio_data) / (1024 * 1024)
        logger.info(f"Audio size: {file_size_mb:.2f} MB")
        if file_size_mb > 10:
            logger.error(f"File too large: {file_size_mb:.2f}MB")
            return error_response(413, f"File too large: {file_size_mb:.2f}MB (max: 10MB)")

        # Step 3: Save to temporary file and predict
        logger.info("Step 3/4: Processing audio...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            # Make prediction
            result = detector.predict_from_file(tmp_path)

            logger.info(f"Prediction complete: {result['prediction']}")

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
        logger.error(f"Model error: {e}")
        return error_response(503, f"Model initialization failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return error_response(500, f"Internal server error: {str(e)}")

    finally:
        logger.info("=" * 80)
        logger.info("LAMBDA INVOCATION COMPLETED")
        logger.info("=" * 80)


def health_handler(event, context):
    """
    Health check endpoint - called by GET /health
    """
    try:
        if event.get("httpMethod") == "OPTIONS":  # <-- NEW
            return _cors_preflight_response()

        logger.info("Health check requested")

        detector = init_detector()
        is_valid, issues = detector.validate_setup()

        if is_valid:
            logger.info("Health check: HEALTHY")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "status": "healthy",
                    "detector_loaded": True,
                    "message": "Pump anomaly detector is ready"
                }),
                "headers": _cors_headers()
            }
        else:
            logger.warning(f"Health check: DEGRADED - {issues}")
            return {
                "statusCode": 503,
                "body": json.dumps({
                    "status": "degraded",
                    "detector_loaded": True,
                    "issues": issues
                }),
                "headers": _cors_headers()
            }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "statusCode": 503,
            "body": json.dumps({
                "status": "unhealthy",
                "error": str(e)
            }),
            "headers": _cors_headers()
        }


def model_info_handler(event, context):
    """
    Model info endpoint - called by GET /model-info
    """
    try:
        if event.get("httpMethod") == "OPTIONS":  # <-- NEW
            return _cors_preflight_response()

        logger.info("Model info requested")

        detector = init_detector()
        model_info = detector.get_model_info()

        return {
            "statusCode": 200,
            "body": json.dumps(model_info, indent=2),
            "headers": _cors_headers()
        }

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return error_response(500, f"Failed to get model info: {str(e)}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _cors_headers():
    """Centralised CORS headers used by every response"""  # <-- NEW helper
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Filename"
    }


def _cors_preflight_response():
    """Handle OPTIONS preflight from browser"""  # <-- NEW helper
    return {
        "statusCode": 200,
        "body": "",
        "headers": _cors_headers()
    }


def success_response(data):
    """Format successful response"""
    return {
        "statusCode": 200,
        "body": json.dumps(data, indent=2, default=str),
        "headers": _cors_headers()
    }


def error_response(status_code, message):
    """Format error response"""
    return {
        "statusCode": status_code,
        "body": json.dumps({
            "error": True,
            "message": message,
            "status_code": status_code
        }),
        "headers": _cors_headers()
    }


# For local testing
if __name__ == "__main__":
    print("Testing Lambda handler locally...")

    test_event = {
        "body": base64.b64encode(open("test_audio.wav", "rb").read()).decode(),
        "isBase64Encoded": False,
        "httpMethod": "POST",
        "headers": {"content-type": "text/plain"}
    }

    context = None
    response = lambda_handler(test_event, context)
    print(json.dumps(response, indent=2))
