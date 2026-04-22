"""
Pump Anomaly Detection - Source Module
"""

from .config import config, FEATURE_NAMES, FEATURE_DESCRIPTIONS
from .feature_extractor import extract_all_features
from .inference_engine import PumpAnomalyDetector, get_detector
from .explainability import ExplainabilityEngine
from .utils import (
    load_audio,
    validate_audio_format,
    get_audio_properties,
    logger,
    ModelLoadingError
)

__version__ = "1.0"
__all__ = [
    'config',
    'FEATURE_NAMES',
    'FEATURE_DESCRIPTIONS',
    'extract_all_features',
    'PumpAnomalyDetector',
    'get_detector',
    'ExplainabilityEngine',
    'load_audio',
    'validate_audio_format',
    'get_audio_properties',
    'logger',
    'ModelLoadingError'
]