"""
Utility Functions for Pump Anomaly Detection System
====================================================
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from .config import config


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def load_audio(file_path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """
    Load audio file with validation
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (None = native)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    
    Raises:
        ValueError: If audio file is invalid
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    # Check file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check file extension
    if file_path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Expected: {config.ALLOWED_EXTENSIONS}")
    
    try:
        # Load audio
        audio, sample_rate = librosa.load(file_path, sr=sr)
        
        # Validate audio
        if len(audio) == 0:
            raise ValueError("Audio file is empty")
        
        # Check duration
        duration = len(audio) / sample_rate
        
        if duration < config.MIN_AUDIO_LENGTH_SEC:
            raise ValueError(f"Audio too short: {duration:.2f}s (minimum: {config.MIN_AUDIO_LENGTH_SEC}s)")
        
        if duration > config.MAX_AUDIO_LENGTH_SEC:
            logger.warning(f"Audio longer than maximum ({duration:.2f}s > {config.MAX_AUDIO_LENGTH_SEC}s). Truncating.")
            max_samples = int(config.MAX_AUDIO_LENGTH_SEC * sample_rate)
            audio = audio[:max_samples]
        
        logger.info(f"Loaded audio: {file_path.name} | Duration: {duration:.2f}s | SR: {sample_rate}Hz")
        
        return audio, int(sample_rate)
    
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def validate_audio_format(file_path: str | Path) -> bool:
    """
    Validate if file is a supported audio format
    """

    file_path = Path(file_path)
    print(f"  Path object: {file_path}")
    print(f"  Suffix: '{file_path.suffix}'")
    print(f"  Is valid: {file_path.suffix.lower() in config.ALLOWED_EXTENSIONS}")
    print("="*60 + "\n")
    # =========================================
    return file_path.suffix.lower() in config.ALLOWED_EXTENSIONS


def get_audio_properties(audio: np.ndarray, sr: int) -> dict:
    """
    Extract basic audio properties
    
    Args:
        audio: Audio array
        sr: Sample rate
    
    Returns:
        Dictionary of audio properties
    """
    duration = len(audio) / sr
    
    return {
        'duration': duration,
        'sample_rate': sr,
        'n_samples': len(audio),
        'channels': 1 if audio.ndim == 1 else audio.shape[1],
        'max_amplitude': float(np.max(np.abs(audio))),
        'mean_amplitude': float(np.mean(np.abs(audio))),
        'rms': float(np.sqrt(np.mean(audio**2)))
    }


def normalize_features(features: np.ndarray, training_stats: dict) -> np.ndarray:
    """
    Normalize features using training statistics
    
    Args:
        features: Feature array
        training_stats: Dict with 'mean' and 'std' arrays
    
    Returns:
        Normalized features
    """
    mean = training_stats['mean']
    std = training_stats['std']
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    normalized = (features - mean) / std
    return normalized


def calculate_z_scores(features: np.ndarray, training_stats: dict) -> np.ndarray:
    """
    Calculate z-scores for features (how many standard deviations from mean)
    
    Args:
        features: Feature array
        training_stats: Dict with 'mean' and 'std' arrays
    
    Returns:
        Z-scores array
    """
    mean = training_stats['mean']
    std = training_stats['std']
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    z_scores = (features - mean) / std
    return z_scores


def get_confidence_level(confidence: float) -> str:
    """
    Convert confidence score to text level
    
    Args:
        confidence: Confidence score (0-1)
    
    Returns:
        Confidence level string
    """
    if confidence >= config.HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    elif confidence >= config.MEDIUM_CONFIDENCE_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def format_percentage(value: float) -> str:
    """
    Format float as percentage string
    
    Args:
        value: Float value (0-1)
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
    
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def validate_feature_array(features: np.ndarray, expected_n_features: int = 42) -> bool:
    """
    Validate feature array shape and values
    
    Args:
        features: Feature array
        expected_n_features: Expected number of features
    
    Returns:
        True if valid, False otherwise
    """
    # Check shape
    if features.shape[0] != expected_n_features:
        logger.error(f"Invalid feature count: {features.shape[0]} (expected: {expected_n_features})")
        return False
    
    # Check for NaN or Inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.error("Features contain NaN or Inf values")
        return False
    
    return True


def get_severity_color(z_score: float) -> str:
    """
    Get color code based on deviation severity
    
    Args:
        z_score: Z-score value
    
    Returns:
        Color code string
    """
    abs_z = abs(z_score)
    
    if abs_z >= config.SEVERE_DEVIATION_THRESHOLD:
        return 'red'  # Severe deviation
    elif abs_z >= config.MODERATE_DEVIATION_THRESHOLD:
        return 'orange'  # Moderate deviation
    else:
        return 'green'  # Normal range


def get_severity_label(z_score: float) -> str:
    """
    Get severity label based on deviation
    
    Args:
        z_score: Z-score value
    
    Returns:
        Severity label string
    """
    abs_z = abs(z_score)
    
    if abs_z >= config.SEVERE_DEVIATION_THRESHOLD:
        return 'Severe'
    elif abs_z >= config.MODERATE_DEVIATION_THRESHOLD:
        return 'Moderate'
    else:
        return 'Normal'


def format_feature_name(feature_name: str) -> str:
    """
    Format feature name for display
    
    Args:
        feature_name: Raw feature name
    
    Returns:
        Formatted feature name
    """
    # Replace underscores with spaces and capitalize
    formatted = feature_name.replace('_', ' ').title()
    return formatted


def truncate_string(s: str, max_length: int = 50) -> str:
    """
    Truncate string to maximum length
    
    Args:
        s: Input string
        max_length: Maximum length
    
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length-3] + "..."


def create_summary_stats(values: np.ndarray) -> dict:
    """
    Create summary statistics for array
    
    Args:
        values: Numpy array
    
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75))
    }


def ensure_directory(directory: Path) -> None:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Path to directory
    """
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


class AudioValidationError(Exception):
    """Custom exception for audio validation errors"""
    pass


class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors"""
    pass


class ModelLoadingError(Exception):
    """Custom exception for model loading errors"""
    pass


def handle_error(error: Exception, context: str = "") -> dict:
    """
    Format error for API response
    
    Args:
        error: Exception object
        context: Additional context
    
    Returns:
        Error dictionary
    """
    error_msg = str(error)
    
    if context:
        error_msg = f"{context}: {error_msg}"
    
    logger.error(error_msg)
    
    return {
        'error': True,
        'message': error_msg,
        'type': type(error).__name__
    }


# Example usage
if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test z-score calculation
    test_features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_stats = {
        'mean': np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        'std': np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    }
    
    z_scores = calculate_z_scores(test_features, test_stats)
    print(f"Z-scores: {z_scores}")
    
    # Test confidence level
    print(f"Confidence 0.95: {get_confidence_level(0.95)}")
    print(f"Confidence 0.70: {get_confidence_level(0.70)}")
    print(f"Confidence 0.50: {get_confidence_level(0.50)}")
    
    # Test severity
    print(f"Z-score 2.5 color: {get_severity_color(2.5)}")
    print(f"Z-score 1.5 color: {get_severity_color(1.5)}")
    print(f"Z-score 0.5 color: {get_severity_color(0.5)}")
    
    print("✅ Utility functions working!")