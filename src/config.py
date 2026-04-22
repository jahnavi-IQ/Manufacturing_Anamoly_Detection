"""
Configuration Management for Pump Anomaly Detection System
===========================================================

Centralized configuration for all system components.
Modify paths and parameters here to adapt to your environment.
"""

import os
from pathlib import Path


class Config:
    """Centralized configuration for the entire system"""
    
    # ========================================================================
    # DIRECTORY PATHS
    # ========================================================================
    
    # Get config.py location (in src/)
    _CONFIG_DIR = Path(__file__).parent.absolute()

    # Project root directory 
    ROOT_DIR = _CONFIG_DIR.parent
    
    # Model artifacts directory
    MODELS_DIR = ROOT_DIR / "models_ml"
    
    # Model files
    MODEL_PATH = MODELS_DIR / "pump_xgb_model.json"
    RESULTS_PATH = MODELS_DIR / "pump_xgb_results.pkl"
    TRAINING_REPORT_PATH = MODELS_DIR / "training_report.json"
    TRAINING_STATS_PATH = MODELS_DIR / "training_statistics.pkl"
    
    # Data directory (for training_stats_generator.py)
    DATA_DIR = ROOT_DIR / "6_dB_pump" / "pump"
    PUMP_IDS = ["id_00", "id_02", "id_04", "id_06"]
    
    # ========================================================================
    # AUDIO PROCESSING PARAMETERS
    # ========================================================================
    
    # Audio loading
    SAMPLE_RATE = None  # None = use native sample rate
    MAX_AUDIO_LENGTH_SEC = 10  # Maximum audio file length in seconds
    MIN_AUDIO_LENGTH_SEC = 0.5  # Minimum audio file length in seconds
    
    # Feature extraction parameters (MUST match pump_ml.py)
    N_MFCC = 13
    
    # Feature extraction flags (MUST match pump_ml.py exactly)
    EXTRACT_MFCCS = True
    EXTRACT_SPECTRAL = True
    EXTRACT_TEMPORAL = True
    EXTRACT_ADVANCED = True
    EXTRACT_NEW_FEATURES = True
    
    # Expected number of features
    EXPECTED_N_FEATURES = 42
    
    # ========================================================================
    # MODEL PARAMETERS
    # ========================================================================
    
    # Classification
    CLASS_NAMES = ['Normal', 'Abnormal']
    DEFAULT_THRESHOLD = 0.5  # Will be overridden by optimal threshold from training
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.80
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60
    
    # ========================================================================
    # API CONFIGURATION
    # ========================================================================
    
    # FastAPI settings
    API_HOST = "0.0.0.0"
    API_PORT = 8001
    API_RELOAD = True  # Set to False in production
    
    # CORS settings
    ALLOW_ORIGINS = ["*"]  # Restrict in production
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]
    ALLOW_HEADERS = ["*"]
    
    # File upload settings
    MAX_UPLOAD_SIZE_MB = 10
    ALLOWED_EXTENSIONS = {".wav"}
    
    # Request limits
    MAX_REQUESTS_PER_MINUTE = 60
    
    # ========================================================================
    # STREAMLIT UI CONFIGURATION
    # ========================================================================
    
    # Streamlit settings
    UI_PORT = 8501
    UI_TITLE = "🔧 Pump Anomaly Detection System"
    UI_ICON = "🔧"
    
    # API endpoint (for Streamlit to call)
    API_URL = f"http://localhost:{API_PORT}"
    
    # Visualization settings
    FIGURE_HEIGHT = 400
    FIGURE_WIDTH = 800
    
    # ========================================================================
    # EXPLAINABILITY CONFIGURATION
    # ========================================================================
    
    # Feature deviation thresholds (in standard deviations)
    SEVERE_DEVIATION_THRESHOLD = 2.0  # Red alert
    MODERATE_DEVIATION_THRESHOLD = 1.0  # Orange warning
    # < 1.0 is considered normal (green)
    
    # Number of features to show in explanations
    TOP_N_FEATURES = 10
    
    # Similar examples (if using similarity-based explainability)
    N_SIMILAR_EXAMPLES = 3
    
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    @classmethod
    def validate(cls):
        """Validate that all required files and directories exist"""
        errors = []
        
        # Check model files
        if not cls.MODEL_PATH.exists():
            errors.append(f"Model file not found: {cls.MODEL_PATH}")
        
        if not cls.RESULTS_PATH.exists():
            errors.append(f"Results file not found: {cls.RESULTS_PATH}")
        
        if not cls.TRAINING_REPORT_PATH.exists():
            errors.append(f"Training report not found: {cls.TRAINING_REPORT_PATH}")
        
        # Training stats is optional (will be generated if missing)
        if not cls.TRAINING_STATS_PATH.exists():
            errors.append(f"WARNING: Training statistics not found: {cls.TRAINING_STATS_PATH}")
            errors.append("Run scripts/training_stats_generator.py to generate it")
        
        return errors
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 80)
        print("PUMP ANOMALY DETECTION - CONFIGURATION")
        print("=" * 80)
        print(f"\nPaths:")
        print(f"  Model:          {cls.MODEL_PATH}")
        print(f"  Results:        {cls.RESULTS_PATH}")
        print(f"  Training Stats: {cls.TRAINING_STATS_PATH}")
        print(f"\nAudio Parameters:")
        print(f"  Sample Rate:    {cls.SAMPLE_RATE or 'Native'}")
        print(f"  N_MFCC:         {cls.N_MFCC}")
        print(f"  Expected Features: {cls.EXPECTED_N_FEATURES}")
        print(f"\nAPI Configuration:")
        print(f"  Host:           {cls.API_HOST}")
        print(f"  Port:           {cls.API_PORT}")
        print(f"  URL:            http://localhost:{cls.API_PORT}")
        print(f"\nUI Configuration:")
        print(f"  Port:           {cls.UI_PORT}")
        print(f"  URL:            http://localhost:{cls.UI_PORT}")
        print("=" * 80)


# Feature names (matches pump_ml.py order - DO NOT MODIFY ORDER)
FEATURE_NAMES = [
    # MFCCs (13 features)
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7',
    'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13',
    
    # Spectral features (6 features)
    'spectral_centroid', 'spectral_rolloff', 'spectral_contrast',
    'spectral_bandwidth', 'spectral_flatness', 'spectral_flux',
    
    # Temporal features (3 features)
    'zero_crossing_rate', 'rms_energy', 'tempo',
    
    # Advanced features (10 features)
    'chroma_stft_mean', 'chroma_stft_std', 'chroma_cqt_mean', 'chroma_cqt_std',
    'tonnetz_mean', 'tonnetz_std', 'poly_features_mean', 'poly_features_std',
    'kurtosis', 'skewness',
    
    # New features (10 features)
    'mel_mean', 'mel_std', 'contrast_mean', 'contrast_std',
    'zcr_mean', 'zcr_std', 'rolloff_mean', 'rolloff_std',
    'bandwidth_mean', 'bandwidth_std'
]


# Feature descriptions for explainability
FEATURE_DESCRIPTIONS = {
    # MFCCs
    'mfcc_1': 'Overall spectral shape',
    'mfcc_2': 'Spectral detail level 2',
    'mfcc_3': 'Spectral detail level 3',
    'mfcc_4': 'Spectral detail level 4',
    'mfcc_5': 'Spectral detail level 5',
    'mfcc_6': 'Spectral detail level 6',
    'mfcc_7': 'Spectral detail level 7',
    'mfcc_8': 'Spectral detail level 8',
    'mfcc_9': 'Spectral detail level 9',
    'mfcc_10': 'Spectral detail level 10',
    'mfcc_11': 'Spectral detail level 11',
    'mfcc_12': 'Spectral detail level 12',
    'mfcc_13': 'Spectral detail level 13',
    
    # Spectral
    'spectral_centroid': 'Brightness / Center of frequency mass',
    'spectral_rolloff': 'High frequency content threshold',
    'spectral_contrast': 'Difference between peaks and valleys',
    'spectral_bandwidth': 'Frequency spread around centroid',
    'spectral_flatness': 'Tone-like vs noise-like quality',
    'spectral_flux': 'Rate of spectral change',
    
    # Temporal
    'zero_crossing_rate': 'Signal noisiness / roughness',
    'rms_energy': 'Overall loudness / amplitude',
    'tempo': 'Rhythmic periodicity',
    
    # Advanced
    'chroma_stft_mean': 'Pitch class distribution (mean)',
    'chroma_stft_std': 'Pitch class variability (std)',
    'chroma_cqt_mean': 'Constant-Q chromagram (mean)',
    'chroma_cqt_std': 'Constant-Q chromagram (std)',
    'tonnetz_mean': 'Tonal centroid (mean)',
    'tonnetz_std': 'Tonal centroid variability',
    'poly_features_mean': 'Polynomial coefficient (mean)',
    'poly_features_std': 'Polynomial coefficient (std)',
    'kurtosis': 'Amplitude distribution shape (tailedness)',
    'skewness': 'Amplitude distribution asymmetry',
    
    # New features
    'mel_mean': 'Mel-scale spectral energy (mean)',
    'mel_std': 'Mel-scale spectral energy (std)',
    'contrast_mean': 'Spectral contrast (mean)',
    'contrast_std': 'Spectral contrast (std)',
    'zcr_mean': 'Zero-crossing rate statistics (mean)',
    'zcr_std': 'Zero-crossing rate variability',
    'rolloff_mean': 'Spectral rolloff statistics (mean)',
    'rolloff_std': 'Spectral rolloff variability',
    'bandwidth_mean': 'Spectral bandwidth statistics (mean)',
    'bandwidth_std': 'Spectral bandwidth variability'
}

class AWSLambdaConfig(Config):
    """AWS Lambda-specific configuration overrides"""
    
    # Check if running in AWS Lambda
    IS_LAMBDA = os.getenv('AWS_EXECUTION_ENV', '').startswith('AWS_Lambda')
    
    if IS_LAMBDA:
        # In Lambda, models are in /tmp (ephemeral storage)
        MODELS_DIR = Path('/tmp/models')
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Update model paths for Lambda
        MODEL_PATH = MODELS_DIR / "pump_xgb_model.json"
        RESULTS_PATH = MODELS_DIR / "pump_xgb_results.pkl"
        TRAINING_REPORT_PATH = MODELS_DIR / "training_report.json"
        TRAINING_STATS_PATH = MODELS_DIR / "training_statistics.pkl"
        
        # Logging for Lambda (CloudWatch)
        LOG_LEVEL = "INFO"
        LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
        LOG_FILE = "/tmp/pump_anomaly.log"
    
    # API settings (for reference, not used in Lambda)
    API_RELOAD = False  # Never reload in Lambda

# Export configuration
config = Config()


if __name__ == "__main__":
    # Print configuration when run directly
    config.print_config()
    
    # Validate configuration
    print("\nValidating configuration...")
    errors = config.validate()
    
    if errors:
        print("\n⚠️  Configuration Issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration valid!")