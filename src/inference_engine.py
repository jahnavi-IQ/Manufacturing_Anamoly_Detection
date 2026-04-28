"""
Inference Engine for Pump Anomaly Detection
============================================

Core prediction engine that:
1. Loads model artifacts
2. Extracts features from audio
3. Makes predictions
4. Generates explanations

"""

import numpy as np
import xgboost as xgb
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from .config import config, FEATURE_NAMES
from .feature_extractor import extract_all_features
from .explainability import ExplainabilityEngine
from .utils import (
    load_audio, 
    get_audio_properties, 
    get_confidence_level,
    validate_feature_array,
    logger,
    ModelLoadingError
)


class PumpAnomalyDetector:
    """
    Complete inference pipeline for pump anomaly detection
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        results_path: Optional[str] = None,
        training_stats_path: Optional[str] = None
    ):
        """
        Initialize detector
        
        Args:
            model_path: Path to XGBoost model JSON
            results_path: Path to results pickle
            training_stats_path: Path to training statistics pickle
        """
        # Use defaults from config if not provided
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.results_path = Path(results_path or config.RESULTS_PATH)
        self.training_stats_path = Path(training_stats_path or config.TRAINING_STATS_PATH)
        
        # Initialize components
        self.model = None
        self.metadata = None
        self.explainer = None
        
        # Model parameters
        self.optimal_threshold = config.DEFAULT_THRESHOLD
        self.feature_names = FEATURE_NAMES
        self.sample_rate = None
        
        # Load everything
        self._load_model_artifacts()
    
    def _load_model_artifacts(self):
        logger.info("Loading model artifacts...")

        try:
            self._load_xgboost_model()

            if self.model is None:
                raise ModelLoadingError("XGBoost model failed to load. self.model is None")

            self._load_metadata()

            if self.metadata is None:
                raise ModelLoadingError("Metadata failed to load. self.metadata is None")

            self._init_explainer()

            if self.explainer is None:
                raise ModelLoadingError("Explainability engine initialization failed")

            logger.info("✅ All artifacts loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise ModelLoadingError(f"Model loading failed: {e}")

    
    def _load_xgboost_model(self):
        """Load XGBoost model from JSON"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path))
            logger.info(f"Loaded XGBoost model!")
        
        except Exception as e:
            raise ModelLoadingError(f"Failed to load XGBoost model: {e}")
    
    def _load_metadata(self):
        """Load metadata from results pickle"""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        
        try:
            with open(self.results_path, 'rb') as f:
                self.metadata = pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" not in str(e):
                raise ModelLoadingError(f"Failed to load metadata: {e}")
            logger.warning("Pickle compatibility shim applied for numpy._core")
            import numpy.core as numpy_core
            sys.modules.setdefault("numpy._core", numpy_core)
            with open(self.results_path, 'rb') as f:
                self.metadata = pickle.load(f)
        except Exception as e:
            raise ModelLoadingError(f"Failed to load metadata: {e}")
            
        try:
            # Extract important parameters
            self.optimal_threshold = self.metadata['threshold_info']['optimal_threshold']
            self.sample_rate = self.metadata['data_info']['sample_rate']
            self.feature_names = self.metadata['data_info']['feature_names']
            
            logger.info(f"Loaded metadata!")
        
        except Exception as e:
            raise ModelLoadingError(f"Failed to load metadata: {e}")
    
    def _init_explainer(self):
        """Initialize explainability engine"""
        try:
            self.explainer = ExplainabilityEngine(
                training_stats_path=str(self.training_stats_path)
            )
            
            if self.explainer.stats_loaded:
                logger.info("✅ Explainability engine initialized with training stats")
            else:
                logger.warning("⚠️  Explainability engine initialized WITHOUT training stats")
                logger.warning("   Run training_stats_generator.py to enable full explainability")
        
        except Exception as e:
            logger.warning(f"Explainability initialization warning: {e}")
            self.explainer = ExplainabilityEngine()  # Initialize without stats
    
    def predict_from_file(self, audio_file_path: str) -> Dict:
        """
        Complete prediction pipeline from audio file
        
        Args:
            audio_file_path: Path to audio file (.wav)
        
        Returns:
            Dictionary with prediction results and explainability
        """
        logger.info(f"Processing audio file: {audio_file_path}")
        
        try:
            # Step 1: Load audio
            audio, sr = load_audio(audio_file_path, sr=self.sample_rate)
            
            # Get audio properties
            audio_props = get_audio_properties(audio, sr)
            logger.info(f"Audio loaded: {audio_props['duration']:.2f}s @ {sr}Hz")
            
            # Step 2: Extract features
            features = extract_all_features(audio, sr)
            
            # Validate features
            if not validate_feature_array(features, config.EXPECTED_N_FEATURES):
                raise ValueError("Feature validation failed")
            
            logger.info(f"Extracted {len(features)} features")
            
            # Step 3: Make prediction
            prediction_result = self._predict_from_features(features)
            
            # Step 4: Add audio properties
            prediction_result['audio_properties'] = audio_props
            
            # Step 5: Add file info
            prediction_result['file_info'] = {
                'filename': Path(audio_file_path).name,
                'file_path': str(audio_file_path)
            }
            
            logger.info(f"Prediction: {prediction_result['prediction']} "
                       f"(confidence: {prediction_result['confidence']:.2f})")
            
            return prediction_result
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _predict_from_features(self, features: np.ndarray) -> Dict:
        """
        Make prediction from feature array
        
        Args:
            features: Feature array 
        
        Returns:
            Prediction dictionary
        """
        if self.model is None:
            raise ModelLoadingError("Model is not loaded. self.model is None")

        if self.explainer is None:
            raise RuntimeError("Explainability engine not initialized")

        if self.metadata is None:
            raise RuntimeError("Model metadata missing (self.metadata is None)")
        
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(features.reshape(1, -1), feature_names=self.feature_names)
        
        # Get prediction probability
        prob = self.model.predict(dmatrix)[0]  # Probability of abnormal class
        
        # Apply optimal threshold
        predicted_class = 1 if prob >= self.optimal_threshold else 0
        prediction_label = config.CLASS_NAMES[predicted_class]
        
        # Calculate probabilities for both classes
        prob_abnormal = float(prob)
        prob_normal = float(1 - prob)
        
        # Confidence is the maximum probability
        confidence = max(prob_normal, prob_abnormal)
        is_confident = confidence >= config.HIGH_CONFIDENCE_THRESHOLD
        
        # Generate explainability
        explainability = self.explainer.create_full_explanation(
            prediction=prediction_label,
            confidence=confidence,
            probabilities={'normal': prob_normal, 'abnormal': prob_abnormal},
            user_features=features
        )
        
        return {
            # Core prediction
            'prediction': prediction_label,
            'predicted_class': int(predicted_class),
            
            # Probabilities
            'probability_normal': prob_normal,
            'probability_abnormal': prob_abnormal,
            'raw_score': prob_abnormal,
            
            # Confidence
            'confidence': confidence,
            'confidence_level': get_confidence_level(confidence),
            'is_confident': is_confident,
            
            # Threshold info
            'threshold_used': float(self.optimal_threshold),
            'default_threshold': 0.5,
            
            # Explainability
            'explainability': explainability,
            
            # Features
            'features': {
                'values': features.tolist(),
                'names': self.feature_names,
                'count': len(features)
            },
            
            # Model info
            'model_info': {
                'model_type': 'XGBoost',
                'version': self.metadata['model_metadata']['version'],
                'training_timestamp': self.metadata['model_metadata']['training_timestamp']
            }
        }
    
    def get_model_info(self) -> Dict:
        """
        Get model metadata and performance metrics
        
        Returns:
            Dictionary with model information
        """
        if not self.metadata:
            return {'error': 'Model metadata not loaded'}
        
        return {
            'model_metadata': self.metadata['model_metadata'],
            'performance_metrics': self.metadata['performance_metrics'],
            'threshold_info': self.metadata['threshold_info'],
            'data_info': {
                'n_features': self.metadata['data_info']['n_features'],
                'pump_ids': self.metadata['data_info']['pump_ids'],
                'sample_rate': self.metadata['data_info']['sample_rate']
            }
        }
    
    def validate_setup(self) -> Tuple[bool, List[str]]:
        """
        Validate that all components are properly loaded
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check model
        if self.model is None:
            issues.append("Model not loaded")
        
        # Check metadata
        if self.metadata is None:
            issues.append("Metadata not loaded")
        
        # Check feature names
        if len(self.feature_names) != config.EXPECTED_N_FEATURES:
            issues.append(f"Feature count mismatch: {len(self.feature_names)} vs {config.EXPECTED_N_FEATURES}")
        
        # Check explainer
        if self.explainer is None:
            issues.append("Explainability engine not initialized")
        
        # Check training stats (warning only)
        if not self.explainer or not self.explainer.stats_loaded:
            issues.append("WARNING: Training statistics not loaded (explainability limited)")
        
        is_valid = len([i for i in issues if not i.startswith('WARNING')]) == 0
        
        return is_valid, issues


# Singleton instance (loaded once for API)
_detector_instance = None


def get_detector() -> PumpAnomalyDetector:
    """
    Get singleton detector instance
    
    Returns:
        PumpAnomalyDetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        logger.info("Initializing PumpAnomalyDetector...")
        _detector_instance = PumpAnomalyDetector()
        
        # Validate setup
        is_valid, issues = _detector_instance.validate_setup()
        
        if not is_valid:
            logger.error("Detector validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("✅ Detector validated successfully")
            if issues:  # Warnings
                for issue in issues:
                    logger.warning(f"  - {issue}")
    
    return _detector_instance


# Example usage
if __name__ == "__main__":
    print("Inference Engine Test")
    print("=" * 60)
    
    try:
        # Initialize detector
        detector = PumpAnomalyDetector()
        
        # Validate
        is_valid, issues = detector.validate_setup()
        
        print("\nValidation Results:")
        if is_valid:
            print("✅ Detector ready for inference")
        else:
            print("❌ Validation failed:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Display model info
        model_info = detector.get_model_info()
        print("\nModel Information:")
        print(f"  Version: {model_info['model_metadata']['version']}")
        print(f"  Training Date: {model_info['model_metadata']['training_timestamp']}")
        print(f"  Test Accuracy: {model_info['performance_metrics']['test']['accuracy']:.4f}")
        print(f"  Test Recall: {model_info['performance_metrics']['test']['recall']:.4f}")
        print(f"  Optimal Threshold: {model_info['threshold_info']['optimal_threshold']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
