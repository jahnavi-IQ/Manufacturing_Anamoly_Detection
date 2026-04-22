"""
Feature Extraction Module for Pump Anomaly Detection
=====================================================

This module extracts audio features EXACTLY as in pump_ml.py training script.
CRITICAL: Do not modify the feature extraction logic or order.

Features extracted (42 total):
- 13 MFCCs (static only, no deltas)
- 6 Spectral features
- 4 Temporal features
- 12 Advanced features
- 7 New discriminative features
"""

import numpy as np
import librosa
from scipy.stats import kurtosis

from .config import config, FEATURE_NAMES
from .utils import logger


def extract_mfccs(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract static MFCCs only (no deltas)
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate
        n_mfcc: Number of MFCCs to extract
    
    Returns:
        Array of 13 MFCC values
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed


def extract_spectral_features(audio: np.ndarray, sample_rate: int) -> tuple:
    """
    Extract 6 spectral features
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate
    
    Returns:
        Tuple of 6 spectral feature values:
        (spectral_centroid_mean, spectral_centroid_std, spectral_rolloff_mean,
         spectral_contrast_mean, spectral_bandwidth_mean, spectral_bandwidth_std)
    """
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
    
    return (
        np.mean(spectral_centroids),
        np.std(spectral_centroids),
        np.mean(spectral_rolloff),
        np.mean(spectral_contrast),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth)
    )


def extract_temporal_features(audio: np.ndarray, sample_rate: int) -> tuple:
    """
    Extract 4 temporal features
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate
    
    Returns:
        Tuple of 4 temporal feature values:
        (zcr_mean, zcr_std, rms_mean, rms_std)
    """
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    rms_energy = librosa.feature.rms(y=audio)[0]
    
    return (
        np.mean(zero_crossing_rate),
        np.std(zero_crossing_rate),
        np.mean(rms_energy),
        np.std(rms_energy)
    )


def extract_advanced_features(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Extract 12 advanced features
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate
    
    Returns:
        Dictionary with 12 advanced feature values
    """
    features = {}
    
    # Spectral Entropy
    spec = np.abs(librosa.stft(audio))
    spec_norm = spec / (np.sum(spec) + 1e-10)
    spectral_entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-10))
    features['Spectral_Entropy'] = spectral_entropy
    
    # Spectral Flux
    spectral_flux = np.mean(np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0)))
    features['Spectral_Flux'] = spectral_flux
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
    features['Spectral_Bandwidth'] = np.mean(spectral_bandwidth)
    
    # RMS Energy
    rms_energy = librosa.feature.rms(y=audio)[0]
    features['RMS_Energy'] = np.mean(rms_energy)
    
    # Energy Entropy
    energy = audio ** 2
    energy_norm = energy / (np.sum(energy) + 1e-10)
    energy_entropy = -np.sum(energy_norm * np.log2(energy_norm + 1e-10))
    features['Energy_Entropy'] = energy_entropy
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    features['Chroma_Mean'] = np.mean(chroma)
    features['Chroma_Std'] = np.std(chroma)
    features['Chroma_Max'] = np.max(chroma)
    
    # Mel spectrogram features
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['Mel_Spec_Mean'] = np.mean(mel_spec_db)
    features['Mel_Spec_Std'] = np.std(mel_spec_db)
    features['Mel_Spec_Max'] = np.max(mel_spec_db)
    features['Mel_Spec_Min'] = np.min(mel_spec_db)
    
    return features


def extract_new_discriminative_features(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Extract 7 new discriminative features
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate
    
    Returns:
        Dictionary with 7 new feature values
    """
    features = {}
    
    # Fundamental Frequency (pitch tracking)
    try:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        features['Fundamental_Freq'] = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
    except:
        features['Fundamental_Freq'] = 0.0
    
    # Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
    features['Spectral_Flatness'] = np.mean(spectral_flatness)
    
    # Spectral Crest
    spec = np.abs(librosa.stft(audio))
    crest_factor = np.max(spec, axis=0) / (np.mean(spec, axis=0) + 1e-10)
    features['Spectral_Crest'] = np.mean(crest_factor)
    
    # Kurtosis - Time domain
    features['Kurtosis_Time'] = kurtosis(audio)
    
    # Kurtosis - Frequency domain
    spec_magnitudes = np.abs(librosa.stft(audio))
    features['Kurtosis_Freq_Mean'] = np.mean([kurtosis(spec_magnitudes[:, i]) 
                                               for i in range(min(10, spec_magnitudes.shape[1]))])
    
    # Kurtosis - RMS
    rms = librosa.feature.rms(y=audio)[0]
    features['Kurtosis_RMS'] = kurtosis(rms)
    
    # Kurtosis - Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['Kurtosis_ZCR'] = kurtosis(zcr)
    
    return features


def extract_all_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract complete 42-feature set matching training
    
    This function MUST extract features in the exact same order as pump_ml.py
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate
    
    Returns:
        Numpy array of 42 features
    
    Raises:
        ValueError: If feature extraction fails
    """
    try:
        features = []
        
        # 1. MFCCs (13 features)
        if config.EXTRACT_MFCCS:
            mfccs = extract_mfccs(audio, sample_rate, n_mfcc=config.N_MFCC)
            features.append(mfccs)
            logger.debug(f"Extracted {len(mfccs)} MFCCs")
        
        # 2. Spectral features (6 features)
        if config.EXTRACT_SPECTRAL:
            spectral_features = extract_spectral_features(audio, sample_rate)
            features.extend(spectral_features)
            logger.debug(f"Extracted {len(spectral_features)} spectral features")
        
        # 3. Temporal features (4 features)
        if config.EXTRACT_TEMPORAL:
            temporal_features = extract_temporal_features(audio, sample_rate)
            features.extend(temporal_features)
            logger.debug(f"Extracted {len(temporal_features)} temporal features")
        
        # 4. Advanced features (12 features)
        if config.EXTRACT_ADVANCED:
            advanced_features = extract_advanced_features(audio, sample_rate)
            features.extend(advanced_features.values())
            logger.debug(f"Extracted {len(advanced_features)} advanced features")
        
        # 5. New discriminative features (7 features)
        if config.EXTRACT_NEW_FEATURES:
            new_features = extract_new_discriminative_features(audio, sample_rate)
            features.extend(new_features.values())
            logger.debug(f"Extracted {len(new_features)} new features")
        
        # Concatenate all features into single array
        feature_array = np.concatenate([np.atleast_1d(f) for f in features])
        
        # Validate
        if len(feature_array) != config.EXPECTED_N_FEATURES:
            raise ValueError(
                f"Feature count mismatch! Expected {config.EXPECTED_N_FEATURES}, "
                f"got {len(feature_array)}"
            )
        
        # Check for NaN or Inf
        if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
            raise ValueError("Feature array contains NaN or Inf values")
        
        logger.info(f"Successfully extracted {len(feature_array)} features")
        
        return feature_array
    
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise ValueError(f"Feature extraction error: {e}")


def get_feature_info() -> dict:
    """
    Get information about features
    
    Returns:
        Dictionary with feature metadata
    """
    return {
        'total_features': config.EXPECTED_N_FEATURES,
        'feature_names': FEATURE_NAMES,
        'feature_groups': {
            'mfccs': list(range(0, 13)),
            'spectral': list(range(13, 19)),
            'temporal': list(range(19, 23)),
            'advanced': list(range(23, 35)),
            'new': list(range(35, 42))
        },
        'n_mfcc': config.N_MFCC
    }


def create_feature_dataframe(features: np.ndarray) -> dict:
    """
    Create a dictionary mapping feature names to values
    
    Args:
        features: Feature array (42 values)
    
    Returns:
        Dictionary of feature_name: value pairs
    """
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Feature count mismatch: {len(features)} vs {len(FEATURE_NAMES)}")
    
    return {name: float(value) for name, value in zip(FEATURE_NAMES, features)}


# Example usage and testing
if __name__ == "__main__":
    print("Feature Extraction Module Test")
    print("=" * 60)
    
    # Generate test audio signal
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    
    # Synthesize test signal (sine wave + noise)
    frequency = 440.0  # A4 note
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
    
    print(f"Test audio: {len(test_audio)} samples, {sr}Hz, {duration}s")
    
    # Extract features
    try:
        features = extract_all_features(test_audio, sr)
        print(f"\n✅ Successfully extracted {len(features)} features")
        
        # Display feature info
        feature_dict = create_feature_dataframe(features)
        
        print("\nFeature Groups:")
        info = get_feature_info()
        for group_name, indices in info['feature_groups'].items():
            print(f"  {group_name}: {len(indices)} features")
        
        print("\nSample Features:")
        for i, (name, value) in enumerate(list(feature_dict.items())[:5]):
            print(f"  {i+1}. {name}: {value:.6f}")
        print("  ...")
        
        # Validate
        if len(features) == config.EXPECTED_N_FEATURES:
            print(f"\n✅ Feature count correct: {len(features)}")
        else:
            print(f"\n❌ Feature count mismatch: {len(features)} vs {config.EXPECTED_N_FEATURES}")
        
        if not (np.any(np.isnan(features)) or np.any(np.isinf(features))):
            print("✅ No NaN or Inf values")
        else:
            print("❌ Contains NaN or Inf values")
            
    except Exception as e:
        print(f"\n❌ Feature extraction failed: {e}")
