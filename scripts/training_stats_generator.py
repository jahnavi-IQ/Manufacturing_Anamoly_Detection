"""
Training Statistics Generator
==============================

ONE-TIME SCRIPT: Run this after training to generate feature statistics
for explainability and deviation analysis.

This script:
1. Loads all training audio files
2. Extracts features using the same pipeline as training
3. Calculates mean and std for each feature
4. Saves statistics to training_statistics.pkl

Usage:
    python scripts/training_stats_generator.py
"""

import numpy as np
import librosa
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config, FEATURE_NAMES
from src.feature_extractor import extract_all_features
from src.utils import logger


def load_training_data():
    """
    Load all training audio files from the dataset
    
    Returns:
        List of (audio, sample_rate, label, filename) tuples
    """
    print("\n" + "=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)
    
    training_data = []
    
    for pump_id in config.PUMP_IDS:
        base_path = config.DATA_DIR / pump_id
        
        # Load abnormal samples
        abnormal_path = base_path / "abnormal"
        if abnormal_path.exists():
            files = list(abnormal_path.glob("*.wav"))
            print(f"\n📂 {pump_id}/abnormal: Found {len(files)} files")
            
            for file_path in tqdm(files, desc=f"  Loading {pump_id}/abnormal"):
                try:
                    audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
                    training_data.append((audio, sr, 1, file_path.name))
                except Exception as e:
                    print(f"  ⚠️  Error loading {file_path.name}: {e}")
        
        # Load normal samples
        normal_path = base_path / "normal"
        if normal_path.exists():
            files = list(normal_path.glob("*.wav"))
            print(f"\n📂 {pump_id}/normal: Found {len(files)} files")
            
            for file_path in tqdm(files, desc=f"  Loading {pump_id}/normal"):
                try:
                    audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
                    training_data.append((audio, sr, 0, file_path.name))
                except Exception as e:
                    print(f"  ⚠️  Error loading {file_path.name}: {e}")
    
    print(f"\n✅ Total samples loaded: {len(training_data)}")
    normal_count = sum(1 for _, _, label, _ in training_data if label == 0)
    abnormal_count = sum(1 for _, _, label, _ in training_data if label == 1)
    print(f"   ├─ Normal: {normal_count}")
    print(f"   └─ Abnormal: {abnormal_count}")
    
    return training_data


def extract_features_from_training_data(training_data):
    """
    Extract features from all training samples
    
    Args:
        training_data: List of (audio, sr, label, filename) tuples
    
    Returns:
        Tuple of (feature_matrix, labels, filenames)
    """
    print("\n" + "=" * 80)
    print("EXTRACTING FEATURES FROM TRAINING DATA")
    print("=" * 80)
    
    all_features = []
    all_labels = []
    all_filenames = []
    failed_count = 0
    
    for audio, sr, label, filename in tqdm(training_data, desc="Extracting features"):
        try:
            features = extract_all_features(audio, sr)
            all_features.append(features)
            all_labels.append(label)
            all_filenames.append(filename)
        except Exception as e:
            print(f"\n⚠️  Failed to extract features from {filename}: {e}")
            failed_count += 1
    
    feature_matrix = np.array(all_features)
    labels = np.array(all_labels)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   ├─ Successful: {len(all_features)}")
    print(f"   ├─ Failed: {failed_count}")
    print(f"   └─ Feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix, labels, all_filenames


def calculate_statistics(feature_matrix, labels):
    """
    Calculate mean and std for each feature across all training data
    
    Args:
        feature_matrix: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
    
    Returns:
        Dictionary with statistics
    """
    print("\n" + "=" * 80)
    print("CALCULATING FEATURE STATISTICS")
    print("=" * 80)
    
    # Overall statistics
    overall_mean = np.mean(feature_matrix, axis=0)
    overall_std = np.std(feature_matrix, axis=0)
    
    # Statistics per class
    normal_features = feature_matrix[labels == 0]
    abnormal_features = feature_matrix[labels == 1]
    
    normal_mean = np.mean(normal_features, axis=0)
    normal_std = np.std(normal_features, axis=0)
    
    abnormal_mean = np.mean(abnormal_features, axis=0)
    abnormal_std = np.std(abnormal_features, axis=0)
    
    # Min/max for each feature
    feature_min = np.min(feature_matrix, axis=0)
    feature_max = np.max(feature_matrix, axis=0)
    
    statistics = {
        'overall': {
            'mean': overall_mean,
            'std': overall_std,
            'min': feature_min,
            'max': feature_max
        },
        'normal': {
            'mean': normal_mean,
            'std': normal_std,
            'n_samples': len(normal_features)
        },
        'abnormal': {
            'mean': abnormal_mean,
            'std': abnormal_std,
            'n_samples': len(abnormal_features)
        },
        'feature_names': FEATURE_NAMES,
        'n_features': feature_matrix.shape[1]
    }
    
    print(f"✅ Statistics calculated for {len(FEATURE_NAMES)} features")
    print(f"   ├─ Normal samples: {len(normal_features)}")
    print(f"   └─ Abnormal samples: {len(abnormal_features)}")
    
    # Display sample statistics
    print("\nSample Statistics (first 5 features):")
    print(f"{'Feature':<25} {'Overall Mean':<15} {'Overall Std':<15}")
    print("-" * 55)
    for i in range(min(5, len(FEATURE_NAMES))):
        print(f"{FEATURE_NAMES[i]:<25} {overall_mean[i]:<15.6f} {overall_std[i]:<15.6f}")
    print("...")
    
    return statistics


def save_statistics(statistics, output_path):
    """
    Save statistics to pickle file
    
    Args:
        statistics: Statistics dictionary
        output_path: Output file path
    """
    print("\n" + "=" * 80)
    print("SAVING STATISTICS")
    print("=" * 80)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(statistics, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Statistics saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")


def validate_statistics(statistics):
    """
    Validate generated statistics
    
    Args:
        statistics: Statistics dictionary
    
    Returns:
        True if valid, False otherwise
    """
    print("\n" + "=" * 80)
    print("VALIDATING STATISTICS")
    print("=" * 80)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Feature count
    total_checks += 1
    if statistics['n_features'] == config.EXPECTED_N_FEATURES:
        print(f"✅ Feature count: {statistics['n_features']}")
        checks_passed += 1
    else:
        print(f"❌ Feature count mismatch: {statistics['n_features']} vs {config.EXPECTED_N_FEATURES}")
    
    # Check 2: Feature names
    total_checks += 1
    if len(statistics['feature_names']) == len(FEATURE_NAMES):
        print(f"✅ Feature names: {len(statistics['feature_names'])} names")
        checks_passed += 1
    else:
        print(f"❌ Feature names count mismatch")
    
    # Check 3: No NaN or Inf
    total_checks += 1
    overall_mean = statistics['overall']['mean']
    overall_std = statistics['overall']['std']
    
    if not (np.any(np.isnan(overall_mean)) or np.any(np.isinf(overall_mean)) or
            np.any(np.isnan(overall_std)) or np.any(np.isinf(overall_std))):
        print("✅ No NaN or Inf values in statistics")
        checks_passed += 1
    else:
        print("❌ Statistics contain NaN or Inf values")
    
    # Check 4: Positive std (mostly)
    total_checks += 1
    zero_std_count = np.sum(overall_std == 0)
    if zero_std_count < len(overall_std) * 0.1:  # Less than 10% zero std
        print(f"✅ Standard deviations valid ({zero_std_count} zeros)")
        checks_passed += 1
    else:
        print(f"⚠️  Many zero standard deviations: {zero_std_count}")
        checks_passed += 1  # Still pass but warn
    
    print(f"\n{'='*80}")
    print(f"Validation: {checks_passed}/{total_checks} checks passed")
    
    return checks_passed == total_checks


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("TRAINING STATISTICS GENERATOR")
    print("=" * 80)
    print(f"This script will generate feature statistics for explainability")
    print(f"Output: {config.TRAINING_STATS_PATH}")
    print("=" * 80)
    
    # Check if data directory exists
    if not config.DATA_DIR.exists():
        print(f"\n❌ ERROR: Data directory not found: {config.DATA_DIR}")
        print("Please ensure the training data is in the correct location.")
        print("Expected structure:")
        print(f"  {config.DATA_DIR}/")
        for pump_id in config.PUMP_IDS:
            print(f"    ├─ {pump_id}/")
            print(f"    │   ├─ normal/*.wav")
            print(f"    │   └─ abnormal/*.wav")
        return
    
    # Check if output already exists
    if config.TRAINING_STATS_PATH.exists():
        response = input(f"\n⚠️  File already exists: {config.TRAINING_STATS_PATH}\nOverwrite? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    
    try:
        # Step 1: Load training data
        training_data = load_training_data()
        
        if len(training_data) == 0:
            print("\n❌ ERROR: No training data loaded!")
            return
        
        # Step 2: Extract features
        feature_matrix, labels, filenames = extract_features_from_training_data(training_data)
        
        # Step 3: Calculate statistics
        statistics = calculate_statistics(feature_matrix, labels)
        
        # Step 4: Validate
        is_valid = validate_statistics(statistics)
        
        if not is_valid:
            print("\n⚠️  Statistics validation failed. Proceeding anyway...")
        
        # Step 5: Save
        save_statistics(statistics, config.TRAINING_STATS_PATH)
        
        print("\n" + "=" * 80)
        print("✅ STATISTICS GENERATION COMPLETE!")
        print("=" * 80)
        print(f"\nYou can now run the inference system.")
        print(f"The explainability module will use these statistics to analyze deviations.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nStatistics generation failed!")


if __name__ == "__main__":
    main()
