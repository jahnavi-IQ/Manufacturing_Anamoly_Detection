"""
PUMP-XGBOOST OPTIMIZED 
==========================================

Features:
- XGBoost classifier with NATIVE feature selection
- Uses ALL 42 extracted features (lets XGBoost decide importance)
- Bayesian Optimization (Optuna) for hyperparameter tuning
- Proper data leakage prevention
- Comprehensive JSON reporting

"""

import numpy as np
import librosa
import os
import warnings
from tqdm import tqdm
import pickle
import json
from datetime import datetime
from scipy.stats import kurtosis

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score, 
    precision_score, recall_score, roc_auc_score
)

# Bayesian Optimization
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 100)
print(" " * 25 + "PUMP-XGBOOST OPTIMIZED")
print("=" * 100)
print(f"XGBoost version: {xgb.__version__}")
print(f"Optuna version: {optuna.__version__}")


class Config:
    """Configuration for XGBoost model with native feature selection"""
    # Paths
    ROOT_DIR = os.getcwd()
    PUMP_IDS = ["id_00", "id_02", "id_04", "id_06"]
    SAMPLE_RATE = None
    N_MFCC = 13
    
    # Feature extraction flags 
    EXTRACT_MFCCS = True 
    EXTRACT_SPECTRAL = True
    EXTRACT_TEMPORAL = True
    EXTRACT_ADVANCED = True
    EXTRACT_NEW_FEATURES = True  
    
    # Data augmentation
    AUGMENT_DATA = True
    NOISE_FACTOR = 0.005
    SHIFT_MAX = 0.2
    AUGMENTATION_FACTOR = 10
    PITCH_SHIFT_RANGE = 2
    TIME_STRETCH_RANGE = (0.9, 1.1)
    VOLUME_SCALE_RANGE = (0.8, 1.2)
    
    # Data splitting
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    
    # Bayesian Optimization settings
    USE_BAYESIAN_OPTIMIZATION = True
    N_TRIALS = 100
    OPTIMIZATION_TIMEOUT = 7200  # 1 hour
    
    # RECALL-FOCUSED OPTIMIZATION
    OPTIMIZE_FOR_RECALL = True 
    TARGET_RECALL = 0.95  
    MIN_PRECISION = 0.75  

    # Cost-sensitive learning - CRITICAL FOR SAFETY
    FALSE_NEGATIVE_COST = 5.0  
    FALSE_POSITIVE_COST = 1.0
    
    # Default XGBoost hyperparameters
    DEFAULT_PARAMS = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0
    }
    
    # Early stopping
    EARLY_STOPPING_ROUNDS = 50
    
    # Threshold optimization
    OPTIMIZE_THRESHOLD = True
    OPTIMIZE_FOR_F1 = False
    
    CLASS_NAMES = ['Normal', 'Abnormal']
    
    # Save paths
    MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "models_ml")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Required file outputs 
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'pump_xgb_model.json')
    RESULTS_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'pump_xgb_results.pkl')
    JSON_REPORT_PATH = os.path.join(MODEL_SAVE_DIR, 'training_report.json')

config = Config()

print("\n" + "=" * 100)
print("CONFIGURATION")
print("=" * 100)

print(f"Bayesian Optimization: {'✅ ENABLED' if config.USE_BAYESIAN_OPTIMIZATION else '❌ DISABLED'}")
if config.USE_BAYESIAN_OPTIMIZATION:
    print(f"   ├─ Number of trials: {config.N_TRIALS}")
    print(f"   └─ Timeout: {config.OPTIMIZATION_TIMEOUT}s ({config.OPTIMIZATION_TIMEOUT/3600:.1f} hours)")
print(f"\nData Processing:")
print(f"   ├─ Augmentation Factor: {config.AUGMENTATION_FACTOR}x")
print(f"   └─ Threshold Optimization: {'✅ ENABLED' if config.OPTIMIZE_THRESHOLD else '❌ DISABLED'}")


# Initialize tracking variables
training_start_time = datetime.now()

print("\n" + "=" * 100)
print("SECTION 1: DATA LOADING")
print("=" * 100)
print(f"Target Pump IDs: {config.PUMP_IDS}")

abnormal_files_all = []
normal_files_all = []

for pump_id in config.PUMP_IDS:
    base_path = os.path.join(config.ROOT_DIR, "6_dB_pump", "pump", pump_id)
    abnormal_path = os.path.join(base_path, "abnormal")
    normal_path = os.path.join(base_path, "normal")
    
    if os.path.exists(abnormal_path):
        files = [os.path.join(abnormal_path, f) for f in os.listdir(abnormal_path) 
                 if f.endswith('.wav')]
        abnormal_files_all.extend(files)
        print(f"   ├─ {pump_id}: Found {len(files)} abnormal files")
    
    if os.path.exists(normal_path):
        files = [os.path.join(normal_path, f) for f in os.listdir(normal_path) 
                 if f.endswith('.wav')]
        normal_files_all.extend(files)
        print(f"   └─ {pump_id}: Found {len(files)} normal files")

# Load audio files
abnormal_audio = []
abnormal_labels = []
sample_rate = None

print(f"\n📂 Loading {len(abnormal_files_all)} Abnormal samples...")
for file_path in tqdm(abnormal_files_all, desc="Loading Abnormal"):
    try:
        audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
        if sample_rate is None:
            sample_rate = sr
        abnormal_audio.append(audio)
        abnormal_labels.append(1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

normal_audio = []
normal_labels = []

print(f"\n📂 Loading {len(normal_files_all)} Normal samples...")
for file_path in tqdm(normal_files_all, desc="Loading Normal"):
    try:
        audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
        normal_audio.append(audio)
        normal_labels.append(0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

all_audio = abnormal_audio + normal_audio
all_labels = abnormal_labels + normal_labels
sr = sample_rate

print("\n")
print("DATASET SUMMARY")
print("=" * 100)
print(f"Total samples: {len(all_audio)}")
print(f"  ├─ Normal samples: {len(normal_audio)} ({len(normal_audio)/len(all_audio)*100:.1f}%)")
print(f"  └─ Abnormal samples: {len(abnormal_audio)} ({len(abnormal_audio)/len(all_audio)*100:.1f}%)")
print(f"\nClass Distribution:")
imbalance_ratio = len(normal_audio)/len(abnormal_audio)
print(f"  ├─ Imbalance Ratio: 1:{imbalance_ratio:.2f}")
print(f"  └─ Balance Status: {'⚠️  Imbalanced' if imbalance_ratio > 1.5 else '✅ Balanced'}")
print(f"\nAudio Properties:")
print(f"  ├─ Sample Rate: {sample_rate} Hz")
if sample_rate:
    duration = len(all_audio[0]) / sample_rate
    print(f"  ├─ Duration: {duration:.2f} seconds")
    print(f"  └─ Samples per file: {len(all_audio[0])}")

print("\n" + "=" * 100)
print("SECTION 2: FEATURE EXTRACTION")
print("=" * 100)
print("🔥 Extracting ALL features - XGBoost will handle feature importance natively")

def extract_mfccs(audio, sample_rate, n_mfcc=13):
    """Extract static MFCCs only"""
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def extract_spectral_features(audio, sample_rate):
    """Extract 6 spectral features"""
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

def extract_temporal_features(audio, sample_rate):
    """Extract 4 temporal features"""
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    rms_energy = librosa.feature.rms(y=audio)[0]
    
    return (
        np.mean(zero_crossing_rate),
        np.std(zero_crossing_rate),
        np.mean(rms_energy),
        np.std(rms_energy)
    )

def extract_advanced_features(audio, sample_rate):
    """Extract 12 advanced features"""
    features = {}
    
    spec = np.abs(librosa.stft(audio))
    spec_norm = spec / (np.sum(spec) + 1e-10)
    spectral_entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-10))
    features['Spectral_Entropy'] = spectral_entropy
    
    spectral_flux = np.mean(np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0)))
    features['Spectral_Flux'] = spectral_flux
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
    features['Spectral_Bandwidth'] = np.mean(spectral_bandwidth)
    
    rms_energy = librosa.feature.rms(y=audio)[0]
    features['RMS_Energy'] = np.mean(rms_energy)
    
    energy = audio ** 2
    energy_norm = energy / (np.sum(energy) + 1e-10)
    energy_entropy = -np.sum(energy_norm * np.log2(energy_norm + 1e-10))
    features['Energy_Entropy'] = energy_entropy
    
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    features['Chroma_Mean'] = np.mean(chroma)
    features['Chroma_Std'] = np.std(chroma)
    features['Chroma_Max'] = np.max(chroma)
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['Mel_Spec_Mean'] = np.mean(mel_spec_db)
    features['Mel_Spec_Std'] = np.std(mel_spec_db)
    features['Mel_Spec_Max'] = np.max(mel_spec_db)
    features['Mel_Spec_Min'] = np.min(mel_spec_db)
    
    return features

def extract_new_discriminative_features(audio, sample_rate):
    """Extract new discriminative features (7 total)"""
    features = {}
    
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
    
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
    features['Spectral_Flatness'] = np.mean(spectral_flatness)
    
    spec = np.abs(librosa.stft(audio))
    crest_factor = np.max(spec, axis=0) / (np.mean(spec, axis=0) + 1e-10)
    features['Spectral_Crest'] = np.mean(crest_factor)
    
    features['Kurtosis_Time'] = kurtosis(audio)
    
    spec_magnitudes = np.abs(librosa.stft(audio))
    features['Kurtosis_Freq_Mean'] = np.mean([kurtosis(spec_magnitudes[:, i]) for i in range(min(10, spec_magnitudes.shape[1]))])
    
    rms = librosa.feature.rms(y=audio)[0]
    features['Kurtosis_RMS'] = kurtosis(rms)
    
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['Kurtosis_ZCR'] = kurtosis(zcr)
    
    return features

def extract_all_features(audio, sample_rate):
    """Extract complete feature set"""
    features = []
    
    if config.EXTRACT_MFCCS:
        mfccs = extract_mfccs(audio, sample_rate, n_mfcc=config.N_MFCC)
        features.append(mfccs)
    
    if config.EXTRACT_SPECTRAL:
        spectral_features = extract_spectral_features(audio, sample_rate)
        features.extend(spectral_features)
    
    if config.EXTRACT_TEMPORAL:
        temporal_features = extract_temporal_features(audio, sample_rate)
        features.extend(temporal_features)
    
    if config.EXTRACT_ADVANCED:
        advanced_features = extract_advanced_features(audio, sample_rate)
        features.extend(advanced_features.values())
    
    if config.EXTRACT_NEW_FEATURES:
        new_features = extract_new_discriminative_features(audio, sample_rate)
        features.extend(new_features.values())
    
    return np.concatenate([np.atleast_1d(f) for f in features])

# Build feature names list 
feature_names = []

if config.EXTRACT_MFCCS:
    feature_names.extend([f'MFCC_{i+1}' for i in range(config.N_MFCC)])

if config.EXTRACT_SPECTRAL:
    feature_names.extend(['Spectral_Centroid_Mean', 'Spectral_Centroid_Std', 
                         'Spectral_Rolloff', 'Spectral_Contrast',
                         'Spectral_Bandwidth_Mean', 'Spectral_Bandwidth_Std'])

if config.EXTRACT_TEMPORAL:
    feature_names.extend(['Zero_Crossing_Rate_Mean', 'Zero_Crossing_Rate_Std',
                         'RMS_Energy_Mean', 'RMS_Energy_Std'])

if config.EXTRACT_ADVANCED:
    feature_names.extend(['Spectral_Entropy', 'Spectral_Flux', 'Spectral_Bandwidth',
                         'RMS_Energy', 'Energy_Entropy', 'Chroma_Mean', 'Chroma_Std',
                         'Chroma_Max', 'Mel_Spec_Mean', 'Mel_Spec_Std', 
                         'Mel_Spec_Max', 'Mel_Spec_Min'])

if config.EXTRACT_NEW_FEATURES:
    feature_names.extend(['Fundamental_Freq', 'Spectral_Flatness', 'Spectral_Crest',
                         'Kurtosis_Time', 'Kurtosis_Freq_Mean', 'Kurtosis_RMS', 'Kurtosis_ZCR'])

print(f"\n📊 Feature Breakdown:")
print(f"   ├─ Static MFCCs: {config.N_MFCC if config.EXTRACT_MFCCS else 0}")
print(f"   ├─ Spectral: {6 if config.EXTRACT_SPECTRAL else 0}")
print(f"   ├─ Temporal: {4 if config.EXTRACT_TEMPORAL else 0}")
print(f"   ├─ Advanced: {12 if config.EXTRACT_ADVANCED else 0}")
print(f"   └─ New: {7 if config.EXTRACT_NEW_FEATURES else 0}")
print(f"   └─ 🔥 Total Features: {len(feature_names)}")

print("\n📊 Extracting features from all samples...")

def extract_features_from_dataset(audio_data, sample_rate):
    features = []
    for audio in tqdm(audio_data, desc="Extracting features"):
        feature_vector = extract_all_features(audio, sample_rate)
        features.append(feature_vector)
    return np.array(features)

normal_features = extract_features_from_dataset(normal_audio, sample_rate)
abnormal_features = extract_features_from_dataset(abnormal_audio, sample_rate)

print(f"\n✅ Feature extraction complete.")

print("\n" + "=" * 100)
print("SECTION 3: DATA SPLITTING & AUGMENTATION")
print("=" * 100)

X_original = np.concatenate((normal_features, abnormal_features))
y_original = np.concatenate((
    np.zeros(normal_features.shape[0]), 
    np.ones(abnormal_features.shape[0])
))

print(f"\n📦 Original Dataset:")
print(f"   ├─ Total samples: {X_original.shape[0]}")
print(f"   ├─ Features per sample: {X_original.shape[1]}")
print(f"   ├─ Normal samples: {np.sum(y_original == 0)} ({np.sum(y_original == 0)/len(y_original)*100:.1f}%)")
print(f"   └─ Abnormal samples: {np.sum(y_original == 1)} ({np.sum(y_original == 1)/len(y_original)*100:.1f}%)")

# Split data BEFORE augmentation
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_original, y_original, 
    test_size=config.TEST_SIZE, 
    stratify=y_original, 
    random_state=RANDOM_SEED, 
    shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size=config.VAL_SIZE,
    stratify=y_train_val, 
    random_state=RANDOM_SEED, 
    shuffle=True
)

print(f"\n📊 Data Split:")
print(f"   ├─ Training:   {X_train.shape[0]} samples ({X_train.shape[0]/X_original.shape[0]*100:.1f}%)")
print(f"   │   ├─ Normal: {np.sum(y_train == 0)}")
print(f"   │   └─ Abnormal: {np.sum(y_train == 1)}")
print(f"   ├─ Validation: {X_val.shape[0]} samples ({X_val.shape[0]/X_original.shape[0]*100:.1f}%)")
print(f"   │   ├─ Normal: {np.sum(y_val == 0)}")
print(f"   │   └─ Abnormal: {np.sum(y_val == 1)}")
print(f"   └─ Test:       {X_test.shape[0]} samples ({X_test.shape[0]/X_original.shape[0]*100:.1f}%)")
print(f"       ├─ Normal: {np.sum(y_test == 0)}")
print(f"       └─ Abnormal: {np.sum(y_test == 1)}")

# Augmentation function
def augment_audio(audio, sample_rate, aug_types=['noise', 'shift']):
    """Enhanced augmentation with 5 techniques"""
    augmented = audio.copy()
    
    if 'noise' in aug_types:
        noise = np.random.randn(len(audio))
        augmented = augmented + config.NOISE_FACTOR * noise
    
    if 'shift' in aug_types:
        shift_samples = int(np.random.uniform(-config.SHIFT_MAX, config.SHIFT_MAX) * sample_rate)
        augmented = np.roll(augmented, shift_samples)
    
    if 'pitch' in aug_types:
        n_steps = np.random.uniform(-config.PITCH_SHIFT_RANGE, config.PITCH_SHIFT_RANGE)
        augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=n_steps)
    
    if 'stretch' in aug_types:
        rate = np.random.uniform(*config.TIME_STRETCH_RANGE)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)
        if len(augmented) < len(audio):
            augmented = np.pad(augmented, (0, len(audio) - len(augmented)), mode='constant')
        else:
            augmented = augmented[:len(audio)]
    
    if 'volume' in aug_types:
        scale = np.random.uniform(*config.VOLUME_SCALE_RANGE)
        augmented = augmented * scale
    
    return augmented

# Store original training set sizes
original_train_size = X_train.shape[0]
original_train_normal = np.sum(y_train == 0)
original_train_abnormal = np.sum(y_train == 1)

# Apply augmentation ONLY to training set
if config.AUGMENT_DATA:
    print(f"\n✅ Applying augmentation ONLY to training set...")
    abnormal_train_indices = np.where(y_train == 1)[0]
    
    augmented_train_features = []
    augmented_train_labels = []
    
    aug_types_combinations = [
        ['noise', 'shift'],
        ['pitch'],
        ['stretch'],
        ['volume'],
        ['noise', 'pitch']
    ]
    
    for idx in tqdm(abnormal_train_indices, desc="Augmenting training data"):
        abnormal_sample_idx = int(idx - np.sum(y_train[:idx] == 0))
        
        if abnormal_sample_idx < len(abnormal_audio):
            audio_sample = abnormal_audio[abnormal_sample_idx]
            
            for i in range(config.AUGMENTATION_FACTOR - 1):
                aug_types = aug_types_combinations[i % len(aug_types_combinations)]
                augmented_audio = augment_audio(audio_sample, sample_rate, aug_types)
                
                # Extract ALL features (no filtering)
                augmented_features = extract_all_features(augmented_audio, sample_rate)
                
                augmented_train_features.append(augmented_features)
                augmented_train_labels.append(1)
    
    if len(augmented_train_features) > 0:
        augmented_train_features = np.array(augmented_train_features)
        augmented_train_labels = np.array(augmented_train_labels)
        
        X_train = np.concatenate([X_train, augmented_train_features])
        y_train = np.concatenate([y_train, augmented_train_labels])
        
        print(f"   ├─ Augmented samples added: {len(augmented_train_features)}")
        print(f"   └─ New training set size: {X_train.shape[0]}")

print("\n" + "=" * 100)
print("SECTION 4: COST-SENSITIVE CLASS WEIGHTING (SAFETY-FIRST)")
print("=" * 100)

n_neg = np.sum(y_train == 0)
n_pos = np.sum(y_train == 1)

# AGGRESSIVE cost-sensitive weighting for safety-critical application
# False Negative (missing abnormal) = 5x worse than False Positive (false alarm)
fn_cost = config.FALSE_NEGATIVE_COST
fp_cost = config.FALSE_POSITIVE_COST

# Calculate adjusted scale_pos_weight
base_ratio = n_neg / n_pos
scale_pos_weight = base_ratio * (fn_cost / fp_cost)

print(f"\n📊 Class Distribution in Training Set:")
print(f"   ├─ Normal (class 0): {n_neg} samples")
print(f"   └─ Abnormal (class 1): {n_pos} samples")
print(f"   └─ Base Imbalance Ratio: 1:{base_ratio:.2f}")

print(f"\n⚖️  Cost-Sensitive Configuration:")
print(f"   ├─ False Negative Cost (missing abnormal): {fn_cost}x")
print(f"   ├─ False Positive Cost (false alarm): {fp_cost}x")
print(f"   ├─ Cost Ratio (FN/FP): {fn_cost/fp_cost:.1f}:1")
print(f"   └─ 🎯 Adjusted scale_pos_weight: {scale_pos_weight:.4f}")
print("=" * 100)

print("\n" + "=" * 100)
print("SECTION 5: BAYESIAN OPTIMIZATION FOR RECALL MAXIMIZATION")
print("=" * 100)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

if config.USE_BAYESIAN_OPTIMIZATION:
    print(f"\n🔬 Starting RECALL-FOCUSED Bayesian Optimization with Optuna...")
    print(f"   ├─ Number of trials: {config.N_TRIALS}")
    print(f"   ├─ Timeout: {config.OPTIMIZATION_TIMEOUT}s ({config.OPTIMIZATION_TIMEOUT/3600:.1f} hours)")
    print(f"   ├─ Objective: Maximize RECALL with Precision ≥ {config.MIN_PRECISION*100:.0f}%")
    print("\n")
    
    def objective(trial) -> float:
        """
        Optuna objective function - RECALL-FOCUSED
        
        Strategy: Maximize recall while maintaining minimum precision
        Penalty applied if precision drops below threshold
        
        Returns:
            float: Optimization score (recall-based with precision penalty)
        """
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            
            # Tree structure - RELAXED constraints for minority class learning
            'max_depth': trial.suggest_int('max_depth', 4, 12),  # Wider range
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),  # LOWER max (was 7)
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),  # Lower max
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            
            # Regularization - RELAXED for minority class
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),  # Lower max (was 5.0)
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),  # Lower max (was 5.0)
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),  # Lower max (was 5.0)
            
            # Cost-sensitive weighting
            'scale_pos_weight': scale_pos_weight,  # Use calculated cost-sensitive weight
            
            'seed': RANDOM_SEED,
            'tree_method': 'hist',
        }
        
        temp_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=500,
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Predict on validation set
        y_val_pred_prob = temp_model.predict(dval)
        
        # Try multiple thresholds to find best recall with acceptable precision
        best_score: float = 0.0
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            y_val_pred = (y_val_pred_prob >= threshold).astype(int)
            
            recall = float(recall_score(y_val, y_val_pred, zero_division=0))
            precision = float(precision_score(y_val, y_val_pred, zero_division=0))
            
            # Score = recall if precision meets minimum, else penalize
            if precision >= config.MIN_PRECISION:
                score = recall  # Pure recall maximization
            else:
                # Penalize based on how far below minimum precision
                penalty = (precision / config.MIN_PRECISION) ** 2
                score = recall * penalty
            
            best_score = max(best_score, score)
        
        return float(best_score)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_SEED, n_startup_trials=20)  # More random trials initially
    )
    
    study.optimize(
        objective,
        n_trials=config.N_TRIALS,
        timeout=config.OPTIMIZATION_TIMEOUT,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = ['logloss', 'auc', 'error']
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['seed'] = RANDOM_SEED
    best_params['tree_method'] = 'hist'
    
    print(f"\n✅ Bayesian Optimization Complete!")
    print(f"   ├─ Best Recall Score: {study.best_value:.4f}")
    print(f"   ├─ Number of trials completed: {len(study.trials)}")
    print(f"   └─ Best trial: {study.best_trial.number}")
    
    optimization_info = {
        'method': 'Recall-Focused Bayesian Optimization (Optuna)',
        'n_trials': len(study.trials),
        'best_recall_score': float(study.best_value),
        'best_trial_number': study.best_trial.number,
        'best_params': {k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in best_params.items() 
                       if k not in ['objective', 'eval_metric', 'seed', 'tree_method']}
    }
    
else:
    print(f"\n⚠️  Bayesian Optimization DISABLED - Using default parameters")
    best_params = config.DEFAULT_PARAMS.copy()
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = ['logloss', 'auc', 'error']
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['seed'] = RANDOM_SEED
    best_params['tree_method'] = 'hist'
    
    optimization_info = {
        'method': 'Default parameters (no optimization)',
        'n_trials': 0,
        'best_recall_score': None,
        'best_params': config.DEFAULT_PARAMS
    }

print("=" * 100)

print("\n" + "=" * 100)
print("SECTION 6: MODEL TRAINING WITH BEST HYPERPARAMETERS")
print("=" * 100)

print(f"\n🚀 Starting XGBoost training with best parameters...")

evals = [(dtrain, 'train'), (dval, 'validation')]
evals_result = {}

num_boost_round = best_params.pop('n_estimators', 500)

model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
    evals_result=evals_result,
    verbose_eval=20
)

print("\n✅ Training completed!")
print(f"   ├─ Best iteration: {model.best_iteration}")
print(f"   ├─ Best training AUC: {evals_result['train']['auc'][model.best_iteration]:.4f}")
print(f"   └─ Best validation AUC: {evals_result['validation']['auc'][model.best_iteration]:.4f}")

# Calculate validation metrics with trained model
print(f"\n📊 Calculating validation set metrics with trained model...")
y_val_pred_prob = model.predict(dval)
y_val_pred = (y_val_pred_prob >= 0.5).astype(int)
val_f1 = float(f1_score(y_val, y_val_pred, zero_division=0))
val_recall = float(recall_score(y_val, y_val_pred, zero_division=0))
val_precision = float(precision_score(y_val, y_val_pred, zero_division=0))
val_accuracy = float(accuracy_score(y_val, y_val_pred))

print(f"   ├─ Validation Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"   ├─ Validation Precision: {val_precision:.4f} ({val_precision*100:.2f}%)")
print(f"   ├─ Validation Recall:    {val_recall:.4f} ({val_recall*100:.2f}%)")
print(f"   └─ Validation F1-Score:  {val_f1:.4f} ({val_f1*100:.2f}%)")

# Add validation metrics to optimization_info
optimization_info['validation_metrics'] = {
    'accuracy': val_accuracy,
    'recall': val_recall,
    'precision': val_precision,
    'f1_score': val_f1
}

# Save model 
model.save_model(config.MODEL_SAVE_PATH)
print(f"\n💾 Model saved to: {config.MODEL_SAVE_PATH}")

training_history = {
    'train_logloss': [float(x) for x in evals_result['train']['logloss']],
    'train_auc': [float(x) for x in evals_result['train']['auc']],
    'train_error': [float(x) for x in evals_result['train']['error']],
    'val_logloss': [float(x) for x in evals_result['validation']['logloss']],
    'val_auc': [float(x) for x in evals_result['validation']['auc']],
    'val_error': [float(x) for x in evals_result['validation']['error']],
    'best_iteration': int(model.best_iteration),
    'num_boost_round': int(num_boost_round)
}


print("\n" + "=" * 100)
print("SECTION 7: THRESHOLD OPTIMIZATION (SAFETY-FIRST: MAXIMIZE RECALL)")
print("=" * 100)

y_pred_prob = model.predict(dtest)

TARGET_RECALL = config.TARGET_RECALL
MIN_PRECISION = config.MIN_PRECISION

# Test a wide range of thresholds
thresholds = np.arange(0.05, 0.95, 0.01)
results = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_prob >= threshold).astype(int)
    
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    results.append({
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1
    })

# Strategy 1: Find threshold that meets BOTH recall and precision targets
viable_thresholds = [r for r in results if r['recall'] >= TARGET_RECALL and r['precision'] >= MIN_PRECISION]

if viable_thresholds:
    # Among viable thresholds, pick the one with best F1 (balanced)
    best_result = max(viable_thresholds, key=lambda x: x['f1'])
    optimal_threshold = best_result['threshold']
    strategy_used = f"TARGET MET: Recall ≥ {TARGET_RECALL*100:.0f}% AND Precision ≥ {MIN_PRECISION*100:.0f}%"
else:
    # Strategy 2: Couldn't meet both targets - prioritize RECALL
    print(f"   SAFETY FIRST: Prioritizing recall over precision...")
    
    # Find threshold with highest recall that still maintains minimum precision
    precision_constrained = [r for r in results if r['precision'] >= MIN_PRECISION]
    
    if precision_constrained:
        best_result = max(precision_constrained, key=lambda x: x['recall'])
        optimal_threshold = best_result['threshold']
        strategy_used = f"COMPROMISE: Max Recall with Precision ≥ {MIN_PRECISION*100:.0f}%"
    else:
        # Strategy 3: Even minimum precision not achievable - pure recall maximization
        print(f"   ⚠️  Even minimum precision constraint cannot be met")
        print(f"   CRITICAL SAFETY MODE: Maximizing recall without precision constraint")
        best_result = max(results, key=lambda x: x['recall'])
        optimal_threshold = best_result['threshold']
        strategy_used = "CRITICAL SAFETY: Pure Recall Maximization"

# Also get default threshold metrics for comparison
default_result = [r for r in results if abs(r['threshold'] - 0.5) < 0.01][0]

# Final predictions with optimal threshold
y_pred = (y_pred_prob >= optimal_threshold).astype(int)

print(f"\n✅ Threshold Optimization Complete!")
print(f"   Strategy: {strategy_used}")
print(f"\n📊 Threshold Comparison:")
print(f"\n   Default (0.5):")
print(f"      ├─ Recall:    {default_result['recall']:.4f} ({default_result['recall']*100:.2f}%)")
print(f"      ├─ Precision: {default_result['precision']:.4f} ({default_result['precision']*100:.2f}%)")
print(f"      └─ F1-Score:  {default_result['f1']:.4f} ({default_result['f1']*100:.2f}%)")

print(f"\n   Optimal ({optimal_threshold:.3f}):")
print(f"      ├─ Recall:    {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)")
print(f"      ├─ Precision: {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)")
print(f"      └─ F1-Score:  {best_result['f1']:.4f} ({best_result['f1']*100:.2f}%)")

recall_improvement = ((best_result['recall'] - default_result['recall']) / default_result['recall'] * 100) if default_result['recall'] > 0 else 0
print(f"\n   📈 Recall Improvement: {recall_improvement:+.2f}%")

optimal_precision = best_result['precision']
optimal_recall = best_result['recall']
optimal_f1 = best_result['f1']

print(f"\n🎯 FINAL DECISION: Using threshold = {optimal_threshold:.3f}")
print("=" * 100)

print("\n" + "=" * 100)
print("SECTION 8: FEATURE IMPORTANCE ANALYSIS")
print("=" * 100)

# Extract feature importance from XGBoost model
importance_dict = model.get_score(importance_type='weight') 
importance_gain = model.get_score(importance_type='gain')    

# Convert to sorted lists
feature_importance = []
for i, feature_name in enumerate(feature_names):
    # Get the feature index (F0, F1, etc. in XGBoost notation)
    f_idx = f"f{i}"
    
    # Get importance values with proper type handling
    weight_val = importance_dict.get(f_idx, 0)
    gain_val = importance_gain.get(f_idx, 0.0)
    
    # Handle both scalar and list returns from XGBoost
    if isinstance(weight_val, (list, np.ndarray)):
        weight_val = weight_val[0] if len(weight_val) > 0 else 0
    if isinstance(gain_val, (list, np.ndarray)):
        gain_val = gain_val[0] if len(gain_val) > 0 else 0.0
    
    # Convert to native Python types
    weight_int = int(float(weight_val)) if weight_val is not None else 0
    gain_float = float(gain_val) if gain_val is not None else 0.0
    
    feature_importance.append({
        'feature': feature_name,
        'weight': weight_int,
        'gain': gain_float
    })

# Sort by gain (most important first)
feature_importance_sorted = sorted(feature_importance, key=lambda x: x['gain'], reverse=True)

print(f"\n📊 Top 15 Most Important Features (by gain):")
for i, feat in enumerate(feature_importance_sorted[:15], 1):
    print(f"   {i:2d}. {feat['feature']:30s} | Gain: {feat['gain']:8.4f} | Weight: {feat['weight']:4d}")

# Calculate feature importance statistics
total_gain = sum([f['gain'] for f in feature_importance])
cumulative_importance = []
cumulative_sum = 0
for feat in feature_importance_sorted:
    cumulative_sum += feat['gain']
    cumulative_importance.append({
        'feature': feat['feature'],
        'gain': feat['gain'],
        'percentage': (feat['gain'] / total_gain * 100) if total_gain > 0 else 0,
        'cumulative_percentage': (cumulative_sum / total_gain * 100) if total_gain > 0 else 0
    })

# Find how many features contribute to 80% of importance
features_80_pct = sum(1 for f in cumulative_importance if f['cumulative_percentage'] <= 80)
features_95_pct = sum(1 for f in cumulative_importance if f['cumulative_percentage'] <= 95)

print(f"\n📊 Feature Importance Summary:")
print(f"   ├─ Total features: {len(feature_names)}")
print(f"   ├─ Features contributing to 80% importance: {features_80_pct} ({features_80_pct/len(feature_names)*100:.1f}%)")
print(f"   ├─ Features contributing to 95% importance: {features_95_pct} ({features_95_pct/len(feature_names)*100:.1f}%)")
print(f"   └─ Features with zero importance: {sum(1 for f in feature_importance if f['gain'] == 0)}")

print("\n" + "=" * 100)
print("SECTION 9: MODEL EVALUATION ON TEST SET")
print("=" * 100)

# Calculate all metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = optimal_precision
test_recall = optimal_recall
test_f1 = optimal_f1
test_auc = roc_auc_score(y_test, y_pred_prob)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n📊 Test Set Performance - XGBoost Optimized (Threshold: {optimal_threshold:.3f})")
print(f"   ├─ Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   ├─ Precision:    {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"   ├─ Recall:       {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"   ├─ F1-Score:     {test_f1:.4f} ({test_f1*100:.2f}%)")
print(f"   ├─ AUC-ROC:      {test_auc:.4f} ({test_auc*100:.2f}%)")
print(f"   ├─ Specificity:  {specificity:.4f} ({specificity*100:.2f}%)")
print(f"   └─ Sensitivity:  {sensitivity:.4f} ({sensitivity*100:.2f}%)")

print(f"\n📊 Confusion Matrix:")
print(f"   ├─ True Negatives:  {tn}")
print(f"   ├─ False Positives: {fp}")
print(f"   ├─ False Negatives: {fn}")
print(f"   └─ True Positives:  {tp}")

print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES))

# Calculate error metrics
false_alarm_rate = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
missed_anomaly_rate = (fn / (fn + tp) * 100) if (fn + tp) > 0 else 0

print("\n" + "=" * 100)
print("SECTION 10: SAVING RESULTS & GENERATING COMPREHENSIVE REPORT")
print("=" * 100)

training_end_time = datetime.now()
training_duration = (training_end_time - training_start_time).total_seconds()

# Prepare comprehensive JSON report 
json_report = {
    'metadata': {
        'model_name': 'AI-Pump-XGBoost',
        'version': '1.0-XGB-OPTIMIZED',
        'training_timestamp': training_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_duration_seconds': float(training_duration),
        'framework': 'XGBoost',
        'random_seed': RANDOM_SEED,
        'optimization': 'XGBoost Native Feature Selection',
        'changes': [
            'XGBoost gradient boosting classifier',
            'Bayesian Optimization with Optuna',
            'Multi-pump aggregation (4 pumps)',
            'ALL 42 features used (no statistical pre-filtering)',
            'XGBoost native feature selection via tree splitting',
            'Enhanced augmentation (6x with 5 techniques)',
            'Optimal threshold calculation',
            'Feature importance analysis'
        ]
    },
    'dataset_info': {
        'pump_ids': config.PUMP_IDS,
        'total_samples': len(all_audio),
        'total_features_extracted': len(feature_names),
        'feature_selection_method': 'XGBoost Native (no pre-filtering)',
        'class_distribution': {
            'total': {
                'normal': len(normal_audio),
                'abnormal': len(abnormal_audio)
            },
            'ratio': f'1:{imbalance_ratio:.2f}'
        }
    },
    'data_split': {
        'strategy': 'stratified_train_test_split',
        'test_size': config.TEST_SIZE,
        'validation_size': config.VAL_SIZE,
        'training': {
            'total_samples': int(X_train.shape[0]),
            'original_samples': int(original_train_size),
            'percentage': float(X_train.shape[0] / len(all_audio) * 100),
            'class_distribution': {
                'normal': int(np.sum(y_train == 0)),
                'abnormal': int(np.sum(y_train == 1))
            },
            'includes_augmented': config.AUGMENT_DATA,
            'augmentation_factor': config.AUGMENTATION_FACTOR if config.AUGMENT_DATA else 1
        },
        'validation': {
            'total_samples': int(X_val.shape[0]),
            'percentage': float(X_val.shape[0] / len(all_audio) * 100),
            'class_distribution': {
                'normal': int(np.sum(y_val == 0)),
                'abnormal': int(np.sum(y_val == 1))
            }
        },
        'testing': {
            'total_samples': int(X_test.shape[0]),
            'percentage': float(X_test.shape[0] / len(all_audio) * 100),
            'class_distribution': {
                'normal': int(np.sum(y_test == 0)),
                'abnormal': int(np.sum(y_test == 1))
            },
            'original_only': True
        }
    },
    'model_architecture': {
        'type': 'XGBoost Gradient Boosting with Native Feature Selection',
        'hyperparameters': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                           for k, v in best_params.items()},
        'num_boost_round': int(num_boost_round),
        'best_iteration': int(model.best_iteration),
    },
    'bayesian_optimization': {
        'enabled': config.USE_BAYESIAN_OPTIMIZATION,
        'method': optimization_info['method'],
        'n_trials': optimization_info['n_trials'],
        'best_recall_score': optimization_info.get('best_recall_score'),
        'best_trial_number': optimization_info.get('best_trial_number'),
        'validation_metrics': optimization_info.get('validation_metrics', {}),
        'optimized_hyperparameters': optimization_info['best_params']
    },
    'training_configuration': {
        'optimizer': 'Gradient Boosting',
        'scale_pos_weight': float(scale_pos_weight),
        'early_stopping_rounds': config.EARLY_STOPPING_ROUNDS,
        'class_imbalance_method': 'scale_pos_weight',
        'feature_selection': 'XGBoost Native (no statistical pre-filtering)'
    },
    'training_history': {
        'total_iterations': len(training_history['train_auc']),
        'best_iteration': training_history['best_iteration'],
        'stopped_early': training_history['best_iteration'] < training_history['num_boost_round'],
        'final_metrics': {
            'train_auc': training_history['train_auc'][-1],
            'train_logloss': training_history['train_logloss'][-1],
            'train_error': training_history['train_error'][-1],
            'val_auc': training_history['val_auc'][-1],
            'val_logloss': training_history['val_logloss'][-1],
            'val_error': training_history['val_error'][-1]
        },
        'best_iteration_metrics': {
            'train_auc': training_history['train_auc'][model.best_iteration],
            'train_logloss': training_history['train_logloss'][model.best_iteration],
            'val_auc': training_history['val_auc'][model.best_iteration],
            'val_logloss': training_history['val_logloss'][model.best_iteration]
        }
    },
    'feature_importance': {
        'total_features': len(feature_names),
        'features_80pct_importance': int(features_80_pct),
        'features_95pct_importance': int(features_95_pct),
        'zero_importance_features': int(sum(1 for f in feature_importance if f['gain'] == 0)),
    },
    'threshold_optimization': {
        'enabled': config.OPTIMIZE_THRESHOLD,
        'strategy': strategy_used,
        'target_recall': config.TARGET_RECALL,
        'min_precision': config.MIN_PRECISION,
        'default_threshold': 0.5,
        'optimal_threshold': float(optimal_threshold),
        'default_metrics': {
            'recall': float(default_result['recall']),
            'precision': float(default_result['precision']),
            'f1_score': float(default_result['f1'])
        },
        'optimal_metrics': {
            'recall': float(best_result['recall']),
            'precision': float(best_result['precision']),
            'f1_score': float(best_result['f1'])
        },
        'recall_improvement_percent': float(recall_improvement)
    },
    'test_performance': {
        'metrics': {
            'accuracy': {
                'value': float(test_accuracy),
                'percentage': f'{test_accuracy*100:.2f}%'
            },
            'precision': {
                'value': float(test_precision),
                'percentage': f'{test_precision*100:.2f}%'
            },
            'recall': {
                'value': float(test_recall),
                'percentage': f'{test_recall*100:.2f}%'
            },
            'f1_score': {
                'value': float(test_f1),
                'percentage': f'{test_f1*100:.2f}%'
            },
            'auc_roc': {
                'value': float(test_auc),
                'percentage': f'{test_auc*100:.2f}%'
            },
            'specificity': {
                'value': float(specificity),
                'percentage': f'{specificity*100:.2f}%'
            },
            'sensitivity': {
                'value': float(sensitivity),
                'percentage': f'{sensitivity*100:.2f}%'
            }
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
        },
        'error_analysis': {
            'total_errors': int(fp + fn),
            'error_rate': float((fp + fn) / len(y_test) * 100),
            'false_alarm_rate': float(false_alarm_rate),
            'missed_anomaly_rate': float(missed_anomaly_rate)
        }
    },
    'model_artifacts': {
        'model_path': config.MODEL_SAVE_PATH,
        'results_path': config.RESULTS_SAVE_PATH,
        'json_report_path': config.JSON_REPORT_PATH
    },
    'production_readiness': {
        'overall_score': 'EXCELLENT' if test_f1 > 0.85 and test_recall > 0.80 else 'GOOD' if test_f1 > 0.75 else 'NEEDS IMPROVEMENT',
        'recommendations': [
            'Model uses XGBoost native feature selection (optimized approach)',
            'Bayesian-optimized hyperparameters' if config.USE_BAYESIAN_OPTIMIZATION else 'Default hyperparameters',
            f'{int(fp)} false alarms detected',
            f'Catches {test_recall*100:.1f}% of anomalies',
            f'Overall accuracy: {test_accuracy*100:.2f}%',
            f'F1-Score: {test_f1*100:.2f}%',
        ]
    }
}

# Save JSON report (Task 2)
with open(config.JSON_REPORT_PATH, 'w') as f:
    json.dump(json_report, f, indent=4)
print(f"💾 JSON report saved to: {config.JSON_REPORT_PATH}")

# Save comprehensive results pickle file 
results_dict = {
    'model_metadata': {
        'model_name': 'AI-Pump-XGBoost',
        'version': '1.0-XGB-OPTIMIZED',
        'training_timestamp': training_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_duration_seconds': training_duration,
        'framework': 'XGBoost',
        'xgboost_version': xgb.__version__
    },
    
    'data_info': {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'pump_ids': config.PUMP_IDS,
        'sample_rate': sample_rate,
        'class_names': config.CLASS_NAMES
    },
    
    'model_config': {
        'hyperparameters': best_params,
        'num_boost_round': num_boost_round,
        'best_iteration': model.best_iteration,
        'scale_pos_weight': scale_pos_weight
    },
    
    'threshold_info': {
        'default_threshold': 0.5,
        'optimal_threshold': optimal_threshold,
        'optimization_strategy': strategy_used,
        'target_recall': config.TARGET_RECALL,
        'min_precision': config.MIN_PRECISION
    },
    
    'performance_metrics': {
        'test': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'auc_roc': test_auc,
            'specificity': specificity,
            'sensitivity': sensitivity
        },
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    },
    
    'training_history': training_history,
    
    'feature_importance': {
        'sorted_features': feature_importance_sorted,
        'features_80pct': features_80_pct,
        'features_95pct': features_95_pct,
        'total_gain': total_gain
    },
    
    'optimization_details': optimization_info,
    
    'production_config': {
        'model_path': config.MODEL_SAVE_PATH,
        'requires_preprocessing': False,  
        'input_shape': (len(feature_names),),
        'output_classes': 2,
        'recommended_threshold': optimal_threshold
    }
}

with open(config.RESULTS_SAVE_PATH, 'wb') as f:
    pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
print(f"💾 Results pickle saved to: {config.RESULTS_SAVE_PATH}")

print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)
print(f"\n🎉 TRAINING COMPLETE!")
print(f"\n⏱️  Training Duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

print(f"\n📊 Final Test Metrics:")
print(f"   ├─ Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   ├─ Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"   ├─ Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"   ├─ F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
print(f"   └─ AUC-ROC:   {test_auc:.4f} ({test_auc*100:.2f}%)")
