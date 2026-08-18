"""
Configuration module for cardiac abnormality detection system.
Centralizes all hyperparameters and settings.
"""

from pathlib import Path


class Config:
    """Configuration class for the entire project."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = PROJECT_ROOT / 'models'
    LOGS_DIR = MODELS_DIR / 'logs'
    
    # Data parameters
    SAMPLING_RATE = 360  # Hz
    SIGNAL_LENGTH = 3600  # Number of samples (10 seconds at 360 Hz)
    
    # Preprocessing parameters
    FILTER_LOWCUT = 0.5  # Hz
    FILTER_HIGHCUT = 50.0  # Hz
    FILTER_ORDER = 4
    BASELINE_FILTER_CUTOFF = 0.5  # Hz
    
    # Normalization
    NORMALIZATION_METHOD = 'standard'  # 'standard' or 'minmax'
    
    # Model hyperparameters - CNN
    CNN_FILTERS = [64, 128, 256]
    CNN_KERNEL_SIZE = 7
    CNN_POOL_SIZE = 2
    CNN_DROPOUT = 0.5
    
    # Model hyperparameters - LSTM
    LSTM_UNITS = [128, 64]
    LSTM_DROPOUT = 0.3
    LSTM_RECURRENT_DROPOUT = 0.3
    
    # Model hyperparameters - Hybrid
    HYBRID_CNN_FILTERS = [32, 64]
    HYBRID_LSTM_UNITS = 64
    HYBRID_DROPOUT = 0.4
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    
    # Data split
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
    
    # Class labels
    CLASS_LABELS = {
        0: 'Normal',
        1: 'Arrhythmia',
        2: 'Myocardial Infarction',
        3: 'Other Abnormality'
    }
    NUM_CLASSES = len(CLASS_LABELS)
    
    # Feature extraction
    FEATURE_NAMES = [
        'mean', 'std', 'min', 'max', 'skewness', 'kurtosis',
        'mean_rr', 'std_rr', 'heart_rate',
        'lf_power', 'hf_power', 'lf_hf_ratio',
        'r_peak_amplitude', 'qrs_width',
        'sdnn', 'rmssd', 'nn50', 'pnn50'
    ]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                         cls.MODELS_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_save_path(cls, model_name):
        """Get path for saving a model."""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return cls.MODELS_DIR / f"{model_name}_{timestamp}.keras"
    
    def __repr__(self):
        return f"Config(sampling_rate={self.SAMPLING_RATE}, signal_length={self.SIGNAL_LENGTH})"


# Top-level aliases for direct imports across modules
PROJECT_ROOT = Config.PROJECT_ROOT
DATA_DIR = Config.DATA_DIR
RAW_DATA_DIR = Config.RAW_DATA_DIR
PROCESSED_DATA_DIR = Config.PROCESSED_DATA_DIR
MODELS_DIR = Config.MODELS_DIR
LOGS_DIR = Config.LOGS_DIR
SAMPLING_RATE = Config.SAMPLING_RATE
SIGNAL_LENGTH = Config.SIGNAL_LENGTH
CLASS_LABELS = Config.CLASS_LABELS

PREPROCESSING_CONFIG = {
    'sampling_rate': Config.SAMPLING_RATE,
    'lowcut': Config.FILTER_LOWCUT,
    'highcut': Config.FILTER_HIGHCUT,
    'filter_order': Config.FILTER_ORDER,
    'baseline_cutoff': Config.BASELINE_FILTER_CUTOFF,
    'normalization': Config.NORMALIZATION_METHOD,
    'target_length': Config.SIGNAL_LENGTH,
}

FEATURE_CONFIG = {
    'extract_time_domain': True,
    'extract_frequency_domain': True,
    'extract_morphological': True,
    'extract_hrv': True,
}


if __name__ == "__main__":
    # Create directories
    Config.create_directories()
    print("Configuration initialized and directories created!")
    print(f"Project root: {Config.PROJECT_ROOT}")
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Models directory: {Config.MODELS_DIR}")
