"""
Data loader module for ECG datasets.
Handles loading from various sources including MIT-BIH, CSV, and custom formats.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import wfdb
from sklearn.model_selection import train_test_split
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config


class ECGDataLoader:
    """Load and manage ECG datasets."""
    
    def __init__(self, config=None):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
    
    def load_mitbih_record(self, record_path, annotation_path=None):
        """
        Load a single MIT-BIH record.
        
        Args:
            record_path: Path to the record file
            annotation_path: Path to annotation file (optional)
            
        Returns:
            signal: ECG signal array
            annotation: Annotation data (if provided)
        """
        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, 0]  # First channel
            
            if annotation_path:
                annotation = wfdb.rdann(record_path, 'atr')
                return signal, annotation
            
            return signal, None
        except Exception as e:
            print(f"Error loading MIT-BIH record: {e}")
            return None, None
    
    def load_dataset_from_csv(self, csv_path, signal_column='signal', label_column='label'):
        """
        Load ECG dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            signal_column: Name of column containing signal data
            label_column: Name of column containing labels
            
        Returns:
            X: Signal data array
            y: Labels array
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Handle different CSV formats
            if signal_column in df.columns:
                X = np.array(df[signal_column].tolist())
            else:
                # Assume all columns except label are signal
                label_cols = [label_column] if label_column in df.columns else []
                signal_cols = [col for col in df.columns if col not in label_cols]
                X = df[signal_cols].values
            
            # Load labels if available
            if label_column in df.columns:
                y = df[label_column].values
            else:
                y = None
            
            return X, y
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None, None
    
    def load_dataset_from_directory(self, directory_path, file_pattern='*.csv'):
        """
        Load multiple ECG files from a directory.
        
        Args:
            directory_path: Path to directory containing ECG files
            file_pattern: File pattern to match
            
        Returns:
            X: Combined signal data
            y: Combined labels
        """
        directory = Path(directory_path)
        files = list(directory.glob(file_pattern))
        
        all_signals = []
        all_labels = []
        
        for file_path in files:
            X, y = self.load_dataset_from_csv(file_path)
            if X is not None:
                all_signals.append(X)
                if y is not None:
                    all_labels.append(y)
        
        if all_signals:
            X = np.vstack(all_signals)
            y = np.concatenate(all_labels) if all_labels else None
            return X, y
        
        return None, None
    
    def create_train_test_split(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input data
            y: Labels
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, data_dict, output_dir):
        """
        Save processed data to disk.
        
        Args:
            data_dict: Dictionary containing data arrays
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for key, value in data_dict.items():
            file_path = output_path / f"{key}.npy"
            np.save(file_path, value)
            print(f"Saved {key} to {file_path}")
    
    def load_processed_data(self, input_dir):
        """
        Load processed data from disk.
        
        Args:
            input_dir: Input directory path
            
        Returns:
            Dictionary containing loaded arrays
        """
        input_path = Path(input_dir)
        data_dict = {}
        
        for file_path in input_path.glob("*.npy"):
            key = file_path.stem
            data_dict[key] = np.load(file_path)
            print(f"Loaded {key} from {file_path}")
        
        return data_dict


def download_sample_mitbih_data(output_dir='data/raw/mitdb'):
    """
    Download sample MIT-BIH data for testing.
    
    Args:
        output_dir: Directory to save downloaded data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading sample MIT-BIH records...")
    
    # Download a few sample records
    sample_records = ['100', '101', '102']
    
    for record in sample_records:
        try:
            wfdb.dl_database('mitdb', output_path, records=[record])
            print(f"Downloaded record {record}")
        except Exception as e:
            print(f"Error downloading record {record}: {e}")


if __name__ == "__main__":
    # Example usage
    loader = ECGDataLoader()
    
    # Download sample data
    print("Downloading sample MIT-BIH data...")
    download_sample_mitbih_data()
    
    print("\nData loader initialized successfully!")
