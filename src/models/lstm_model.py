"""
LSTM model for ECG classification.
Long Short-Term Memory network for temporal pattern recognition in ECG signals.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import MODEL_CONFIG, NUM_CLASSES


def build_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int = NUM_CLASSES,
    config: dict = None
) -> keras.Model:
    """
    Build LSTM model for ECG classification.
    
    Args:
        input_shape: Shape of input (signal_length, n_channels)
        num_classes: Number of output classes
        config: Model configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = MODEL_CONFIG['lstm']
    
    model = models.Sequential(name='ECG_LSTM')
    
    # First LSTM layer
    if config['bidirectional']:
        model.add(layers.Bidirectional(
            layers.LSTM(
                config['units'][0],
                return_sequences=True,
                dropout=config['dropout_rate'],
                recurrent_dropout=config['recurrent_dropout'],
                name='lstm1'
            ),
            input_shape=input_shape,
            name='bi_lstm1'
        ))
    else:
        model.add(layers.LSTM(
            config['units'][0],
            return_sequences=True,
            dropout=config['dropout_rate'],
            recurrent_dropout=config['recurrent_dropout'],
            input_shape=input_shape,
            name='lstm1'
        ))
    
    model.add(layers.BatchNormalization(name='bn1'))
    
    # Second LSTM layer
    if config['bidirectional']:
        model.add(layers.Bidirectional(
            layers.LSTM(
                config['units'][1],
                return_sequences=False,
                dropout=config['dropout_rate'],
                recurrent_dropout=config['recurrent_dropout'],
                name='lstm2'
            ),
            name='bi_lstm2'
        ))
    else:
        model.add(layers.LSTM(
            config['units'][1],
            return_sequences=False,
            dropout=config['dropout_rate'],
            recurrent_dropout=config['recurrent_dropout'],
            name='lstm2'
        ))
    
    model.add(layers.BatchNormalization(name='bn2'))
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu', name='dense1'))
    model.add(layers.Dropout(0.5, name='dropout'))
    
    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        loss = 'sparse_categorical_crossentropy'
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def build_gru_model(
    input_shape: Tuple[int, int],
    num_classes: int = NUM_CLASSES
) -> keras.Model:
    """
    Build GRU model as an alternative to LSTM.
    GRU is computationally more efficient than LSTM.
    
    Args:
        input_shape: Shape of input (signal_length, n_channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='ECG_GRU')
    
    # First GRU layer
    model.add(layers.Bidirectional(
        layers.GRU(
            128,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.3,
            name='gru1'
        ),
        input_shape=input_shape,
        name='bi_gru1'
    ))
    model.add(layers.BatchNormalization(name='bn1'))
    
    # Second GRU layer
    model.add(layers.Bidirectional(
        layers.GRU(
            64,
            return_sequences=False,
            dropout=0.3,
            recurrent_dropout=0.3,
            name='gru2'
        ),
        name='bi_gru2'
    ))
    model.add(layers.BatchNormalization(name='bn2'))
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu', name='dense1'))
    model.add(layers.Dropout(0.5, name='dropout'))
    
    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        loss = 'sparse_categorical_crossentropy'
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Building LSTM model...")
    model = build_lstm_model(input_shape=(1000, 1), num_classes=4)
    model.summary()
    
    print("\n" + "="*50)
    print("Building GRU model...")
    gru_model = build_gru_model(input_shape=(1000, 1), num_classes=4)
    gru_model.summary()
    
    print("\n[OK] LSTM/GRU models built successfully!")
