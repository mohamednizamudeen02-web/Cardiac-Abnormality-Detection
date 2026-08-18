"""
1D CNN models for ECG classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config


def build_1d_cnn(input_shape, num_classes, config=None):
    if config is None:
        config = Config()
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(config.CNN_FILTERS[0], config.CNN_KERNEL_SIZE, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(config.CNN_POOL_SIZE),
        layers.Dropout(0.2),
        layers.Conv1D(config.CNN_FILTERS[1], config.CNN_KERNEL_SIZE, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(config.CNN_POOL_SIZE),
        layers.Dropout(0.3),
        layers.Conv1D(config.CNN_FILTERS[2], config.CNN_KERNEL_SIZE, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(config.CNN_DROPOUT),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
