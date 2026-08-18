"""
Hybrid CNN-LSTM model for ECG classification.
Combines CNN for feature extraction with LSTM for temporal modeling.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import MODEL_CONFIG, NUM_CLASSES


def build_hybrid_model(
    input_shape: Tuple[int, int],
    num_classes: int = NUM_CLASSES,
    config: dict = None
) -> keras.Model:
    """
    Build hybrid CNN-LSTM model for ECG classification.
    CNN layers extract spatial features, LSTM captures temporal dependencies.
    
    Args:
        input_shape: Shape of input (signal_length, n_channels)
        num_classes: Number of output classes
        config: Model configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = MODEL_CONFIG['hybrid']
    
    model = models.Sequential(name='ECG_Hybrid_CNN_LSTM')
    
    # CNN feature extraction layers
    model.add(layers.Conv1D(
        filters=config['cnn_filters'][0],
        kernel_size=config['cnn_kernel_size'],
        activation='relu',
        padding='same',
        input_shape=input_shape,
        name='conv1'
    ))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool1'))
    model.add(layers.Dropout(0.3, name='dropout1'))
    
    model.add(layers.Conv1D(
        filters=config['cnn_filters'][1],
        kernel_size=config['cnn_kernel_size'],
        activation='relu',
        padding='same',
        name='conv2'
    ))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool2'))
    model.add(layers.Dropout(0.3, name='dropout2'))
    
    model.add(layers.Conv1D(
        filters=config['cnn_filters'][2],
        kernel_size=config['cnn_kernel_size'],
        activation='relu',
        padding='same',
        name='conv3'
    ))
    model.add(layers.BatchNormalization(name='bn3'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool3'))
    model.add(layers.Dropout(0.3, name='dropout3'))
    
    # LSTM temporal modeling layers
    model.add(layers.Bidirectional(
        layers.LSTM(
            config['lstm_units'],
            return_sequences=True,
            dropout=config['dropout_rate'],
            name='lstm1'
        ),
        name='bi_lstm1'
    ))
    
    model.add(layers.Bidirectional(
        layers.LSTM(
            config['lstm_units'] // 2,
            return_sequences=False,
            dropout=config['dropout_rate'],
            name='lstm2'
        ),
        name='bi_lstm2'
    ))
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu', name='dense1'))
    model.add(layers.Dropout(0.5, name='dropout4'))
    
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


def build_attention_hybrid_model(
    input_shape: Tuple[int, int],
    num_classes: int = NUM_CLASSES
) -> keras.Model:
    """
    Build hybrid model with attention mechanism.
    Attention helps the model focus on important parts of the ECG signal.
    
    Args:
        input_shape: Shape of input (signal_length, n_channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # CNN feature extraction
    x = layers.Conv1D(64, 5, activation='relu', padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(2, name='pool1')(x)
    
    x = layers.Conv1D(128, 5, activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(2, name='pool2')(x)
    
    # LSTM with attention
    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, name='lstm'),
        name='bi_lstm'
    )(x)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh', name='attention_dense')(lstm_out)
    attention = layers.Flatten(name='attention_flatten')(attention)
    attention = layers.Activation('softmax', name='attention_softmax')(attention)
    attention = layers.RepeatVector(128, name='attention_repeat')(attention)
    attention = layers.Permute([2, 1], name='attention_permute')(attention)
    
    # Apply attention
    attended = layers.Multiply(name='attention_multiply')([lstm_out, attention])
    attended = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(attended)
    
    # Dense layers
    x = layers.Dense(64, activation='relu', name='dense1')(attended)
    x = layers.Dropout(0.5, name='dropout')(x)
    
    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ECG_Attention_Hybrid')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Building Hybrid CNN-LSTM model...")
    model = build_hybrid_model(input_shape=(1000, 1), num_classes=4)
    model.summary()
    
    print("\n" + "="*50)
    print("Building Attention-based Hybrid model...")
    attention_model = build_attention_hybrid_model(input_shape=(1000, 1), num_classes=4)
    attention_model.summary()
    
    print("\n[OK] Hybrid models built successfully!")
