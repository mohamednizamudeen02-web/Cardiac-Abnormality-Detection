"""
Training pipeline for ECG classification models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config


def create_callbacks(model_name, config=None):
    if config is None:
        config = Config()
    config.create_directories()
    checkpoint_path = config.get_model_save_path(f"{model_name}_best")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', 
                                save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', 
                              patience=config.EARLY_STOPPING_PATIENCE,
                              restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                 factor=config.REDUCE_LR_FACTOR,
                                 patience=config.REDUCE_LR_PATIENCE,
                                 min_lr=1e-7, verbose=1)
    return [checkpoint, early_stop, reduce_lr]


def train_model(model, X_train, y_train, X_val, y_val, model_name='model', config=None):
    if config is None:
        config = Config()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    callbacks = create_callbacks(model_name, config)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                       callbacks=callbacks, verbose=1)
    return history
