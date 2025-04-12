import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np

def lr_schedule(epoch):
    """Learning rate scheduler for adaptive training."""
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return max(lr, 1e-6)

def create_model(input_shape, complexity='standard'):
    """
    Create an enhanced neural network model for cyber threat detection.
    
    Args:
        input_shape (tuple): Shape of the input data
        complexity (str): Model complexity level ('standard', 'complex')
        
    Returns:
        model: Compiled TensorFlow model
    """
    inputs = Input(shape=input_shape)
    
    if complexity == 'complex':
        # Enhanced architecture for more complex threat patterns
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
    else:
        # Standard architecture
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use Adam optimizer with learning rate scheduling
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def get_callbacks(model_path='models/threat_detection_model'):
    """Get callbacks for model training with checkpointing and LR scheduling."""
    return [
        ModelCheckpoint(
            filepath=model_path + '_{epoch:02d}.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        LearningRateScheduler(lr_schedule)
    ]

def update_model(model, new_data_x, new_data_y, epochs=5):
    """Update the model with new data in real-time.
    
    Args:
        model: Existing model to update
        new_data_x: New features data
        new_data_y: New labels
        epochs: Number of epochs for updating
        
    Returns:
        Updated model and training history
    """
    history = model.fit(
        new_data_x,
        new_data_y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=get_callbacks()
    )
    return model, history
