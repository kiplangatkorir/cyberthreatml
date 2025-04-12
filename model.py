import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LSTM, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, GlobalAveragePooling1D, Concatenate
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

class AnomalyScorer(tf.keras.layers.Layer):
    """Custom layer for computing anomaly scores."""
    def __init__(self, threshold=0.5):
        super(AnomalyScorer, self).__init__()
        self.threshold = threshold
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        # Compute base prediction
        base_pred = self.dense(inputs)
        
        # Compute anomaly score based on feature patterns
        feature_variance = tf.math.reduce_std(inputs, axis=-1, keepdims=True)
        anomaly_score = tf.math.sigmoid(feature_variance) * base_pred
        
        return base_pred, anomaly_score

def create_model(input_shape, complexity='standard', sequence_length=None):
    """
    Create an enhanced neural network model for cyber threat detection.
    
    Args:
        input_shape (tuple): Shape of the input data
        complexity (str): Model complexity level ('standard', 'complex')
        sequence_length (int): Length of temporal sequence (None for non-temporal)
        
    Returns:
        model: Compiled TensorFlow model with advanced threat detection
    """
    if sequence_length:
        # Temporal input shape: (batch, sequence_length, features)
        inputs = Input(shape=(sequence_length, input_shape[-1]))
        
        # Temporal feature extraction with LSTM
        temporal = LSTM(64, return_sequences=True)(inputs)
        
        # Multi-head attention for pattern detection
        attention = MultiHeadAttention(
            num_heads=4, key_dim=16
        )(temporal, temporal)
        
        # Combine attention with temporal features
        x = Concatenate()([temporal, attention])
        x = LayerNormalization()(x)
        
        # Global temporal context
        x = GlobalAveragePooling1D()(x)
    else:
        # Standard feature input
        inputs = Input(shape=input_shape)
        x = inputs
    
    if complexity == 'complex':
        # Enhanced architecture with residual connections
        residual = x
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = x + residual  # Residual connection
        
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
    else:
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
    
    # Anomaly scoring layer
    base_pred, anomaly_score = AnomalyScorer()(x)
    outputs = Concatenate()([base_pred, anomaly_score])
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use Adam optimizer with learning rate scheduling
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss={
            '0': 'binary_crossentropy',  # Base prediction loss
            '1': 'mse'  # Anomaly score loss
        },
        loss_weights={'0': 1.0, '1': 0.5},
        metrics={
            '0': ['accuracy', tf.keras.metrics.AUC(), 
                  tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            '1': ['mse', 'mae']
        }
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

def update_model(model, new_data_x, new_data_y, epochs=5, sequence_length=None):
    """Update the model with new data in real-time.
    
    Args:
        model: Existing model to update
        new_data_x: New features data
        new_data_y: New labels
        epochs: Number of epochs for updating
        
    Returns:
        Updated model and training history
    """
    if sequence_length:
        # Reshape data for temporal analysis if needed
        if len(new_data_x.shape) == 2:
            new_data_x = tf.expand_dims(new_data_x, axis=1)
        
        # Create anomaly scores for training
        anomaly_scores = tf.zeros_like(new_data_y)  # Initialize with zeros
        y = {'0': new_data_y, '1': anomaly_scores}
    else:
        y = {'0': new_data_y, '1': tf.zeros_like(new_data_y)}
    
    history = model.fit(
        new_data_x,
        y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=get_callbacks()
    )
    return model, history
