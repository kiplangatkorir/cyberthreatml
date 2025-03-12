import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_shape):
    """
    Create a neural network model for cyber threat detection.
    
    Args:
        input_shape (tuple): Shape of the input data
        
    Returns:
        model: Compiled TensorFlow model
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model
