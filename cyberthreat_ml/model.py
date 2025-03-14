import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model as tf_load_model
from tensorflow.keras.layers import Dense, Dropout
import json
import os

class ThreatDetectionModel:
    """
    A neural network model for cyber threat detection.
    """
    
    def __init__(self, input_shape, model_config=None):
        """
        Initialize the model with given input shape and optional configuration.
        
        Args:
            input_shape (tuple): Shape of the input data
            model_config (dict): Optional model configuration
        """
        self.input_shape = input_shape
        self.model_config = model_config or {
            'hidden_layers': [64, 32],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'output_activation': 'sigmoid',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'optimizer': 'adam'
        }
        self.model = self._build_model()
    
    def _build_model(self):
        """Build and compile the model."""
        model = Sequential()
        
        # First layer needs input shape
        model.add(Dense(
            self.model_config['hidden_layers'][0],
            activation=self.model_config['activation'],
            input_shape=self.input_shape
        ))
        model.add(Dropout(self.model_config['dropout_rate']))
        
        # Add remaining hidden layers
        for units in self.model_config['hidden_layers'][1:]:
            model.add(Dense(units, activation=self.model_config['activation']))
            model.add(Dropout(self.model_config['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation=self.model_config['output_activation']))
        
        model.compile(
            optimizer=self.model_config['optimizer'],
            loss=self.model_config['loss'],
            metrics=self.model_config['metrics']
        )
        return model
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional arguments passed to model.fit()
            
        Returns:
            History object from training
        """
        return self.model.fit(X_train, y_train, **kwargs)
    
    def predict_proba(self, X):
        """
        Make probability predictions.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        return self.model.predict(X).flatten()
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions.
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def save_model(self, model_path, metadata_path):
        """
        Save the model and its metadata.
        
        Args:
            model_path: Path to save the model
            metadata_path: Path to save the metadata
        """
        # Save the model with .keras extension
        model_path_with_ext = model_path + '.keras'
        save_model(self.model, model_path_with_ext)
        
        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'model_config': self.model_config
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

def load_model(model_path, metadata_path):
    """
    Load a saved model and its metadata.
    
    Args:
        model_path: Path to the saved model
        metadata_path: Path to the saved metadata
        
    Returns:
        Loaded ThreatDetectionModel instance
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model instance
    model = ThreatDetectionModel(
        input_shape=tuple(metadata['input_shape']),
        model_config=metadata['model_config']
    )
    
    # Load weights with .keras extension
    model_path_with_ext = model_path + '.keras'
    model.model = tf_load_model(model_path_with_ext)
    return model
