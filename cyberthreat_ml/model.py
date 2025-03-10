"""
Module for defining, training, and saving threat detection models.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as tf_load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .logger import logger

class ThreatDetectionModel:
    """
    A class for building, training, and using threat detection models.
    Supports both binary and multi-class classification for different types of cyber threats.
    """
    
    def __init__(self, input_shape, num_classes=2, model_config=None):
        """
        Initialize the threat detection model.
        
        Args:
            input_shape (tuple): Shape of the input data (without batch dimension).
            num_classes (int): Number of threat classes (2 for binary, >2 for multi-class).
            model_config (dict, optional): Configuration for the model architecture.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Determine if this is a binary or multi-class problem
        self.is_binary = (num_classes == 2)
        
        # Set default configuration based on classification type
        default_config = {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'sigmoid' if self.is_binary else 'softmax',
            'loss': 'binary_crossentropy' if self.is_binary else 'categorical_crossentropy',
            'metrics': ['accuracy', 'AUC', 'Precision', 'Recall'] if self.is_binary else ['accuracy'],
            'optimizer': 'adam',
            'class_names': None  # Optional list of class names for multi-class
        }
        
        # Override defaults with provided config
        self.model_config = default_config.copy()
        if model_config:
            self.model_config.update(model_config)
        
        self.model = self._build_model()
        self.history = None
        logger.info(f"Initialized ThreatDetectionModel with input shape: {input_shape}, " 
                   f"{'binary' if self.is_binary else 'multi-class'} classification "
                   f"({num_classes} classes)")
    
    def _build_model(self):
        """
        Build the TensorFlow model according to the configuration.
        
        Returns:
            tensorflow.keras.models.Sequential: Built model.
        """
        model = Sequential()
        
        # Add first layer with input shape
        model.add(Dense(
            self.model_config['hidden_layers'][0],
            activation=self.model_config['activation'],
            input_shape=self.input_shape
        ))
        model.add(Dropout(self.model_config['dropout_rate']))
        
        # Add additional hidden layers
        for units in self.model_config['hidden_layers'][1:]:
            model.add(Dense(units, activation=self.model_config['activation']))
            model.add(Dropout(self.model_config['dropout_rate']))
        
        # Add output layer - 1 neuron for binary, num_classes for multi-class
        output_units = 1 if self.is_binary else self.num_classes
        model.add(Dense(output_units, activation=self.model_config['output_activation']))
        
        # Compile the model
        model.compile(
            optimizer=self.model_config['optimizer'],
            loss=self.model_config['loss'],
            metrics=self.model_config['metrics']
        )
        
        logger.info(f"Built model with architecture: {self.model_config['hidden_layers']}, "
                   f"output units: {output_units}")
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, 
              early_stopping=True, early_stopping_patience=3, checkpoint_path=None):
        """
        Train the model on the provided data.
        
        Args:
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
            X_val (numpy.ndarray, optional): Validation features.
            y_val (numpy.ndarray, optional): Validation labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            early_stopping (bool): Whether to use early stopping.
            early_stopping_patience (int): Patience for early stopping.
            checkpoint_path (str, optional): Path to save model checkpoints.
            
        Returns:
            tensorflow.keras.callbacks.History: Training history.
        """
        callbacks = []
        
        if early_stopping and X_val is not None and y_val is not None:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ))
        
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            ))
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        logger.info(f"Training model with {X_train.shape[0]} samples, {epochs} epochs, batch size {batch_size}")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.history
    
    def predict(self, X, threshold=0.5):
        """
        Make class predictions with the model.
        
        Args:
            X (numpy.ndarray): Input features.
            threshold (float): Threshold for binary classification (ignored for multi-class).
            
        Returns:
            numpy.ndarray: Class predictions (0/1 for binary, class indices for multi-class).
        """
        if self.is_binary:
            # Binary classification - apply threshold
            y_pred_proba = self.predict_proba(X)
            if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 1:
                # Single column for binary
                return (y_pred_proba >= threshold).astype(int)
            else:
                # In case we get a 2D array (should not happen in binary case)
                return np.argmax(y_pred_proba, axis=1)
        else:
            # Multi-class classification - return class with highest probability
            y_pred_proba = self.predict_proba(X)
            return np.argmax(y_pred_proba, axis=1)
    
    def predict_proba(self, X):
        """
        Make probability predictions with the model.
        
        Args:
            X (numpy.ndarray): Input features.
            
        Returns:
            numpy.ndarray: Probability predictions (1D array for binary, 2D array for multi-class).
        """
        predictions = self.model.predict(X)
        
        if self.is_binary:
            # Return probabilities as a 1D array for binary classification
            return predictions.flatten()
        else:
            # Return probabilities as a 2D array (samples x classes) for multi-class
            return predictions
            
    def predict_with_explanations(self, X):
        """
        Make predictions and provide explanations for the decisions.
        
        Args:
            X (numpy.ndarray): Input features.
            
        Returns:
            dict: Dictionary with predictions and explanation data.
        """
        # Get predictions
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Create result dictionary
        result = {
            'predictions': predictions,
            'probabilities': probabilities,
            'explanation': {
                'model_type': 'binary' if self.is_binary else 'multi-class',
                'class_names': self.model_config.get('class_names'),
                'confidence': np.max(probabilities, axis=1) if not self.is_binary else probabilities
            }
        }
        
        return result
    
    def save_model(self, model_path, metadata_path=None):
        """
        Save the model and its metadata.
        
        Args:
            model_path (str): Path to save the TensorFlow model.
            metadata_path (str, optional): Path to save model metadata.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        self.model.save(model_path)
        
        # Save metadata if path is provided
        if metadata_path:
            metadata = {
                'input_shape': self.input_shape,
                'model_config': self.model_config
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        logger.info(f"Model saved to {model_path}")
        if metadata_path:
            logger.info(f"Model metadata saved to {metadata_path}")


def load_model(model_path, metadata_path=None):
    """
    Load a saved threat detection model.
    
    Args:
        model_path (str): Path to the saved TensorFlow model.
        metadata_path (str, optional): Path to the saved model metadata.
        
    Returns:
        ThreatDetectionModel: Loaded model instance.
    """
    # Load the TensorFlow model first to get its architecture
    tf_model = tf_load_model(model_path)
    
    if metadata_path:
        # Load metadata if provided
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Determine number of classes from output layer
        last_layer = tf_model.layers[-1]
        output_shape = last_layer.output_shape
        num_classes = output_shape[-1] if len(output_shape) > 1 and output_shape[-1] > 1 else 2
        
        # Create model instance with the right number of classes
        model_instance = ThreatDetectionModel(
            input_shape=tuple(metadata['input_shape']),
            num_classes=num_classes,
            model_config=metadata['model_config']
        )
    else:
        # Try to infer configuration from the model
        input_shape = tf_model.layers[0].input_shape[0][1:]
        
        # Determine number of classes from output layer
        last_layer = tf_model.layers[-1]
        output_shape = last_layer.output_shape
        num_classes = output_shape[-1] if len(output_shape) > 1 and output_shape[-1] > 1 else 2
        
        # Create model instance
        model_instance = ThreatDetectionModel(
            input_shape=input_shape,
            num_classes=num_classes
        )
    
    # Load the model weights
    model_instance.model = tf_model
    
    # Update the is_binary flag to match the loaded model
    model_instance.is_binary = (model_instance.num_classes == 2)
    
    logger.info(f"Loaded model from {model_path} with {model_instance.num_classes} classes "
               f"({'binary' if model_instance.is_binary else 'multi-class'} classification)")
    return model_instance
