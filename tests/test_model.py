"""
Unit tests for the model module.
"""

import unittest 
import numpy as np
import tensorflow as tf
import os
import tempfile
from cyberthreat_ml.model import ThreatDetectionModel, load_model

class TestThreatDetectionModel(unittest.TestCase):
    """
    Test cases for the ThreatDetectionModel class.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create a small input shape for testing
        self.input_shape = (10,)
        
        # Create model instance
        self.model = ThreatDetectionModel(self.input_shape)
        
        # Generate dummy data
        self.X_train = np.random.random((100, 10))
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.random((20, 10))
    
    def test_model_initialization(self):
        """
        Test model initialization.
        """
        # Test default initialization
        model = ThreatDetectionModel(self.input_shape)
        self.assertIsNotNone(model.model)
        
        # Test with custom configuration
        custom_config = {
            'hidden_layers': [32, 16],
            'dropout_rate': 0.3,
            'activation': 'tanh',
            'output_activation': 'sigmoid',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'optimizer': 'sgd'
        }
        model = ThreatDetectionModel(self.input_shape, model_config=custom_config)
        self.assertIsNotNone(model.model)
    
    def test_model_training(self):
        """
        Test model training.
        """
        # Train the model for a few epochs
        history = self.model.train(
            self.X_train, self.y_train,
            epochs=2,
            batch_size=16
        )
        
        # Check if history object is returned
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)
    
    def test_model_prediction(self):
        """
        Test model prediction.
        """
        # Train the model for a single epoch
        self.model.train(self.X_train, self.y_train, epochs=1)
        
        # Make probability predictions
        proba = self.model.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (self.X_test.shape[0],))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        
        # Make binary predictions
        binary = self.model.predict(self.X_test)
        self.assertEqual(binary.shape, (self.X_test.shape[0],))
        self.assertTrue(np.all((binary == 0) | (binary == 1)))
    
    def test_model_save_load(self):
        """
        Test saving and loading the model.
        """
        # Train the model for a single epoch
        self.model.train(self.X_train, self.y_train, epochs=1)
        
        # Get predictions from original model
        original_predictions = self.model.predict_proba(self.X_test)
        
        # Create temporary directory for saving
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model')
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            
            # Save the model
            self.model.save_model(model_path, metadata_path)
            
            # Check if files exist (model has .keras extension)
            self.assertTrue(os.path.exists(model_path + '.keras'))
            self.assertTrue(os.path.exists(metadata_path))
            
            # Load the model
            loaded_model = load_model(model_path, metadata_path)
            
            # Check if loaded model works
            loaded_predictions = loaded_model.predict_proba(self.X_test)
            
            # Verify predictions are the same
            np.testing.assert_allclose(original_predictions, loaded_predictions, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
