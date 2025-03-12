"""
Unit tests for the evaluation module. 
"""

import unittest
import numpy as np
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.evaluation import evaluate_model, classification_report, find_optimal_threshold

class TestEvaluation(unittest.TestCase):
    """
    Test cases for the evaluation module.
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
        np.random.seed(42)  # For reproducibility
        self.X_train = np.random.random((100, 10))
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.random((30, 10))
        self.y_test = np.random.randint(0, 2, 30)
        
        # Train the model for a single epoch
        self.model.train(self.X_train, self.y_train, epochs=1, batch_size=16)
    
    def test_evaluate_model(self):
        """
        Test model evaluation.
        """
        # Evaluate the model
        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        
        # Check if metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Check metric values
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        
        self.assertIsInstance(metrics['confusion_matrix'], np.ndarray)
        self.assertEqual(metrics['confusion_matrix'].shape, (2, 2))
    
    def test_classification_report(self):
        """
        Test classification report generation.
        """
        # Generate classification report
        report = classification_report(self.model, self.X_test, self.y_test)
        
        # Check if report is returned
        self.assertIsInstance(report, str)
        self.assertIn('precision', report.lower())
        self.assertIn('recall', report.lower())
        self.assertIn('f1-score', report.lower())
    
    def test_find_optimal_threshold(self):
        """
        Test finding optimal threshold.
        """
        # Find optimal threshold
        threshold = find_optimal_threshold(self.model, self.X_test, self.y_test, metric='f1')
        
        # Check if threshold is returned
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        
        # Test with different metrics
        for metric in ['precision', 'recall', 'accuracy']:
            threshold = find_optimal_threshold(self.model, self.X_test, self.y_test, metric=metric)
            self.assertIsInstance(threshold, float)
            self.assertGreaterEqual(threshold, 0.0)
            self.assertLessEqual(threshold, 1.0)
        
        # Test with invalid metric
        with self.assertRaises(ValueError):
            find_optimal_threshold(self.model, self.X_test, self.y_test, metric='invalid')


if __name__ == '__main__':
    unittest.main()
