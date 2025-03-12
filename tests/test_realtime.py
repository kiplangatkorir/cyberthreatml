"""
Unit tests for the realtime module.
"""

import unittest
import numpy as np
import time
import threading 
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.realtime import RealTimeDetector, PacketStreamDetector

class TestRealTimeDetector(unittest.TestCase):
    """
    Test cases for the RealTimeDetector class.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create a small input shape for testing
        self.input_shape = (10,)
        
        # Create model instance
        self.model = ThreatDetectionModel(self.input_shape)
        
        # Generate dummy data and train model
        np.random.seed(42)  # For reproducibility
        X_train = np.random.random((50, 10))
        y_train = np.random.randint(0, 2, 50)
        self.model.train(X_train, y_train, epochs=1)
        
        # Create detector instance
        self.detector = RealTimeDetector(
            model=self.model,
            threshold=0.5,
            batch_size=5,
            processing_interval=0.1
        )
    
    def test_detector_initialization(self):
        """
        Test detector initialization.
        """
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.threshold, 0.5)
        self.assertEqual(self.detector.batch_size, 5)
        self.assertEqual(self.detector.processing_interval, 0.1)
        self.assertFalse(self.detector.is_running)
    
    def test_detector_start_stop(self):
        """
        Test starting and stopping the detector.
        """
        # Start the detector
        self.detector.start()
        self.assertTrue(self.detector.is_running)
        self.assertIsNotNone(self.detector._processor_thread)
        
        # Stop the detector
        self.detector.stop()
        self.assertFalse(self.detector.is_running)
    
    def test_data_processing(self):
        """
        Test processing data.
        """
        # Start the detector
        self.detector.start()
        
        # Add sample data
        sample_data = np.random.random(10).reshape(1, -1)
        self.detector.add_data(sample_data[0])
        
        # Wait for processing
        time.sleep(0.2)
        
        # Get result
        result = self.detector.get_result(timeout=0.1)
        
        # Check result
        self.assertIsNotNone(result)
        self.assertIn('threat_score', result)
        self.assertIn('is_threat', result)
        
        # Stop the detector
        self.detector.stop()
    
    def test_callback_registration(self):
        """
        Test callback registration.
        """
        # Define test callbacks
        def test_threat_callback(result):
            pass
        
        def test_processing_callback(results):
            pass
        
        # Register callbacks
        self.detector.register_threat_callback(test_threat_callback)
        self.detector.register_processing_callback(test_processing_callback)
        
        # Check registration
        self.assertEqual(self.detector.on_threat_detected, test_threat_callback)
        self.assertEqual(self.detector.on_data_processed, test_processing_callback)


class TestPacketStreamDetector(unittest.TestCase):
    """
    Test cases for the PacketStreamDetector class.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create a small input shape for testing
        self.input_shape = (10,)
        
        # Create model instance
        self.model = ThreatDetectionModel(self.input_shape)
        
        # Generate dummy data and train model
        np.random.seed(42)  # For reproducibility
        X_train = np.random.random((50, 10))
        y_train = np.random.randint(0, 2, 50)
        self.model.train(X_train, y_train, epochs=1)
        
        # Create detector instance
        self.detector = PacketStreamDetector(
            model=self.model,
            feature_extractor=None,  # No feature extractor for testing
            threshold=0.5,
            batch_size=5,
            processing_interval=0.1
        )
    
    def test_packet_processing(self):
        """
        Test processing packets.
        """
        # Start the detector
        self.detector.start()
        
        # Process sample packets
        for _ in range(10):
            # Create a dummy packet (using feature vector directly for testing)
            packet = np.random.random(10)
            self.detector.process_packet(packet)
        
        # Wait for processing
        time.sleep(0.2)
        
        # Check statistics
        stats = self.detector.get_stats()
        self.assertEqual(stats['packet_count'], 10)
        
        # Stop the detector
        self.detector.stop()
    
    def test_threat_detection(self):
        """
        Test threat detection with callbacks.
        """
        # Track detected threats
        detected_threats = []
        
        # Define threat callback
        def on_threat(result):
            detected_threats.append(result)
        
        # Register callback
        self.detector.register_threat_callback(on_threat)
        
        # Start the detector
        self.detector.start()
        
        # Process packets that will definitely be classified as threats
        high_risk_packets = [np.ones(10) for _ in range(5)]  # These should trigger high scores
        for packet in high_risk_packets:
            self.detector.process_packet(packet)
        
        # Wait for processing
        time.sleep(0.3)
        
        # Check if threats were detected (at least one should be)
        self.assertGreaterEqual(len(detected_threats), 1)
        
        # Stop the detector
        self.detector.stop()


if __name__ == '__main__':
    unittest.main()
