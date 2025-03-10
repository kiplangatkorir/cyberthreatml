"""
Unit tests for the preprocessing module.
"""

import unittest
import numpy as np
import pandas as pd
from cyberthreat_ml.preprocessing import FeatureExtractor, extract_packet_features, extract_flow_features

class TestFeatureExtractor(unittest.TestCase):
    """
    Test cases for the FeatureExtractor class.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create sample data for testing
        self.data = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [5.5, 4.4, 3.3, 2.2, 1.1],
            'categorical1': ['A', 'B', 'A', 'C', 'B'],
            'categorical2': ['X', 'Y', 'Z', 'X', 'Y'],
            'ip_addr': ['192.168.1.1', '10.0.0.1', '172.16.0.1', '127.0.0.1', '8.8.8.8']
        })
    
    def test_feature_extractor_initialization(self):
        """
        Test feature extractor initialization.
        """
        # Test with different feature types
        extractor = FeatureExtractor(
            categorical_features=['categorical1', 'categorical2'],
            numeric_features=['numeric1', 'numeric2'],
            ip_features=['ip_addr']
        )
        
        self.assertEqual(extractor.categorical_features, ['categorical1', 'categorical2'])
        self.assertEqual(extractor.numeric_features, ['numeric1', 'numeric2'])
        self.assertEqual(extractor.ip_features, ['ip_addr'])
    
    def test_feature_extraction(self):
        """
        Test feature extraction.
        """
        # Create extractor with all feature types
        extractor = FeatureExtractor(
            categorical_features=['categorical1', 'categorical2'],
            numeric_features=['numeric1', 'numeric2'],
            ip_features=['ip_addr'],
            scaling='standard'
        )
        
        # Fit and transform
        transformed = extractor.fit_transform(self.data)
        
        # Check if output is a numpy array
        self.assertIsInstance(transformed, np.ndarray)
        
        # Numeric features (2) + one-hot encoded categorical features + IP features (2 per IP)
        # 2 + (3 categories for categorical1 + 3 categories for categorical2) + 2 = 10
        # The actual number might vary depending on the implementation details
        self.assertTrue(transformed.shape[1] >= 2 + 3 + 3 + 2)
    
    def test_transform_new_data(self):
        """
        Test transforming new data after fitting.
        """
        # Create extractor
        extractor = FeatureExtractor(
            numeric_features=['numeric1', 'numeric2'],
            scaling='standard'
        )
        
        # Fit on original data
        extractor.fit(self.data)
        
        # Create new data
        new_data = pd.DataFrame({
            'numeric1': [6.0, 7.0],
            'numeric2': [0.5, -0.5]
        })
        
        # Transform new data
        transformed = extractor.transform(new_data)
        
        # Check output
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.shape, (2, 2))  # 2 samples, 2 features


class TestFeatureExtractionFunctions(unittest.TestCase):
    """
    Test cases for the feature extraction functions.
    """
    
    def test_extract_packet_features(self):
        """
        Test extracting features from packet data.
        """
        # Create sample packet data
        packet_data = {
            'header': {
                'protocol': 6,  # TCP
                'length': 1500,
                'ttl': 64,
                'flags': 0x02,  # SYN
                'src_ip': '192.168.1.1',
                'dst_ip': '10.0.0.1',
                'src_port': 12345,
                'dst_port': 80
            },
            'payload': b'GET / HTTP/1.1\r\nHost: example.com\r\n\r\n'
        }
        
        # Extract features
        features = extract_packet_features(packet_data)
        
        # Check if key features are present
        self.assertIn('header_protocol', features)
        self.assertIn('header_length', features)
        self.assertIn('src_ip', features)
        self.assertIn('dst_ip', features)
        self.assertIn('src_port', features)
        self.assertIn('dst_port', features)
        
        if 'payload_length' in features:
            self.assertEqual(features['payload_length'], len(packet_data['payload']))
    
    def test_extract_flow_features(self):
        """
        Test extracting features from flow data.
        """
        # Create sample flow data
        flow_data = {
            'duration': 10.5,
            'protocol': 6,  # TCP
            'total_packets': 32,
            'total_bytes': 4800,
            'src_ip': '192.168.1.1',
            'dst_ip': '10.0.0.1',
            'src_port': 12345,
            'dst_port': 80,
            'src_packets': 15,
            'dst_packets': 17,
            'src_bytes': 2300,
            'dst_bytes': 2500,
            'syn_count': 1,
            'ack_count': 31,
            'fin_count': 1
        }
        
        # Extract features
        features = extract_flow_features(flow_data)
        
        # Check if key features are present
        self.assertIn('duration', features)
        self.assertIn('protocol', features)
        self.assertIn('total_packets', features)
        self.assertIn('src_ip', features)
        self.assertIn('dst_ip', features)
        
        # Check derived features
        if 'packet_rate' in features:
            self.assertAlmostEqual(features['packet_rate'], 32 / 10.5)


if __name__ == '__main__':
    unittest.main()
