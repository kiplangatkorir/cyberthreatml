"""
Module for data preprocessing and feature extraction for cybersecurity data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import ipaddress
from .logger import logger

class FeatureExtractor:
    """
    A class for extracting and preprocessing features from network traffic data.
    """
    
    def __init__(self, categorical_features=None, numeric_features=None, 
                 ip_features=None, scaling='standard', handle_missing=True):
        """
        Initialize the feature extractor.
        
        Args:
            categorical_features (list, optional): List of categorical feature names.
            numeric_features (list, optional): List of numeric feature names.
            ip_features (list, optional): List of IP address feature names.
            scaling (str): Scaling method ('standard', 'minmax', or None).
            handle_missing (bool): Whether to handle missing values.
        """
        self.categorical_features = categorical_features or []
        self.numeric_features = numeric_features or []
        self.ip_features = ip_features or []
        self.scaling = scaling
        self.handle_missing = handle_missing
        self.preprocessor = None
        self.fitted = False
        logger.info(f"Initialized FeatureExtractor with {len(self.categorical_features)} categorical, "
                   f"{len(self.numeric_features)} numeric, and {len(self.ip_features)} IP features")
    
    def _create_preprocessor(self):
        """
        Create a preprocessing pipeline for the features.
        
        Returns:
            sklearn.compose.ColumnTransformer: Preprocessing pipeline.
        """
        transformers = []
        
        # Numeric features preprocessing
        if self.numeric_features:
            numeric_pipeline = []
            
            if self.handle_missing:
                numeric_pipeline.append(('imputer', SimpleImputer(strategy='mean')))
            
            if self.scaling == 'standard':
                numeric_pipeline.append(('scaler', StandardScaler()))
            elif self.scaling == 'minmax':
                numeric_pipeline.append(('scaler', MinMaxScaler()))
            
            if numeric_pipeline:
                transformers.append(
                    ('numeric', Pipeline(numeric_pipeline), self.numeric_features)
                )
        
        # Categorical features preprocessing
        if self.categorical_features:
            categorical_pipeline = []
            
            if self.handle_missing:
                categorical_pipeline.append(('imputer', SimpleImputer(strategy='most_frequent')))
            
            categorical_pipeline.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
            
            transformers.append(
                ('categorical', Pipeline(categorical_pipeline), self.categorical_features)
            )
        
        # IP address features preprocessing
        if self.ip_features:
            transformers.append(
                ('ip', IPAddressTransformer(), self.ip_features)
            )
        
        return ColumnTransformer(transformers, remainder='drop')
    
    def fit(self, X):
        """
        Fit the feature extractor to the data.
        
        Args:
            X (pandas.DataFrame): Input data.
            
        Returns:
            FeatureExtractor: Self instance.
        """
        # Create preprocessor if not already created
        if self.preprocessor is None:
            self.preprocessor = self._create_preprocessor()
        
        # Fit the preprocessor
        self.preprocessor.fit(X)
        self.fitted = True
        
        logger.info("Feature extractor fitted to data")
        return self
    
    def transform(self, X):
        """
        Transform the data using the fitted preprocessor.
        
        Args:
            X (pandas.DataFrame): Input data.
            
        Returns:
            numpy.ndarray: Transformed features.
        """
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        logger.info(f"Transforming data with shape {X.shape}")
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X):
        """
        Fit the feature extractor and transform the data.
        
        Args:
            X (pandas.DataFrame): Input data.
            
        Returns:
            numpy.ndarray: Transformed features.
        """
        return self.fit(X).transform(X)
    
    def get_feature_names(self):
        """
        Get the names of the output features after transformation.
        
        Returns:
            list: Feature names.
        """
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before getting feature names")
        
        try:
            return self.preprocessor.get_feature_names_out()
        except AttributeError:
            # For older scikit-learn versions
            return ['feature_' + str(i) for i in range(self.transform(pd.DataFrame(
                {feature: [0] for feature in self.categorical_features + self.numeric_features + self.ip_features}
            )).shape[1])]


class IPAddressTransformer:
    """
    Transformer for IP address features. Converts IP addresses to numeric features.
    """
    
    def fit(self, X, y=None):
        """
        Fit method (no-op for this transformer).
        
        Returns:
            IPAddressTransformer: Self instance.
        """
        return self
    
    def transform(self, X):
        """
        Transform IP addresses to numeric features.
        
        Args:
            X (pandas.DataFrame): Input data with IP address columns.
            
        Returns:
            numpy.ndarray: Transformed features.
        """
        X_transformed = np.zeros((X.shape[0], X.shape[1] * 2))
        
        for i, col in enumerate(X.columns):
            # Process each IP address
            for j, ip_str in enumerate(X[col]):
                try:
                    # Handle missing values
                    if pd.isna(ip_str):
                        X_transformed[j, i*2] = 0
                        X_transformed[j, i*2+1] = 0
                        continue
                    
                    # Convert IP to numeric representation
                    ip = ipaddress.ip_address(str(ip_str))
                    
                    # Store version and integer representation
                    X_transformed[j, i*2] = ip.version  # 4 or 6
                    X_transformed[j, i*2+1] = int(ip)
                except (ValueError, TypeError):
                    # Invalid IP address
                    X_transformed[j, i*2] = 0
                    X_transformed[j, i*2+1] = 0
        
        return X_transformed


def extract_packet_features(packet_data, include_headers=True, include_payload=True, 
                           max_payload_length=1024):
    """
    Extract features from network packet data.
    
    Args:
        packet_data (dict): Network packet data.
        include_headers (bool): Whether to include header features.
        include_payload (bool): Whether to include payload features.
        max_payload_length (int): Maximum payload length to include.
        
    Returns:
        dict: Extracted features.
    """
    features = {}
    
    # Extract header features
    if include_headers and 'header' in packet_data:
        header = packet_data['header']
        
        # Extract common header fields
        for field in ['protocol', 'length', 'ttl', 'flags']:
            if field in header:
                features[f'header_{field}'] = header[field]
        
        # Extract IP addresses
        if 'src_ip' in header:
            features['src_ip'] = header['src_ip']
        if 'dst_ip' in header:
            features['dst_ip'] = header['dst_ip']
        
        # Extract port numbers
        if 'src_port' in header:
            features['src_port'] = header['src_port']
        if 'dst_port' in header:
            features['dst_port'] = header['dst_port']
    
    # Extract payload features
    if include_payload and 'payload' in packet_data:
        payload = packet_data['payload']
        
        # Calculate payload length
        features['payload_length'] = len(payload)
        
        # Calculate entropy of payload
        if payload:
            try:
                from collections import Counter
                import math
                
                # Calculate entropy
                counter = Counter(payload[:max_payload_length])
                entropy = 0
                total = min(len(payload), max_payload_length)
                
                for byte, count in counter.items():
                    probability = count / total
                    entropy -= probability * math.log2(probability)
                
                features['payload_entropy'] = entropy
            except Exception as e:
                logger.warning(f"Error calculating payload entropy: {str(e)}")
                features['payload_entropy'] = 0
        else:
            features['payload_entropy'] = 0
    
    return features


def extract_flow_features(flow_data):
    """
    Extract features from network flow data.
    
    Args:
        flow_data (dict): Network flow data.
        
    Returns:
        dict: Extracted features.
    """
    features = {}
    
    # Basic flow features
    for key in ['duration', 'protocol', 'total_packets', 'total_bytes']:
        if key in flow_data:
            features[key] = flow_data[key]
    
    # Source and destination information
    for prefix in ['src', 'dst']:
        for suffix in ['ip', 'port', 'packets', 'bytes']:
            key = f'{prefix}_{suffix}'
            if key in flow_data:
                features[key] = flow_data[key]
    
    # Calculate packet rate and byte rate if duration is available
    if 'duration' in flow_data and flow_data['duration'] > 0:
        if 'total_packets' in flow_data:
            features['packet_rate'] = flow_data['total_packets'] / flow_data['duration']
        if 'total_bytes' in flow_data:
            features['byte_rate'] = flow_data['total_bytes'] / flow_data['duration']
    
    # Flag counts
    for flag in ['syn', 'ack', 'fin', 'rst', 'psh', 'urg']:
        key = f'{flag}_count'
        if key in flow_data:
            features[key] = flow_data[key]
    
    return features
