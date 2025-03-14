"""
Threat interpretation module for CyberThreat-ML.

This module provides tools for interpreting model predictions using SHAP and rules-based approaches.
"""

import numpy as np
import shap
import logging

logger = logging.getLogger(__name__)

class ThreatInterpreter:
    """Interprets predictions from the threat detection model."""
    
    def __init__(self, model, feature_names, explainers=None):
        """
        Initialize the threat interpreter.
        
        Args:
            model: The trained model to interpret
            feature_names (list): Names of input features
            explainers (list): List of explainer types to use ['shap', 'rules']
        """
        self.model = model
        self.feature_names = feature_names
        self.explainers = explainers or ['shap', 'rules']
        
        # Initialize SHAP explainer
        if 'shap' in self.explainers:
            try:
                # Create background dataset for SHAP
                background_data = np.random.normal(0.3, 0.1, (100, len(feature_names)))
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba,
                    background_data
                )
                logger.info("SHAP explainer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
                self.shap_explainer = None
    
    def interpret_prediction(self, features):
        """
        Generate interpretation for a model prediction.
        
        Args:
            features (numpy.ndarray): Input features to interpret
            
        Returns:
            dict: Dictionary containing interpretation results
        """
        interpretations = {}
        
        # Get model prediction
        pred_proba = self.model.predict_proba(features.reshape(1, -1))[0]
        interpretations['prediction_probability'] = float(pred_proba[0])
        
        # Generate SHAP interpretation
        if 'shap' in self.explainers and self.shap_explainer is not None:
            try:
                # Calculate SHAP values
                shap_values = self.shap_explainer.shap_values(features.reshape(1, -1))
                
                # Convert SHAP values to feature importance dict
                feature_importance = {}
                for idx, name in enumerate(self.feature_names):
                    feature_importance[name] = float(abs(shap_values[0][idx]))
                
                interpretations['shap'] = {
                    'feature_importance': feature_importance,
                    'base_value': float(self.shap_explainer.expected_value),
                    'shap_values': shap_values[0].tolist()
                }
            except Exception as e:
                logger.error(f"SHAP interpretation failed: {str(e)}")
                interpretations['shap'] = {'error': str(e)}
        
        # Generate rules-based interpretation
        if 'rules' in self.explainers:
            try:
                rules = self._apply_detection_rules(features)
                interpretations['rules'] = rules
            except Exception as e:
                logger.error(f"Rules-based interpretation failed: {str(e)}")
                interpretations['rules'] = {'error': str(e)}
        
        return interpretations
    
    def _apply_detection_rules(self, features):
        """
        Apply detection rules to features.
        
        Args:
            features (numpy.ndarray): Input features
            
        Returns:
            dict: Dictionary of triggered rules
        """
        rules = []
        feature_dict = dict(zip(self.feature_names, features))
        
        # Port scan detection
        if (feature_dict['unique_dests'] > 0.8 and 
            feature_dict['syn_rate'] > 0.7):
            rules.append({
                'name': 'Port Scan',
                'confidence': min(feature_dict['unique_dests'], feature_dict['syn_rate']),
                'features': ['unique_dests', 'syn_rate']
            })
        
        # DDoS detection
        if (feature_dict['packet_count'] > 0.9 and 
            feature_dict['src_ip_entropy'] > 0.8):
            rules.append({
                'name': 'DDoS Attack',
                'confidence': min(feature_dict['packet_count'], feature_dict['src_ip_entropy']),
                'features': ['packet_count', 'src_ip_entropy']
            })
        
        # Data exfiltration detection
        if (feature_dict['bytes_transferred'] > 0.8 and 
            feature_dict['encrypted'] > 0.7):
            rules.append({
                'name': 'Data Exfiltration',
                'confidence': min(feature_dict['bytes_transferred'], feature_dict['encrypted']),
                'features': ['bytes_transferred', 'encrypted']
            })
        
        # C2 communication detection
        if (feature_dict['inter_arrival_time'] > 0.7 and 
            feature_dict['payload_entropy'] > 0.8 and
            feature_dict['encrypted'] > 0.7):
            rules.append({
                'name': 'Command & Control',
                'confidence': min(feature_dict['inter_arrival_time'],
                                feature_dict['payload_entropy'],
                                feature_dict['encrypted']),
                'features': ['inter_arrival_time', 'payload_entropy', 'encrypted']
            })
        
        return {
            'triggered_rules': rules,
            'total_rules': len(rules)
        }
