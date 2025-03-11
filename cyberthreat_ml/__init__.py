"""
CyberThreat-ML: A Python library for real-time cybersecurity threat detection using TensorFlow.

This library provides tools for:
- Processing network traffic data
- Building, training, and evaluating ML models for threat detection
- Real-time prediction of cyber threats
- Zero-day attack detection using anomaly-based approaches
- Model explainability using SHAP
- Text-based visualization and reporting capabilities
"""

__version__ = '0.1.0'

# Import modules with try/except to handle missing dependencies gracefully
try:
    from cyberthreat_ml.model import ThreatDetectionModel, load_model
    from cyberthreat_ml.preprocessing import FeatureExtractor
    from cyberthreat_ml.evaluation import evaluate_model, classification_report
    from cyberthreat_ml.explain import explain_prediction, explain_model
    from cyberthreat_ml.realtime import RealTimeDetector, PacketStreamDetector
    from cyberthreat_ml.anomaly import ZeroDayDetector, RealTimeZeroDayDetector
    from cyberthreat_ml.text_visualization import TextVisualizer, SecurityReportGenerator
except ImportError as e:
    import warnings
    warnings.warn(f"Some modules could not be imported: {str(e)}")

__all__ = [
    'ThreatDetectionModel',
    'load_model',
    'FeatureExtractor',
    'evaluate_model',
    'classification_report',
    'explain_prediction',
    'explain_model',
    'RealTimeDetector',
    'PacketStreamDetector',
    'ZeroDayDetector',
    'RealTimeZeroDayDetector',
    'TextVisualizer',
    'SecurityReportGenerator'
]
