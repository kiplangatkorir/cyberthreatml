"""
Module for anomaly-based zero-day threat detection. 

This module provides implementations of various anomaly detection algorithms
for identifying potentially malicious activities that don't match known threat patterns.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import logging

from .logger import CyberThreatLogger

logger = CyberThreatLogger("cyberthreat_ml.anomaly", logging.INFO).get_logger()

class ZeroDayDetector:
    """
    Detector for identifying zero-day attacks using anomaly detection techniques.
    
    This detector can run alongside signature-based detection to identify
    previously unknown threats by detecting unusual patterns.
    """
    
    def __init__(self, method='isolation_forest', contamination=0.1, 
                 feature_columns=None, min_samples=100, threshold=None):
        """
        Initialize the zero-day detector.
        
        Args:
            method (str): Anomaly detection algorithm to use. Options:
                - 'isolation_forest': Effective for high-dimensional data
                - 'local_outlier_factor': Good for detecting local density deviations
                - 'robust_covariance': Works well for normally distributed data
                - 'one_class_svm': Effective for separating outliers in feature space
                - 'ensemble': Use multiple methods and combine results
            contamination (float): Expected proportion of outliers in the data (0.0 to 0.5)
            feature_columns (list, optional): Subset of feature columns to use for detection
            min_samples (int): Minimum samples required before detecting anomalies
            threshold (float, optional): Custom threshold for anomaly score (model-specific)
        """
        self.method = method
        self.contamination = contamination
        self.feature_columns = feature_columns
        self.min_samples = min_samples
        self.threshold = threshold
        self.models = {}
        self.fitted = False
        self.feature_names = None
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_min = None
        self.baseline_max = None
        self.sample_history = []
        self.detection_history = []
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize anomaly detection models based on the selected method."""
        if self.method == 'ensemble' or self.method == 'isolation_forest':
            self.models['isolation_forest'] = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            
        if self.method == 'ensemble' or self.method == 'local_outlier_factor':
            self.models['local_outlier_factor'] = LocalOutlierFactor(
                novelty=True,
                contamination=self.contamination,
                n_jobs=-1
            )
            
        if self.method == 'ensemble' or self.method == 'robust_covariance':
            self.models['robust_covariance'] = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
            
        if self.method == 'ensemble' or self.method == 'one_class_svm':
            self.models['one_class_svm'] = OneClassSVM(
                nu=self.contamination,
                kernel="rbf",
                gamma='scale'
            )
    
    def fit(self, X, feature_names=None):
        """
        Fit the anomaly detection models on baseline normal traffic data.
        
        Args:
            X (numpy.ndarray): Feature matrix of normal traffic data.
            feature_names (list, optional): Names of the features in X.
            
        Returns:
            ZeroDayDetector: Self instance.
        """
        if self.feature_columns is not None:
            if feature_names is not None:
                # Select specific feature columns by name
                feature_indices = [feature_names.index(col) for col in self.feature_columns 
                                   if col in feature_names]
                X = X[:, feature_indices]
            else:
                # If no feature names provided, assume feature_columns are indices
                X = X[:, self.feature_columns]
        
        self.feature_names = feature_names
        
        # Calculate baseline statistics
        self.baseline_mean = np.mean(X, axis=0)
        self.baseline_std = np.std(X, axis=0)
        self.baseline_min = np.min(X, axis=0)
        self.baseline_max = np.max(X, axis=0)
        
        # Fit each model
        for name, model in self.models.items():
            try:
                model.fit(X)
                logger.info(f"Fitted {name} model on {X.shape[0]} samples")
            except Exception as e:
                logger.error(f"Error fitting {name} model: {str(e)}")
        
        self.fitted = True
        return self
    
    def detect(self, X, threshold=None, return_scores=False):
        """
        Detect potential zero-day threats in new data.
        
        Args:
            X (numpy.ndarray): Feature matrix to detect anomalies in.
            threshold (float, optional): Custom threshold for anomaly score.
            return_scores (bool): Whether to return raw anomaly scores.
            
        Returns:
            tuple: (predictions, scores) where predictions is a binary array (1=normal, -1=anomaly)
                  and scores are the raw anomaly scores.
        """
        if not self.fitted:
            logger.warning("Models not fitted yet. Call fit() before detect().")
            if return_scores:
                return np.ones(X.shape[0]), np.zeros(X.shape[0])
            return np.ones(X.shape[0])
            
        if self.feature_columns is not None:
            if self.feature_names is not None:
                # Select specific feature columns by name
                feature_indices = [self.feature_names.index(col) for col in self.feature_columns 
                                   if col in self.feature_names]
                X = X[:, feature_indices]
            else:
                # If no feature names provided, assume feature_columns are indices
                X = X[:, self.feature_columns]
        
        # Store samples in history
        self.sample_history.extend(X)
        if len(self.sample_history) > self.min_samples * 10:
            # Keep history size reasonable
            self.sample_history = self.sample_history[-self.min_samples * 10:]
        
        # If we don't have enough samples yet, assume everything is normal
        if len(self.sample_history) < self.min_samples:
            logger.info(f"Not enough samples yet: {len(self.sample_history)}/{self.min_samples}")
            if return_scores:
                return np.ones(X.shape[0]), np.zeros(X.shape[0])
            return np.ones(X.shape[0])
        
        # Get anomaly scores from each model
        scores_dict = {}
        for name, model in self.models.items():
            try:
                if name == 'local_outlier_factor':
                    # LOF has a different API for prediction
                    scores = -model.score_samples(X)
                else:
                    scores = -model.decision_function(X)
                scores_dict[name] = scores
            except Exception as e:
                logger.error(f"Error getting scores from {name} model: {str(e)}")
        
        if not scores_dict:
            # If no models could score, assume normal
            if return_scores:
                return np.ones(X.shape[0]), np.zeros(X.shape[0])
            return np.ones(X.shape[0])
        
        # Combine scores for ensemble method
        if self.method == 'ensemble':
            final_scores = np.zeros(X.shape[0])
            for scores in scores_dict.values():
                # Normalize scores to [0, 1] range for each model
                normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                final_scores += normalized_scores
            
            final_scores /= len(scores_dict)
        else:
            # Use the specified method's scores
            final_scores = scores_dict[self.method]
        
        # Use custom threshold if provided, otherwise use model's threshold
        thresh = threshold if threshold is not None else self.threshold
        
        if thresh is not None:
            # Use custom threshold
            predictions = np.where(final_scores > thresh, -1, 1)
        else:
            # Use model's built-in predict method for default threshold
            try:
                if self.method == 'ensemble':
                    # For ensemble, use median of final_scores as default threshold
                    thresh = np.median(final_scores) * 1.5
                    predictions = np.where(final_scores > thresh, -1, 1)
                else:
                    # Use the specified model's predict method
                    predictions = self.models[self.method].predict(X)
            except Exception as e:
                logger.error(f"Error predicting with {self.method} model: {str(e)}")
                # Fallback: use simple threshold on scores
                thresh = np.percentile(final_scores, 100 * self.contamination)
                predictions = np.where(final_scores > thresh, -1, 1)
        
        # Store detection results
        for i, pred in enumerate(predictions):
            if pred == -1:  # Anomaly detected
                self.detection_history.append((X[i], final_scores[i]))
                
        # Keep detection history size reasonable
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        if return_scores:
            return predictions, final_scores
        return predictions
    
    def analyze_anomaly(self, sample, score=None):
        """
        Analyze why a sample was flagged as anomalous.
        
        Args:
            sample (numpy.ndarray): Feature vector of the anomalous sample.
            score (float, optional): Anomaly score if already computed.
            
        Returns:
            dict: Analysis results including feature contributions and severity.
        """
        if not self.fitted or self.baseline_mean is None:
            return {"error": "Detector not fitted yet"}
        
        if sample.ndim == 2 and sample.shape[0] == 1:
            sample = sample.flatten()
        
        # Compute deviation from baseline
        z_scores = (sample - self.baseline_mean) / (self.baseline_std + 1e-10)
        normalized_values = (sample - self.baseline_min) / (self.baseline_max - self.baseline_min + 1e-10)
        
        # Get feature names (if available) or indices
        if self.feature_names is not None:
            feature_ids = self.feature_names
        else:
            feature_ids = [f"Feature_{i}" for i in range(len(sample))]
        
        # Calculate feature contributions
        contrib = {}
        for i, feature in enumerate(feature_ids):
            contrib[feature] = {
                "value": float(sample[i]),
                "z_score": float(z_scores[i]),
                "normalized": float(normalized_values[i]),
                "baseline_mean": float(self.baseline_mean[i]),
                "baseline_std": float(self.baseline_std[i]),
                "deviation": float(abs(z_scores[i]))
            }
        
        # Sort features by contribution
        sorted_features = sorted(contrib.items(), key=lambda x: x[1]["deviation"], reverse=True)
        
        # Calculate overall severity (normalized between 0 and 1)
        overall_deviation = np.mean(np.abs(z_scores))
        severity = min(1.0, overall_deviation / 5.0)  # Cap at 1.0, 5 std devs is very severe
        
        # Compute anomaly score if not provided
        if score is None:
            if self.method == 'ensemble':
                # Use multiple models for scoring
                score = 0
                for name, model in self.models.items():
                    try:
                        if name == 'local_outlier_factor':
                            score += -model.score_samples(sample.reshape(1, -1))[0]
                        else:
                            score += -model.decision_function(sample.reshape(1, -1))[0]
                    except:
                        pass
                score /= max(1, len(self.models))
            else:
                # Use the specified model
                try:
                    model = self.models[self.method]
                    if self.method == 'local_outlier_factor':
                        score = -model.score_samples(sample.reshape(1, -1))[0]
                    else:
                        score = -model.decision_function(sample.reshape(1, -1))[0]
                except:
                    score = None
        
        # Determine top contributors (features with z-score > 2)
        top_contributors = [f for f, v in sorted_features[:5] if v["deviation"] > 2]
        
        return {
            "anomaly_score": float(score) if score is not None else None,
            "severity": float(severity),
            "severity_level": classify_severity(severity),
            "top_contributors": top_contributors,
            "feature_details": dict(sorted_features),
            "num_standard_deviations": float(overall_deviation)
        }
    
    def get_stats(self):
        """
        Get statistics about the detector.
        
        Returns:
            dict: Statistics about the detector state.
        """
        return {
            "samples_collected": len(self.sample_history),
            "anomalies_detected": len(self.detection_history),
            "models_used": list(self.models.keys()),
            "min_samples_required": self.min_samples,
            "contamination_rate": self.contamination,
            "is_fitted": self.fitted
        }
    
    def reset(self):
        """
        Reset the detector state while keeping the fitted models.
        """
        self.sample_history = []
        self.detection_history = []
        
    def get_recent_anomalies(self, n=10):
        """
        Get the most recent anomalies detected.
        
        Args:
            n (int): Number of recent anomalies to retrieve.
            
        Returns:
            list: List of recent anomalies with their scores.
        """
        return self.detection_history[-n:]

class RealTimeZeroDayDetector:
    """
    Real-time detector for zero-day threats that can run alongside signature-based detection.
    """
    def __init__(self, feature_extractor, baseline_data=None, feature_names=None, 
                method='isolation_forest', contamination=0.05, time_window=3600):
        """
        Initialize the real-time zero-day detector.
        
        Args:
            feature_extractor: Feature extractor compatible with the detector
            baseline_data (numpy.ndarray, optional): Initial baseline data for fitting
            feature_names (list, optional): Names of features
            method (str): Anomaly detection method
            contamination (float): Expected proportion of anomalies
            time_window (int): Time window in seconds for temporal analysis
        """
        self.feature_extractor = feature_extractor
        self.feature_names = feature_names
        self.detector = ZeroDayDetector(
            method=method,
            contamination=contamination
        )
        self.time_window = time_window
        self.recent_samples = []
        self.recent_timestamps = []
        self.recent_anomalies = []
        
        # Fit detector if baseline data is provided
        if baseline_data is not None:
            self.detector.fit(baseline_data, feature_names)
            
    def add_sample(self, sample_data, timestamp=None):
        """
        Add a sample for zero-day detection.
        
        Args:
            sample_data: Raw data to be processed by feature extractor
            timestamp (float, optional): Timestamp for the sample
            
        Returns:
            dict: Detection result if anomaly detected, None otherwise
        """
        # Use current time if timestamp not provided
        if timestamp is None:
            import time
            timestamp = time.time()
        
        # Extract features
        try:
            features = self.feature_extractor.transform(sample_data)
            if features.ndim == 1:
                features = features.reshape(1, -1)
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
        
        # Add to recent samples
        self.recent_samples.append(features)
        self.recent_timestamps.append(timestamp)
        
        # Remove samples outside time window
        current_time = timestamp
        cutoff_time = current_time - self.time_window
        while self.recent_timestamps and self.recent_timestamps[0] < cutoff_time:
            self.recent_timestamps.pop(0)
            self.recent_samples.pop(0)
        
        # Detect anomalies
        if self.detector.fitted:
            predictions, scores = self.detector.detect(features, return_scores=True)
            
            if predictions[0] == -1:  # Anomaly detected
                # Analyze the anomaly
                analysis = self.detector.analyze_anomaly(features[0], scores[0])
                
                result = {
                    "timestamp": timestamp,
                    "is_anomaly": True,
                    "anomaly_score": float(scores[0]),
                    "severity": analysis["severity"],
                    "severity_level": analysis["severity_level"],
                    "analysis": analysis,
                    "features": features[0].tolist() if hasattr(features[0], "tolist") else features[0],
                    "raw_data": sample_data
                }
                
                self.recent_anomalies.append(result)
                # Keep recent anomalies list reasonably sized
                if len(self.recent_anomalies) > 100:
                    self.recent_anomalies = self.recent_anomalies[-100:]
                    
                return result
        
        # If fit needed or no anomaly detected
        return None
    
    def analyze_time_patterns(self):
        """
        Analyze temporal patterns in recent data.
        
        Returns:
            dict: Analysis of temporal patterns.
        """
        if not self.recent_timestamps:
            return {"error": "No recent data available"}
            
        timestamps = np.array(self.recent_timestamps)
        
        # Calculate time differences between consecutive samples
        time_diffs = np.diff(timestamps)
        
        # Calculate statistics
        avg_interval = np.mean(time_diffs) if len(time_diffs) > 0 else 0
        std_interval = np.std(time_diffs) if len(time_diffs) > 0 else 0
        max_interval = np.max(time_diffs) if len(time_diffs) > 0 else 0
        min_interval = np.min(time_diffs) if len(time_diffs) > 0 else 0
        
        # Calculate sample rate (samples per second)
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        sample_rate = len(timestamps) / max(1, duration)
        
        # Calculate anomaly rate
        anomaly_rate = len(self.recent_anomalies) / max(1, len(self.recent_samples))
        
        return {
            "total_samples": len(self.recent_samples),
            "time_window_seconds": self.time_window,
            "avg_interval_seconds": float(avg_interval),
            "std_interval_seconds": float(std_interval),
            "sample_rate_per_second": float(sample_rate),
            "anomalies_detected": len(self.recent_anomalies),
            "anomaly_rate": float(anomaly_rate),
            "window_start_time": float(timestamps[0]) if len(timestamps) > 0 else None,
            "window_end_time": float(timestamps[-1]) if len(timestamps) > 0 else None
        }
        
    def get_stats(self):
        """
        Get combined statistics about the detector.
        
        Returns:
            dict: Combined statistics about detector state and temporal patterns.
        """
        detector_stats = self.detector.get_stats()
        time_stats = self.analyze_time_patterns()
        
        return {
            **detector_stats,
            **time_stats
        }
    
    def train_on_recent_normal(self, min_samples=100):
        """
        Train detector on recent samples that weren't flagged as anomalies.
        
        Args:
            min_samples (int): Minimum required normal samples for training.
            
        Returns:
            bool: Whether training was successful.
        """
        # Get indices of samples that aren't in recent_anomalies
        normal_indices = []
        anomaly_timestamps = set(a["timestamp"] for a in self.recent_anomalies)
        
        for i, ts in enumerate(self.recent_timestamps):
            if ts not in anomaly_timestamps:
                normal_indices.append(i)
        
        if len(normal_indices) < min_samples:
            logger.warning(f"Not enough normal samples: {len(normal_indices)}/{min_samples}")
            return False
            
        # Combine samples from normal indices
        normal_samples = [self.recent_samples[i][0] for i in normal_indices 
                          if i < len(self.recent_samples)]
        normal_samples = np.array(normal_samples)
        
        # Fit detector on normal samples
        self.detector.fit(normal_samples, self.feature_names)
        logger.info(f"Trained detector on {len(normal_samples)} recent normal samples")
        
        return True

def classify_severity(severity_score):
    """
    Classify severity level based on severity score.
    
    Args:
        severity_score (float): Severity score between 0 and 1.
        
    Returns:
        str: Severity level classification.
    """
    if severity_score < 0.2:
        return "Low"
    elif severity_score < 0.4:
        return "Medium-Low"
    elif severity_score < 0.6:
        return "Medium"
    elif severity_score < 0.8:
        return "Medium-High"
    else:
        return "High"

def get_anomaly_description(analysis):
    """
    Generate a human-readable description of an anomaly.
    
    Args:
        analysis (dict): Anomaly analysis from analyze_anomaly().
        
    Returns:
        str: Human-readable description.
    """
    severity = analysis["severity_level"]
    score = analysis.get("anomaly_score", "Unknown")
    
    if not analysis.get("top_contributors"):
        return f"Anomalous activity detected with {severity} severity (score: {score:.2f})."
    
    # Get top contributors
    contributors = analysis["top_contributors"][:3]  # Use top 3 at most
    
    if len(contributors) == 0:
        return f"Anomalous activity detected with {severity} severity (score: {score:.2f})."
    
    # Create description
    if len(contributors) == 1:
        return f"Anomalous {contributors[0]} activity detected with {severity} severity (score: {score:.2f})."
    elif len(contributors) == 2:
        return f"Anomalous {contributors[0]} and {contributors[1]} activity detected with {severity} severity (score: {score:.2f})."
    else:
        return f"Anomalous activity involving {contributors[0]}, {contributors[1]}, and {contributors[2]} detected with {severity} severity (score: {score:.2f})."

def recommend_action(analysis):
    """
    Recommend actions based on anomaly analysis.
    
    Args:
        analysis (dict): Anomaly analysis from analyze_anomaly().
        
    Returns:
        dict: Recommended actions.
    """
    severity = analysis["severity_level"]
    
    if severity == "Low":
        return {
            "priority": "Low",
            "actions": [
                "Monitor for repeated occurrences",
                "Log the event for future reference"
            ],
            "investigation": "Not required unless pattern continues"
        }
    elif severity == "Medium-Low":
        return {
            "priority": "Medium-Low",
            "actions": [
                "Monitor closely for 24 hours",
                "Check related systems for similar patterns"
            ],
            "investigation": "Recommended during next regular review"
        }
    elif severity == "Medium":
        return {
            "priority": "Medium",
            "actions": [
                "Flag for security team review",
                "Enable enhanced logging on affected systems",
                "Verify normal operation of affected services"
            ],
            "investigation": "Investigate within 12 hours"
        }
    elif severity == "Medium-High":
        return {
            "priority": "High",
            "actions": [
                "Alert security team immediately",
                "Isolate affected systems if disruption is minimal",
                "Collect forensic data for investigation",
                "Implement temporary security controls"
            ],
            "investigation": "Immediate investigation required"
        }
    else:  # High
        return {
            "priority": "Critical",
            "actions": [
                "Activate incident response protocol",
                "Isolate affected systems immediately",
                "Implement emergency security controls",
                "Collect comprehensive forensic data",
                "Consider contacting external security experts"
            ],
            "investigation": "Immediate full investigation, all hands required"
        }
