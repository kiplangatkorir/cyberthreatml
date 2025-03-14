"""
Module for real-time threat detection.
"""

import time
import threading
import queue
import numpy as np
from .model import ThreatDetectionModel
from .preprocessing import FeatureExtractor
from .logger import logger

class RealTimeDetector:
    """
    Class for real-time cyber threat detection.
    """
    
    def __init__(self, model, feature_extractor=None, threshold=0.5, 
                batch_size=32, processing_interval=1.0):
        """
        Initialize real-time detector.
        
        Args:
            model (ThreatDetectionModel): Trained threat detection model.
            feature_extractor (FeatureExtractor, optional): Feature extractor.
            threshold (float): Classification threshold.
            batch_size (int): Batch size for processing.
            processing_interval (float): Time interval between batch processing in seconds.
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        
        # Initialize queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Callback functions
        self.on_threat_detected = None
        self.on_data_processed = None
        
        # State variables
        self.is_running = False
        self._processor_thread = None
        
        logger.info(f"Initialized RealTimeDetector with threshold={threshold}, batch_size={batch_size}")
    
    def start(self):
        """
        Start the real-time detection process.
        """
        if self.is_running:
            logger.warning("Real-time detector is already running")
            return
        
        self.is_running = True
        self._processor_thread = threading.Thread(target=self._process_data, daemon=True)
        self._processor_thread.start()
        
        logger.info("Real-time detector started")
    
    def stop(self):
        """
        Stop the real-time detection process.
        """
        if not self.is_running:
            logger.warning("Real-time detector is not running")
            return
        
        self.is_running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
            self._processor_thread = None
        
        logger.info("Real-time detector stopped")
    
    def add_data(self, data):
        """
        Add data for threat detection.
        
        Args:
            data: Data to be processed (will be passed to feature extractor or used directly).
        """
        self.input_queue.put(data)
    
    def get_result(self, timeout=None):
        """
        Get a detection result from the output queue.
        
        Args:
            timeout (float, optional): Timeout in seconds. If None, block indefinitely.
            
        Returns:
            dict: Detection result or None if timeout occurred.
        """
        try:
            return self.output_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def register_threat_callback(self, callback):
        """
        Register a callback function for threat detection.
        
        Args:
            callback (callable): Function to call when a threat is detected.
                                The function should accept a result dictionary.
        """
        self.on_threat_detected = callback
    
    def register_processing_callback(self, callback):
        """
        Register a callback function for data processing.
        
        Args:
            callback (callable): Function to call when data is processed.
                                The function should accept a batch of result dictionaries.
        """
        self.on_data_processed = callback
    
    def _process_data(self):
        """
        Process data from the input queue.
        """
        while self.is_running:
            batch_data = []
            
            # Collect batch of data
            start_time = time.time()
            while len(batch_data) < self.batch_size and time.time() - start_time < self.processing_interval:
                try:
                    data_item = self.input_queue.get(block=True, timeout=0.1)
                    batch_data.append(data_item)
                except queue.Empty:
                    continue
            
            if not batch_data:
                time.sleep(0.1)
                continue
            
            try:
                # Process batch
                batch_results = self._process_batch(batch_data)
                
                # Put results in output queue
                for result in batch_results:
                    self.output_queue.put(result)
                
                # Call processing callback
                if self.on_data_processed:
                    self.on_data_processed(batch_results)
                
                # Call threat callback for detected threats
                if self.on_threat_detected:
                    for result in batch_results:
                        if result.get('is_threat', False):
                            self.on_threat_detected(result)
                
                # Sleep if needed to maintain processing interval
                elapsed = time.time() - start_time
                if elapsed < self.processing_interval:
                    time.sleep(self.processing_interval - elapsed)
            
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
    
    def _process_batch(self, batch_data):
        """
        Process a batch of data.
        
        Args:
            batch_data (list): Batch of data items.
            
        Returns:
            list: List of result dictionaries.
        """
        results = []
        
        try:
            # Extract features if needed
            if self.feature_extractor is not None:
                # Assume batch_data contains feature-extractable items
                features = np.array([self.feature_extractor.transform(item) for item in batch_data])
            else:
                # Assume batch_data is already feature vectors
                features = np.array(batch_data)
            
            # Check if model is binary or multi-class
            is_binary = getattr(self.model, 'is_binary', True)
            
            # Get predictions based on model type
            if is_binary:
                # Binary classification
                predictions = self.model.predict_proba(features)
                threats = predictions >= self.threshold
                
                # Create result dictionaries for binary classification
                for i, (data_item, prediction, is_threat) in enumerate(zip(batch_data, predictions, threats)):
                    result = {
                        'id': i,
                        'timestamp': time.time(),
                        'data': data_item,
                        'threat_score': float(prediction),
                        'is_threat': bool(is_threat),
                        'threshold': self.threshold,
                        'is_binary': True
                    }
                    results.append(result)
                
                threat_count = sum(threats)
                
            else:
                # Multi-class classification
                class_probabilities = self.model.predict_proba(features)
                predicted_classes = self.model.predict(features)
                
                # Get class names if available
                class_names = self.model.model_config.get('class_names', None)
                
                # Create result dictionaries for multi-class classification
                for i, (data_item, probs, pred_class) in enumerate(zip(batch_data, class_probabilities, predicted_classes)):
                    # For multi-class, any non-zero class is considered a threat
                    is_threat = pred_class > 0  # Class 0 is typically "normal"
                    
                    result = {
                        'id': i,
                        'timestamp': time.time(),
                        'data': data_item,
                        'class_probabilities': probs.tolist() if isinstance(probs, np.ndarray) else probs,
                        'predicted_class': int(pred_class),
                        'is_threat': bool(is_threat),
                        'confidence': float(probs[pred_class]) if len(probs) > pred_class else 0.0,
                        'threshold': self.threshold,
                        'is_binary': False,
                        'class_names': class_names
                    }
                    results.append(result)
                
                threat_count = sum(1 for r in results if r['is_threat'])
            
            logger.debug(f"Processed batch of {len(batch_data)} items, found {threat_count} threats "
                        f"using {'binary' if is_binary else 'multi-class'} classification")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return results


class PacketStreamDetector(RealTimeDetector):
    """
    Real-time detector for network packet streams.
    """
    
    def __init__(self, model, feature_extractor, threshold=0.5, 
                batch_size=32, processing_interval=1.0):
        """
        Initialize packet stream detector.
        
        Args:
            model (ThreatDetectionModel): Trained threat detection model.
            feature_extractor (FeatureExtractor): Feature extractor for packets.
            threshold (float): Classification threshold.
            batch_size (int): Batch size for processing.
            processing_interval (float): Time interval between batch processing in seconds.
        """
        super().__init__(model, feature_extractor, threshold, batch_size, processing_interval)
        
        # Packet statistics
        self.packet_count = 0
        self.threat_count = 0
        
        logger.info("Initialized PacketStreamDetector")
    
    def process_packet(self, packet_data):
        """
        Process a single network packet.
        
        Args:
            packet_data (dict): Network packet data.
        """
        self.packet_count += 1
        self.add_data(packet_data)
    
    def get_stats(self):
        """
        Get detector statistics.
        
        Returns:
            dict: Statistics dictionary.
        """
        return {
            'packet_count': self.packet_count,
            'threat_count': self.threat_count,
            'queue_size': self.input_queue.qsize(),
            'is_running': self.is_running,
            'threshold': self.threshold
        }
    
    def _process_batch(self, batch_data):
        """
        Process a batch of packet data.
        
        Args:
            batch_data (list): Batch of packet data.
            
        Returns:
            list: List of result dictionaries.
        """
        # Extract features from packet data
        features_list = []
        for packet in batch_data:
            try:
                # Extract features
                if self.feature_extractor:
                    features = self.feature_extractor.transform(packet)
                else:
                    # If no feature extractor, assume packet is already in feature format
                    features = packet
                
                features_list.append(features)
            except Exception as e:
                logger.error(f"Error extracting features from packet: {str(e)}")
                features_list.append(None)
        
        # Filter out failed feature extractions
        valid_indices = [i for i, f in enumerate(features_list) if f is not None]
        valid_features = np.array([features_list[i] for i in valid_indices])
        
        if len(valid_features) == 0:
            logger.warning("No valid features extracted from batch")
            return []
        
        # Check if model is binary or multi-class
        is_binary = getattr(self.model, 'num_classes', None) is None
        
        results = []
        
        if is_binary:
            # Binary classification
            predictions = self.model.predict_proba(valid_features)
            threats = predictions >= self.threshold
            
            # Update threat count for binary classification
            self.threat_count += int(np.sum(threats))
            
            # Create result dictionaries
            for i, (pred, is_threat) in enumerate(zip(predictions, threats)):
                packet = batch_data[valid_indices[i]]
                result = {
                    'id': valid_indices[i],
                    'timestamp': time.time(),
                    'packet': packet,
                    'threat_score': float(pred),
                    'is_threat': bool(is_threat),
                    'threshold': self.threshold,
                    'is_binary': True
                }
                results.append(result)
        else:
            # Multi-class classification
            class_probabilities = self.model.predict_proba(valid_features)
            predicted_classes = np.argmax(class_probabilities, axis=1)
            
            # Get class names if available
            class_names = getattr(self.model, 'class_names', None)
            
            # Update threat count for multi-class (any non-zero class is a threat)
            self.threat_count += int(np.sum(predicted_classes > 0))
            
            # Create result dictionaries
            for i, (probs, pred_class) in enumerate(zip(class_probabilities, predicted_classes)):
                packet = batch_data[valid_indices[i]]
                # For multi-class, any non-zero class is considered a threat
                is_threat = pred_class > 0  # Class 0 is "normal"
                
                result = {
                    'id': valid_indices[i],
                    'timestamp': time.time(),
                    'packet': packet,
                    'class_probabilities': probs.tolist(),
                    'predicted_class': int(pred_class),
                    'is_threat': bool(is_threat),
                    'confidence': float(probs[pred_class]),
                    'threshold': self.threshold,
                    'is_binary': False,
                    'class_names': class_names
                }
                results.append(result)
        
        return results
