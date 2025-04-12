"""
Training script for CyberThreat-ML model using real-world network security data.
Supports multiple data sources and provides detailed training metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.preprocessing import FeatureExtractor
from cyberthreat_ml.evaluation import evaluate_model
from cyberthreat_ml.logger import CyberThreatLogger
from cyberthreat_ml.visualization import plot_training_history
from cyberthreat_ml.interpretability import ThreatInterpreter

# Initialize logger
logger = CyberThreatLogger("model_training", log_level="INFO").get_logger()

class DataLoader:
    """Handles loading and preprocessing of various network security datasets."""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        
    def load_pcap_data(self, pcap_file):
        """Load and process pcap network capture files."""
        try:
            from scapy.all import rdpcap
            packets = rdpcap(str(self.data_dir / pcap_file))
            features = []
            labels = []
            
            for packet in packets:
                # Extract features from packet
                feature_vector = self.feature_extractor.transform(packet)
                features.append(feature_vector)
                
                # Determine label based on packet characteristics
                label = self._classify_packet(packet)
                labels.append(label)
            
            return np.array(features), np.array(labels)
        except Exception as e:
            logger.error(f"Error loading pcap file: {e}")
            return None, None
    
    def load_csv_data(self, csv_file):
        """Load preprocessed CSV data with features and labels."""
        try:
            df = pd.read_csv(self.data_dir / csv_file)
            
            # Separate features and labels
            X = df.drop(['label', 'timestamp'], axis=1, errors='ignore')
            y = df['label']
            
            return X.values, y.values
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return None, None
    
    def _classify_packet(self, packet):
        """Classify a packet based on its characteristics."""
        # Add your packet classification logic here
        # This is a simplified example
        if 'TCP' in packet and packet['TCP'].flags.S:
            return 1  # Potential port scan
        return 0  # Normal traffic

def prepare_data(X, y, test_size=0.2, val_size=0.2):
    """Prepare and split data for training."""
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Second split: separate validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_model(model, data_splits, batch_size=32, epochs=50):
    """Train the model with early stopping and learning rate scheduling."""
    (X_train, y_train), (X_val, y_val), _ = data_splits
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/checkpoint_{epoch:02d}_{val_accuracy:.4f}.keras',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def main():
    """Main training pipeline."""
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_output", exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader("datasets")
    
    # Load and combine data from multiple sources
    logger.info("Loading training data...")
    features_list = []
    labels_list = []
    
    # Load PCAP files
    pcap_files = list(Path("datasets").glob("*.pcap"))
    for pcap_file in pcap_files:
        X, y = data_loader.load_pcap_data(pcap_file)
        if X is not None and y is not None:
            features_list.append(X)
            labels_list.append(y)
    
    # Load CSV files
    csv_files = list(Path("datasets").glob("*.csv"))
    for csv_file in csv_files:
        X, y = data_loader.load_csv_data(csv_file)
        if X is not None and y is not None:
            features_list.append(X)
            labels_list.append(y)
    
    if not features_list:
        logger.error("No valid data files found!")
        return
    
    # Combine all data
    X = np.concatenate(features_list)
    y = np.concatenate(labels_list)
    
    # Prepare data splits
    logger.info("Preparing data splits...")
    data_splits = prepare_data(X, y)
    
    # Create and compile model
    logger.info("Creating model...")
    input_shape = X.shape[1:]
    model = ThreatDetectionModel(input_shape)
    
    # Train model
    logger.info("Starting training...")
    history = train_model(model, data_splits)
    
    # Evaluate model
    logger.info("Evaluating model...")
    _, _, (X_test, y_test) = data_splits
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = f"models/threat_detection_model_{timestamp}.keras"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Plot training history
    plot_path = f"training_output/training_history_{timestamp}.png"
    plot_training_history(history, plot_path)
    
    # Initialize interpreter for feature importance analysis
    interpreter = ThreatInterpreter(model)
    interpreter.analyze_feature_importance(X_test[:100])  # Analyze a subset
    
    # Save evaluation results
    results_path = f"training_output/evaluation_results_{timestamp}.txt"
    with open(results_path, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training completed at: {datetime.now()}\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 20 + "\n")
        for metric, value in evaluation_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    logger.info(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()
