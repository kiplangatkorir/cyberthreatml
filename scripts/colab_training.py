"""
Colab training script for CyberThreat-ML using GPU acceleration.
This script assumes the dataset is uploaded to Google Drive.
"""
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mount_drive():
    """Mount Google Drive and return the path to the dataset."""
    from google.colab import drive
    drive.mount('/content/drive')
    return '/content/drive/MyDrive/CICIDS2023'

def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess the CICIDS2023 dataset from Google Drive.
    
    Args:
        dataset_path: Path to the dataset directory in Google Drive
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    logger.info("Loading dataset from Google Drive...")
    
    # Load all CSV files
    dataframes = []
    for file in Path(dataset_path).glob("*.csv"):
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # Combine all dataframes
    data = pd.concat(dataframes, ignore_index=True)
    
    # Convert labels to binary (0 for benign, 1 for attack)
    data['Label'] = data['Label'].apply(lambda x: 0 if x.lower() == 'benign' else 1)
    
    # Select numeric columns and handle missing values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_cols].fillna(0)
    
    # Split features and labels
    X = data.drop('Label', axis=1)
    y = data['Label']
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test, feature_names, scaler

def create_model(input_dim):
    """Create the neural network model with GPU optimization."""
    with tf.device('/GPU:0'):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with GPU acceleration and callbacks."""
    # Create callbacks
    checkpoint_path = "/content/drive/MyDrive/model_checkpoints/"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + "model_{epoch:02d}_{val_accuracy:.4f}.keras",
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"/content/drive/MyDrive/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    ]
    
    # Train with GPU
    with tf.device('/GPU:0'):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=256,
            callbacks=callbacks
        )
    return history

def generate_shap_explanations(model, X_test, feature_names):
    """Generate SHAP explanations for model interpretability."""
    # Create background dataset for SHAP
    background = X_test[np.random.choice(X_test.shape[0], 100, replace=False)]
    
    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test[:1000])  # Analyze first 1000 test samples
    
    # Save SHAP visualization
    shap.summary_plot(
        shap_values[0], 
        X_test[:1000], 
        feature_names=feature_names,
        show=False
    )
    plt.savefig('/content/drive/MyDrive/shap_summary.png')
    plt.close()
    
    return shap_values

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and generate metrics."""
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    with open('/content/drive/MyDrive/model_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    return metrics

def main():
    """Main training pipeline for Colab."""
    try:
        # Mount Google Drive
        dataset_path = mount_drive()
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data(dataset_path)
        
        # Create and train model
        model = create_model(X_train.shape[1])
        history = train_model(model, X_train, y_train, X_test, y_test)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        logger.info(f"Model metrics: {metrics}")
        
        # Generate explanations
        shap_values = generate_shap_explanations(model, X_test, feature_names)
        
        # Save the model
        model.save('/content/drive/MyDrive/final_model.keras')
        
        # Save the scaler
        with open('/content/drive/MyDrive/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
