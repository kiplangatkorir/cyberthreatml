"""
Real-world testing of the cyberthreat detection model using CIC-IDS2023 dataset.
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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model architecture from our research
def create_model(input_dim):
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

def load_and_preprocess_data():
    dataset_dir = Path(__file__).resolve().parent.parent / "datasets" / "CICIDS2023"
    
    # Load all CSV files
    dataframes = []
    for file in dataset_dir.glob("*.csv"):
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # Combine all dataframes
    data = pd.concat(dataframes, ignore_index=True)
    
    # Identify attack types as binary (0 for benign, 1 for attack)
    data['Label'] = data['Label'].apply(lambda x: 0 if x.lower() == 'benign' else 1)
    
    # Remove any non-numeric columns and handle missing values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_cols].fillna(0)
    
    # Split features and labels
    X = data.drop('Label', axis=1)
    y = data['Label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and generate metrics."""
    results = model.evaluate(X_test, y_test, verbose=0)
    metrics = dict(zip(model.metrics_names, results))
    
    logger.info("\nModel Performance Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    
    return metrics

def generate_shap_explanations(model, X_test, feature_names):
    """Generate SHAP explanations for model predictions."""
    # Create SHAP explainer
    background = X_test[np.random.choice(X_test.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(model, background)
    
    # Calculate SHAP values for a subset of test data
    sample_size = min(1000, X_test.shape[0])
    sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
    shap_values = explainer.shap_values(X_test[sample_indices])
    
    # Get feature importance
    feature_importance = np.abs(shap_values[0]).mean(0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    logger.info(importance_df.head(10))
    
    return importance_df

def main():
    logger.info("Loading and preprocessing CIC-IDS2023 dataset...")
    X, y, feature_names = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train model
    logger.info("Training model...")
    model = create_model(X.shape[1])
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Generate SHAP explanations
    logger.info("\nGenerating SHAP explanations...")
    importance_df = generate_shap_explanations(model, X_test, feature_names)
    
    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    importance_df.to_csv(results_dir / "feature_importance.csv", index=False)
    logger.info(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main()
