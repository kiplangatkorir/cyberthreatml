import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from model import create_model
import shap
from explainability import explain_predictions

# Generate synthetic cybersecurity data
def generate_sample_data(n_samples=1000):
    """Generate synthetic network traffic data with features that might indicate threats."""
    np.random.seed(42)
    
    # Features: packet_size, frequency, duration, port_number, protocol_type
    normal_traffic = np.random.normal(loc=0.5, scale=0.2, size=(n_samples // 2, 5))
    anomalous_traffic = np.random.normal(loc=0.8, scale=0.3, size=(n_samples // 2, 5))
    
    X = np.vstack([normal_traffic, anomalous_traffic])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    return X, y

def plot_training_history(history):
    """Plot training metrics."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Generate data
    print("Generating synthetic data...")
    X, y = generate_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    print("Training model...")
    model = create_model(input_shape=(5,))
    history = model.fit(
        X_train_scaled, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Generate SHAP values for model interpretability
    print("Generating SHAP values...")
    shap_values = explain_predictions(model, X_test_scaled[:100])  # Using first 100 test samples
    
    print("\nTest completed successfully!")
    print("Check 'training_history.png' for visualization of training metrics.")

if __name__ == "__main__":
    main()
