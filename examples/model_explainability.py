"""
Example of model explainability with SHAP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import CyberThreat-ML components
from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.explain import (
    explain_model, explain_prediction, 
    plot_shap_summary, plot_shap_waterfall,
    get_top_features
)
from cyberthreat_ml.utils import split_data

def main():
    """
    Example of using SHAP for model explainability.
    """
    print("CyberThreat-ML Model Explainability Example")
    print("-------------------------------------------")
    
    # Step 1: Load or create a model
    print("\nStep 1: Loading model...")
    try:
        # Try to load a saved model
        model = load_model('threat_detection_model', 'threat_detection_metadata.json')
        print("Loaded existing model")
    except:
        print("No existing model found. Creating and training a new model...")
        model = create_and_train_model()
    
    # Step 2: Generate sample data for explanation
    print("\nStep 2: Generating sample data...")
    X, y = create_sample_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.25, random_state=42
    )
    
    # Create feature names for better visualization
    feature_names = [
        'Source Port', 'Destination Port', 'Packet Size', 
        'TCP Protocol', 'UDP Protocol', 'ICMP Protocol',
        'TCP Flags', 'TTL', 'Payload Entropy', 'High Port Combo'
    ]
    
    # Step 3: Explain the model globally
    print("\nStep 3: Generating global model explanation...")
    explainer, shap_values = explain_model(
        model, 
        X_background=X_train[:100],  # Use 100 training samples as background
        X_explain=X_test[:100],      # Explain 100 test samples
        feature_names=feature_names
    )
    
    # Create output directory for visualizations
    os.makedirs('explanations', exist_ok=True)
    
    # Step 4: Create and save summary plot
    print("\nStep 4: Creating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    plot_shap_summary(shap_values, feature_names=feature_names)
    plt.tight_layout()
    plt.savefig('explanations/shap_summary.png')
    plt.close()
    
    # Step 5: Get and print top features
    print("\nStep 5: Identifying top features by importance...")
    top_features = get_top_features(shap_values, feature_names=feature_names)
    print("Top features by importance:")
    for i, (feature, importance) in enumerate(top_features):
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    # Step 6: Explain specific examples
    print("\nStep 6: Explaining specific examples...")
    
    # Find a threat example
    threat_indices = np.where(y_test == 1)[0]
    if len(threat_indices) > 0:
        threat_idx = threat_indices[0]
        threat_sample = X_test[threat_idx]
        
        print(f"\nExplaining a threat example (probability: {model.predict_proba(threat_sample.reshape(1, -1))[0]:.4f}):")
        
        # Create waterfall plot for the threat
        plt.figure(figsize=(12, 6))
        plot_shap_waterfall(shap_values, sample_idx=threat_idx, feature_names=feature_names)
        plt.tight_layout()
        plt.savefig('explanations/threat_waterfall.png')
        plt.close()
        print("  Threat waterfall plot saved to 'explanations/threat_waterfall.png'")
    
    # Find a normal example
    normal_indices = np.where(y_test == 0)[0]
    if len(normal_indices) > 0:
        normal_idx = normal_indices[0]
        normal_sample = X_test[normal_idx]
        
        print(f"\nExplaining a normal example (probability: {model.predict_proba(normal_sample.reshape(1, -1))[0]:.4f}):")
        
        # Create waterfall plot for the normal sample
        plt.figure(figsize=(12, 6))
        plot_shap_waterfall(shap_values, sample_idx=normal_idx, feature_names=feature_names)
        plt.tight_layout()
        plt.savefig('explanations/normal_waterfall.png')
        plt.close()
        print("  Normal waterfall plot saved to 'explanations/normal_waterfall.png'")
    
    # Step 7: Demonstrate explaining a new prediction
    print("\nStep 7: Demonstrating explanation for a new prediction...")
    
    # Create a new sample that looks suspicious
    suspicious_sample = np.array([
        0.9,    # High source port (normalized)
        0.9,    # High destination port (normalized)
        0.1,    # Small packet size (normalized)
        1.0,    # TCP protocol
        0.0,    # Not UDP
        0.0,    # Not ICMP
        0.6,    # Some TCP flags
        0.4,    # Medium TTL
        0.9,    # High entropy
        1.0     # High port combination flag
    ])
    
    # Make and explain prediction
    prediction = model.predict_proba(suspicious_sample.reshape(1, -1))[0]
    explanation = explain_prediction(model, suspicious_sample, feature_names=feature_names)
    
    print(f"  Prediction for suspicious sample: {prediction:.4f}")
    print("  Feature contributions:")
    
    # Sort features by contribution magnitude
    feature_contribs = list(zip(feature_names, explanation['shap_values']))
    feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, contrib in feature_contribs:
        sign = '+' if contrib > 0 else ''
        print(f"    {feature}: {sign}{contrib:.6f}")
    
    # Create a report with all findings
    create_explanation_report(feature_names, top_features, explanation)
    print("\nExplanation report saved to 'explanations/report.html'")
    
    print("\nModel explainability example completed successfully!")


def create_and_train_model():
    """
    Create and train a threat detection model using synthetic data.
    
    Returns:
        ThreatDetectionModel: Trained model.
    """
    # Generate synthetic training data
    X, y = create_sample_data(n_samples=1000)
    
    # Create and train model
    model = ThreatDetectionModel(input_shape=(X.shape[1],))
    
    # Train model
    model.train(X, y, epochs=10, batch_size=32)
    
    # Save the model
    model.save_model('threat_detection_model', 'threat_detection_metadata.json')
    
    return model


def create_sample_data(n_samples=500):
    """
    Create synthetic data for cybersecurity threat detection.
    
    Args:
        n_samples (int): Number of samples to generate.
        
    Returns:
        tuple: (X, y) - features and labels.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate feature matrix with 10 features
    X = np.random.random((n_samples, 10))
    
    # The features are:
    # 0: Source port (normalized)
    # 1: Destination port (normalized)
    # 2: Packet size (normalized)
    # 3-5: Protocol one-hot (TCP, UDP, ICMP)
    # 6: TCP flags (normalized)
    # 7: TTL (normalized)
    # 8: Payload entropy
    # 9: High port combination flag
    
    # Make protocols one-hot
    X[:, 3:6] = 0  # Zero out protocol columns
    protocols = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    for i, p in enumerate(protocols):
        X[i, 3 + p] = 1
    
    # Generate synthetic labels based on rules
    # High entropy + unusual ports = more likely to be threats
    threat_scores = (
        X[:, 0] * 0.3 +      # Source port contribution
        X[:, 1] * 0.3 +      # Destination port contribution
        X[:, 8] * 0.4 +      # Payload entropy contribution
        X[:, 9] * 0.5        # High port combination flag
    )
    
    # TCP protocol with certain flags combinations are more suspicious
    tcp_factor = X[:, 3] * X[:, 6] * 0.2
    threat_scores += tcp_factor
    
    # Low TTL is sometimes suspicious
    ttl_factor = (1 - X[:, 7]) * 0.1
    threat_scores += ttl_factor
    
    # Small packet sizes can be suspicious
    size_factor = (1 - X[:, 2]) * 0.1
    threat_scores += size_factor
    
    # Add some random noise
    threat_scores += 0.1 * np.random.random(n_samples)
    
    # Normalize scores to 0-1
    threat_scores = (threat_scores - threat_scores.min()) / (threat_scores.max() - threat_scores.min())
    
    # Convert to binary labels (20% threats)
    threshold = np.percentile(threat_scores, 80)
    y = (threat_scores >= threshold).astype(int)
    
    return X, y


def create_explanation_report(feature_names, top_features, sample_explanation):
    """
    Create an HTML report with the explanation findings.
    
    Args:
        feature_names (list): Names of features.
        top_features (list): List of tuples (feature_name, importance_value).
        sample_explanation (dict): Explanation for a sample prediction.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CyberThreat-ML Model Explanation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; margin-top: 30px; }
            .container { max-width: 1000px; margin: 0 auto; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .feature-bar { height: 20px; background-color: #3498db; }
            .positive { color: #27ae60; }
            .negative { color: #e74c3c; }
            .summary { background-color: #f9f9f9; padding: 20px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CyberThreat-ML Model Explanation Report</h1>
            
            <div class="summary">
                <p>This report provides insights into how the threat detection model makes decisions, 
                highlighting the most important features and how they contribute to predictions.</p>
            </div>
            
            <h2>Feature Importance</h2>
            <p>The following features have the most influence on the model's predictions:</p>
            
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance</th>
                    <th>Visualization</th>
                </tr>
    """
    
    # Add rows for each top feature
    for i, (feature, importance) in enumerate(top_features):
        # Scale importance for visualization
        scaled_importance = min(100, importance * 500)
        
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{feature}</td>
                    <td>{importance:.6f}</td>
                    <td><div class="feature-bar" style="width: {scaled_importance}px"></div></td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Sample Prediction Explanation</h2>
            <p>The following shows how different features contributed to a specific prediction:</p>
            
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Contribution</th>
                </tr>
    """
    
    # Add rows for each feature contribution in the sample
    base_value = sample_explanation['base_value']
    prediction = sample_explanation['prediction']
    feature_values = list(zip(feature_names, sample_explanation['shap_values']))
    feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, value in feature_values:
        class_name = "positive" if value > 0 else "negative"
        sign = "+" if value > 0 else ""
        
        html_content += f"""
                <tr>
                    <td>{feature}</td>
                    <td class="{class_name}">{sign}{value:.6f}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
            
            <p>Base value: {base_value:.6f}</p>
            <p>Final prediction: {prediction:.6f}</p>
            
            <h2>Interpretation Guide</h2>
            <p>How to interpret this report:</p>
            <ul>
                <li><strong>Feature Importance</strong>: Shows which features have the greatest overall impact on model predictions.</li>
                <li><strong>Sample Prediction</strong>: Shows how each feature contributed to a specific prediction:
                    <ul>
                        <li><span class="positive">Positive values (green)</span>: Increase the likelihood of being classified as a threat</li>
                        <li><span class="negative">Negative values (red)</span>: Decrease the likelihood of being classified as a threat</li>
                    </ul>
                </li>
                <li><strong>Base value</strong>: The average model output over the training dataset</li>
                <li><strong>Final prediction</strong>: The probability that the sample is a threat (0-1)</li>
            </ul>
            
            <h2>Visualization Guide</h2>
            <p>Please refer to the following visualization files for more insights:</p>
            <ul>
                <li><strong>shap_summary.png</strong>: Overall feature importance and impact direction</li>
                <li><strong>threat_waterfall.png</strong>: Detailed breakdown of a threat prediction</li>
                <li><strong>normal_waterfall.png</strong>: Detailed breakdown of a normal prediction</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Create explanations directory if it doesn't exist
    os.makedirs('explanations', exist_ok=True)
    
    # Write HTML to file
    with open('explanations/report.html', 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    main()
