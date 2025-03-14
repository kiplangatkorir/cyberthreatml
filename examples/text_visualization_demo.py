import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from collections import Counter
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shap

def generate_sample_data(n_samples=1000):
    """Generate synthetic cybersecurity event data with features"""
    attack_types = ['SQL Injection', 'XSS', 'DDoS', 'Brute Force', 'Malware']
    severity_levels = ['Low', 'Medium', 'High', 'Critical']
    
    # Generate more realistic features
    data = {
        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 168)) 
                     for _ in range(n_samples)],
        'packet_size': np.random.normal(500, 200, n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'HTTP'], n_samples),
        'port': np.random.choice([80, 443, 22, 3306, 8080], n_samples),
        'packet_rate': np.random.exponential(50, n_samples),
        'connection_duration': np.random.exponential(30, n_samples),
        'bytes_transferred': np.random.exponential(1000, n_samples),
        'error_rate': np.random.exponential(0.1, n_samples),
        'attack_type': np.random.choice(attack_types, n_samples, p=[0.2, 0.15, 0.3, 0.25, 0.1]),
        'severity': np.random.choice(severity_levels, n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'source_ip': [f'192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}' 
                     for _ in range(n_samples)],
        'success': np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    }
    
    return pd.DataFrame(data)

def create_and_train_model(X, y):
    """Create and train a simple neural network for binary classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    
    # Convert inputs to float32
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model

def plot_attack_distribution(df, save_dir):
    """Plot attack type distribution"""
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame for seaborn
    attack_counts = df['attack_type'].value_counts().reset_index()
    attack_counts.columns = ['Attack Type', 'Count']
    
    # Create bar plot with updated seaborn syntax
    sns.barplot(data=attack_counts, x='Count', y='Attack Type', hue='Attack Type', 
                palette='viridis', legend=False)
    plt.title('Distribution of Attack Types')
    plt.xlabel('Number of Attacks')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{save_dir}/attack_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_severity_heatmap(df, save_dir):
    """Create severity vs attack type heatmap"""
    plt.figure(figsize=(12, 6))
    
    # Create cross-tabulation with custom order
    severity_order = ['Critical', 'High', 'Medium', 'Low']
    heatmap_data = pd.crosstab(df['severity'], df['attack_type'])
    heatmap_data = heatmap_data.reindex(severity_order)
    
    # Plot heatmap with enhanced styling
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Number of Events'})
    plt.title('Attack Types by Severity Level')
    plt.ylabel('Severity')
    plt.xlabel('Attack Type')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{save_dir}/severity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_pattern(df, save_dir):
    """Visualize temporal attack patterns"""
    plt.figure(figsize=(12, 6))
    
    # Group by hour and attack type
    df['hour'] = df['timestamp'].dt.hour
    hourly_attacks = df.groupby(['hour', 'attack_type']).size().unstack()
    
    # Create stacked area plot
    hourly_attacks.plot(kind='area', stacked=True, alpha=0.7)
    plt.title('Attack Distribution by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Attacks')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{save_dir}/temporal_pattern.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_success_rate(df, save_dir):
    """Visualize attack success rates by type"""
    plt.figure(figsize=(12, 6))
    
    # Calculate success rates and create DataFrame
    success_rates = df.groupby('attack_type')['success'].agg(['mean', 'count']).reset_index()
    success_rates['mean'] = success_rates['mean'] * 100
    success_rates = success_rates.sort_values('mean', ascending=True)
    
    # Create horizontal bar plot with updated seaborn syntax
    sns.barplot(data=success_rates, x='mean', y='attack_type', hue='attack_type',
                palette='RdYlGn_r', legend=False)
    
    # Add count annotations
    for i, row in success_rates.iterrows():
        plt.text(row['mean'] + 1, i, f'n={row["count"]}', va='center')
    
    plt.title('Attack Success Rates by Type')
    plt.xlabel('Success Rate (%)')
    plt.ylabel('Attack Type')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{save_dir}/success_rates.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(df, model, X, save_dir):
    """Plot SHAP-based feature importance"""
    plt.figure(figsize=(12, 6))
    
    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, X[:100])
    shap_values = explainer.shap_values(X[:100])
    
    # Plot SHAP summary
    shap.summary_plot(shap_values[0], X[:100], feature_names=X.columns, 
                     show=False, plot_size=(12, 6))
    plt.title('Feature Importance (SHAP Values)')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{save_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("CyberThreat-ML Text Visualization Demo")
    print("=====================================\n")
    
    # Create visualizations directory
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate sample data
    print("Generating synthetic cybersecurity event data...")
    df = generate_sample_data()
    
    # Prepare features for model
    numeric_features = ['packet_size', 'packet_rate', 'connection_duration', 
                       'bytes_transferred', 'error_rate']
    categorical_features = ['protocol', 'port']
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df[categorical_features])
    
    # Combine numeric and encoded categorical features
    df_model = pd.concat([df[numeric_features], df_encoded], axis=1)
    
    # Convert to numpy arrays
    X = df_model.values
    y = (df['attack_type'] != 'Normal').astype(int).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining neural network for feature importance analysis...")
    model = create_and_train_model(X_train, y_train)
    
    # Evaluate model
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Performance:")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test AUC: {test_auc:.3f}")
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Total Events: {len(df):,}")
    print(f"Unique Attack Types: {df['attack_type'].nunique()}")
    print(f"Time Range: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    print(f"Most Common Attack: {df['attack_type'].mode()[0]}")
    print(f"Most Severe Attacks: {df[df['severity'] == 'Critical']['attack_type'].value_counts().head(1).index[0]}")
    
    # Create and save visualizations
    print("\nGenerating and saving visualizations...")
    
    print("\n1. Attack Distribution")
    plot_attack_distribution(df, save_dir)
    
    print("\n2. Severity Heatmap")
    plot_severity_heatmap(df, save_dir)
    
    print("\n3. Temporal Pattern")
    plot_temporal_pattern(df, save_dir)
    
    print("\n4. Success Rates")
    plot_success_rate(df, save_dir)
    
    print("\n5. Feature Importance")
    plot_feature_importance(df, model, df_model, save_dir)
    
    print(f"\nVisualization demo completed successfully!")
    print(f"All plots saved to: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()