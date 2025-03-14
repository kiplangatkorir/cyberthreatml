import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_threat_data(n_samples=1000):
    """Generate synthetic cybersecurity threat data"""
    # Core threat features
    data = {
        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 168)) 
                     for _ in range(n_samples)],
        'packet_rate': np.random.exponential(50, n_samples),
        'bytes_per_packet': np.random.normal(500, 200, n_samples),
        'connection_duration': np.random.exponential(30, n_samples),
        'error_rate': np.random.exponential(0.1, n_samples),
        'is_threat': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    # Add threat-specific features
    data['threat_type'] = ['Normal' if not is_threat else 
                          np.random.choice(['DDoS', 'Brute Force', 'Data Exfiltration'])
                          for is_threat in data['is_threat']]
    
    data['severity'] = ['Low' if not is_threat else 
                       np.random.choice(['Medium', 'High', 'Critical'], p=[0.5, 0.3, 0.2])
                       for is_threat in data['is_threat']]
    
    return pd.DataFrame(data)

def create_threat_model(input_shape):
    """Create TensorFlow-based threat detection model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def plot_threat_distribution(df, save_dir):
    """Plot threat type distribution"""
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame for plotting
    threat_data = df['threat_type'].value_counts().reset_index()
    threat_data.columns = ['Threat Type', 'Count']
    
    sns.barplot(data=threat_data, x='Count', y='Threat Type', palette='viridis')
    plt.title('Distribution of Threat Types')
    plt.xlabel('Number of Events')
    plt.ylabel('Threat Type')
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/threat_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_severity_distribution(df, save_dir):
    """Plot severity distribution by threat type"""
    plt.figure(figsize=(12, 6))
    
    # Create cross-tabulation
    severity_threat = pd.crosstab(df['severity'], df['threat_type'])
    severity_order = ['Low', 'Medium', 'High', 'Critical']
    severity_threat = severity_threat.reindex(severity_order)
    
    sns.heatmap(severity_threat, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Threat Types by Severity Level')
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/severity_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_pattern(df, save_dir):
    """Plot temporal threat patterns"""
    plt.figure(figsize=(12, 6))
    
    # Group by hour and calculate threat counts
    df['hour'] = df['timestamp'].dt.hour
    hourly_threats = df[df['is_threat'] == 1].groupby('hour').size()
    
    # Create line plot
    plt.plot(hourly_threats.index, hourly_threats.values, marker='o')
    plt.title('Threat Distribution by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Threats')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/temporal_pattern.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_correlations(df, save_dir):
    """Plot feature correlations"""
    plt.figure(figsize=(10, 8))
    
    # Select numeric features
    features = ['packet_rate', 'bytes_per_packet', 'connection_duration', 
                'error_rate', 'is_threat']
    correlation_matrix = df[features].corr()
    
    # Create heatmap with custom styling
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Feature Correlations with Threat Detection')
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/feature_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("CyberThreat-ML Visualization Demo")
    print("================================\n")
    
    # Create visualizations directory
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate threat data
    print("Generating synthetic threat data...")
    df = generate_threat_data()
    
    # Prepare features for model
    features = ['packet_rate', 'bytes_per_packet', 'connection_duration', 'error_rate']
    X = df[features].values
    y = df['is_threat'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    
    # Train model
    print("\nTraining threat detection model...")
    model = create_threat_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Display statistics
    print("\nDataset Statistics:")
    print(f"Total Events: {len(df):,}")
    print(f"Threat Events: {df['is_threat'].sum():,} ({df['is_threat'].mean():.1%})")
    print(f"Time Range: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    print(f"\nModel Performance:")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("1. Threat Distribution")
    plot_threat_distribution(df, save_dir)
    
    print("2. Severity Distribution")
    plot_severity_distribution(df, save_dir)
    
    print("3. Temporal Pattern")
    plot_temporal_pattern(df, save_dir)
    
    print("4. Feature Correlations")
    plot_feature_correlations(df, save_dir)
    
    print(f"\nVisualization demo completed successfully!")
    print(f"All plots saved to: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()