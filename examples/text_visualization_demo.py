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
    """Plot threat type distribution with threat severity breakdown"""
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")
    
    # Create stacked bars for severity levels
    threat_severity = pd.crosstab(df['threat_type'], df['severity'])
    severity_order = ['Critical', 'High', 'Medium', 'Low']
    threat_severity = threat_severity[severity_order]
    
    # Plot stacked bars
    bottom = np.zeros(len(threat_severity))
    colors = ['#FF4B4B', '#FF8C42', '#FFC857', '#8FD694']
    
    for severity, color in zip(severity_order, colors):
        plt.bar(threat_severity.index, threat_severity[severity], 
                bottom=bottom, label=severity, color=color)
        bottom += threat_severity[severity]
    
    plt.title('Threat Distribution by Type and Severity', pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Threat Type', fontsize=10)
    plt.ylabel('Number of Events', fontsize=10)
    plt.legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/threat_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_severity_distribution(df, save_dir):
    """Plot severity distribution with percentage annotations"""
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")
    
    # Create cross-tabulation with percentages
    severity_threat = pd.crosstab(df['severity'], df['threat_type'], normalize='columns') * 100
    severity_order = ['Critical', 'High', 'Medium', 'Low']
    severity_threat = severity_threat.reindex(severity_order)
    
    # Create heatmap with custom styling
    sns.heatmap(severity_threat, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Percentage of Events (%)'})
    
    plt.title('Severity Distribution by Threat Type', pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Threat Type', fontsize=10)
    plt.ylabel('Severity Level', fontsize=10)
    
    # Add total counts as text
    for i, threat in enumerate(severity_threat.columns):
        total = len(df[df['threat_type'] == threat])
        plt.text(i, -0.2, f'n={total}', ha='center', va='top')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/severity_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_pattern(df, save_dir):
    """Plot temporal threat patterns with severity breakdown"""
    sns.set_style("darkgrid")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Plot 1: Hourly threat counts by severity
    df['hour'] = df['timestamp'].dt.hour
    severity_order = ['Critical', 'High', 'Medium', 'Low']
    colors = ['#FF4B4B', '#FF8C42', '#FFC857', '#8FD694']
    
    for severity, color in zip(severity_order, colors):
        hourly_severity = df[df['severity'] == severity].groupby('hour').size()
        ax1.plot(hourly_severity.index, hourly_severity.values, 
                 marker='o', label=severity, color=color, linewidth=2)
    
    ax1.set_title('Hourly Threat Distribution by Severity', pad=20, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hour of Day', fontsize=10)
    ax1.set_ylabel('Number of Events', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Severity', bbox_to_anchor=(1.05, 1))
    ax1.set_xticks(range(0, 24, 2))
    
    # Plot 2: Threat type distribution over time
    threat_types = df['threat_type'].unique()
    for threat in threat_types:
        if threat != 'Normal':
            hourly_threats = df[df['threat_type'] == threat].groupby('hour').size()
            ax2.plot(hourly_threats.index, hourly_threats.values, 
                    marker='o', label=threat, linewidth=2)
    
    ax2.set_title('Hourly Distribution by Threat Type', pad=20, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour of Day', fontsize=10)
    ax2.set_ylabel('Number of Events', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Threat Type', bbox_to_anchor=(1.05, 1))
    ax2.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/temporal_pattern.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_correlations(df, save_dir):
    """Plot feature correlations with enhanced styling"""
    plt.figure(figsize=(12, 10))
    sns.set_style("darkgrid")
    
    # Select and rename features for better readability
    feature_map = {
        'packet_rate': 'Packet Rate',
        'bytes_per_packet': 'Bytes/Packet',
        'connection_duration': 'Conn Duration',
        'error_rate': 'Error Rate',
        'is_threat': 'Is Threat'
    }
    
    features = list(feature_map.keys())
    correlation_matrix = df[features].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    
    # Create heatmap with custom styling
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    # Update labels
    labels = [feature_map[f] for f in features]
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels, rotation=0)
    
    plt.title('Network Feature Correlations\nfor Threat Detection', 
              pad=20, fontsize=12, fontweight='bold')
    
    # Add correlation interpretation guide
    plt.figtext(0.99, 0.15, 
                'Correlation Guide:\n' +
                '> 0.7: Strong Positive\n' +
                '0.3 to 0.7: Moderate Positive\n' +
                '-0.3 to 0.3: Weak\n' +
                '-0.7 to -0.3: Moderate Negative\n' +
                '< -0.7: Strong Negative',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=8,
                ha='right')
    
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