import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter
import pandas as pd

def generate_sample_data(n_samples=1000):
    """Generate synthetic cybersecurity event data"""
    # Attack types
    attack_types = ['SQL Injection', 'XSS', 'DDoS', 'Brute Force', 'Malware']
    severity_levels = ['Low', 'Medium', 'High', 'Critical']
    
    # Generate random data
    data = {
        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 168)) 
                     for _ in range(n_samples)],
        'attack_type': np.random.choice(attack_types, n_samples, p=[0.2, 0.15, 0.3, 0.25, 0.1]),
        'severity': np.random.choice(severity_levels, n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'source_ip': [f'192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}' 
                     for _ in range(n_samples)],
        'success': np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    }
    
    return pd.DataFrame(data)

def plot_attack_distribution(df):
    """Plot attack type distribution"""
    plt.figure(figsize=(12, 6))
    attack_counts = df['attack_type'].value_counts()
    
    # Create bar plot
    sns.barplot(x=attack_counts.values, y=attack_counts.index, palette='viridis')
    plt.title('Distribution of Attack Types')
    plt.xlabel('Number of Attacks')
    plt.tight_layout()
    plt.show()

def plot_severity_heatmap(df):
    """Create severity vs attack type heatmap"""
    plt.figure(figsize=(12, 6))
    
    # Create cross-tabulation
    heatmap_data = pd.crosstab(df['severity'], df['attack_type'])
    
    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Attack Types by Severity Level')
    plt.tight_layout()
    plt.show()

def plot_temporal_pattern(df):
    """Visualize temporal attack patterns"""
    plt.figure(figsize=(12, 6))
    
    # Group by hour
    df['hour'] = df['timestamp'].dt.hour
    hourly_attacks = df.groupby('hour').size()
    
    # Create line plot
    sns.lineplot(x=hourly_attacks.index, y=hourly_attacks.values)
    plt.title('Attack Distribution by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Attacks')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_success_rate(df):
    """Visualize attack success rates by type"""
    plt.figure(figsize=(12, 6))
    
    # Calculate success rates
    success_rates = df.groupby('attack_type')['success'].mean().sort_values(ascending=True)
    
    # Create horizontal bar plot
    sns.barplot(x=success_rates.values * 100, y=success_rates.index, palette='RdYlGn_r')
    plt.title('Attack Success Rates by Type')
    plt.xlabel('Success Rate (%)')
    plt.tight_layout()
    plt.show()

def main():
    print("CyberThreat-ML Text Visualization Demo")
    print("=====================================\n")
    
    # Generate sample data
    print("Generating synthetic cybersecurity event data...")
    df = generate_sample_data()
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Total Events: {len(df)}")
    print(f"Unique Attack Types: {df['attack_type'].nunique()}")
    print(f"Time Range: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    print("\n1. Attack Distribution")
    plot_attack_distribution(df)
    
    print("\n2. Severity Heatmap")
    plot_severity_heatmap(df)
    
    print("\n3. Temporal Pattern")
    plot_temporal_pattern(df)
    
    print("\n4. Success Rates")
    plot_success_rate(df)
    
    print("\nVisualization demo completed successfully!")

if __name__ == "__main__":
    main()