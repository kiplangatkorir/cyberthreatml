import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from collections import Counter
import pandas as pd
import os

def generate_sample_data(n_samples=1000):
    """Generate synthetic cybersecurity event data"""
    attack_types = ['SQL Injection', 'XSS', 'DDoS', 'Brute Force', 'Malware']
    severity_levels = ['Low', 'Medium', 'High', 'Critical']
    
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

def main():
    print("CyberThreat-ML Text Visualization Demo")
    print("=====================================\n")
    
    # Create visualizations directory
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate sample data
    print("Generating synthetic cybersecurity event data...")
    df = generate_sample_data()
    
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
    
    print(f"\nVisualization demo completed successfully!")
    print(f"All plots saved to: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()