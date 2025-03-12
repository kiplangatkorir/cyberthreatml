# Real-World Testing with CyberThreat-ML 

This guide explains how to evaluate the CyberThreat-ML library against real-world attack data using public cybersecurity datasets.

## Overview

The CyberThreat-ML library includes a comprehensive testing framework for evaluating its performance against real-world network attacks. This allows security teams to:

1. Validate detection capabilities against known attack patterns
2. Measure performance metrics (accuracy, precision, recall, F1-score)
3. Compare signature-based and anomaly-based detection methods
4. Test zero-day detection capabilities

## Supported Datasets

CyberThreat-ML can be evaluated against several public cybersecurity datasets:

| Dataset | Description | Attack Types |
|---------|-------------|--------------|
| CICIDS2017 | Comprehensive network traffic dataset with multiple attack types | Brute Force, DoS/DDoS, Web Attack, Infiltration, Port Scan, Botnet |
| UNSW-NB15 | Modern attack dataset | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms |
| CIC-DDoS2019 | DDoS-focused dataset | Various DDoS attack methods |
| CSE-CIC-IDS2018 | Updated IDS dataset | Brute Force, XSS, SQL Injection, DoS, DDoS, Infiltration, etc. |

The included `real_world_testing.py` script is preconfigured to work with the CICIDS2017 dataset.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CyberThreat-ML installed
- Sufficient storage space for datasets (CICIDS2017 is several GB)

### Installation

1. Ensure CyberThreat-ML is installed
2. Download the dataset of choice (see below)
3. Create directories for the dataset and results

```bash
mkdir -p datasets/CICIDS2017
mkdir -p evaluation_results
```

### Downloading Datasets

The CICIDS2017 dataset is available from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

You can get download instructions by running:

```bash
python examples/real_world_testing.py --download
```

After downloading, extract the files to the `datasets/CICIDS2017` directory.

## Running Tests

The testing script provides several options for evaluation:

```bash
# Run both signature-based and anomaly-based evaluation
python examples/real_world_testing.py

# Only evaluate signature-based detection
python examples/real_world_testing.py --signature-only

# Only evaluate anomaly-based detection
python examples/real_world_testing.py --anomaly-only
```

## Understanding Results

The evaluation produces several outputs in the `evaluation_results/{timestamp}` directory:

### Signature-Based Detection

- `signature_based_results.txt`: Detailed performance metrics
- `signature_based_confusion_matrix.png`: Visualization of model performance
- `signature_based_training_history.png`: Training progression graph

### Anomaly-Based Detection

- `anomaly_based_results.txt`: Detailed performance metrics
- `anomaly_based_roc_curve.png`: ROC curve for anomaly detection
- `anomaly_based_pr_curve.png`: Precision-recall curve
- `anomaly_score_distribution.png`: Distribution of anomaly scores

### Model Comparison

- `model_comparison.txt`: Side-by-side comparison of methods
- `model_comparison_chart.png`: Visual comparison of performance metrics

## Advanced Usage

### Using Other Datasets

To use a different dataset, modify the following in `real_world_testing.py`:

1. Update the `SELECTED_FILES` constant with the dataset file names
2. Adjust the `ATTACK_MAPPING` dictionary to match the dataset's attack labels
3. Modify the `FEATURE_GROUPS` dictionary based on available features

### Customizing Evaluation Parameters

You can customize various parameters:

- Model architecture (`hidden_layers`, `dropout_rate`, etc.)
- Training parameters (`epochs`, `batch_size`, etc.)
- Anomaly detection method (`isolation_forest`, `local_outlier_factor`, etc.)
- Contamination rate for anomaly detection

## Interpreting Results

### Key Metrics to Consider

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual positives (detection rate)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

### Signature vs. Anomaly Detection

The evaluation compares both detection approaches:

- **Signature-Based**: Better for classifying known threats with high precision
- **Anomaly-Based**: Better for detecting unknown/zero-day threats

## Use Case: Security Operations Center (SOC)

For a SOC team, this testing framework can:

1. Validate the effectiveness of CyberThreat-ML before deployment
2. Determine appropriate detection thresholds
3. Identify which attack types might require additional tuning
4. Compare performance against existing security solutions

## Best Practices

1. **Representative Data**: Use datasets that represent your network environment
2. **Regular Re-evaluation**: Test periodically as new attack types emerge
3. **Multiple Metrics**: Don't rely on accuracy alone; consider precision and recall
4. **Combine Approaches**: Use both signature and anomaly detection in production

## Conclusion

Real-world testing is essential for ensuring that CyberThreat-ML provides effective protection against actual cyber threats. The provided testing framework allows detailed performance evaluation and tuning to optimize detection capabilities for your specific security needs.
