# CyberThreat-ML: Explainability Guide

Explainability is a critical aspect of any security tool. This guide explores the explainability features of CyberThreat-ML that help security professionals understand *why* something was identified as a threat.

## Why Explainability Matters in Cybersecurity 

In cybersecurity, false positives can be costly and time-consuming. Without understanding why an alert was triggered, security teams may:
- Waste time investigating benign activities
- Miss actual threats due to alert fatigue
- Be unable to properly remediate issues
- Struggle to communicate findings to stakeholders

CyberThreat-ML addresses this by providing comprehensive explainability features for both signature-based and anomaly-based detections.

## Signature-Based Detection Explainability

### 1. SHAP Integration

CyberThreat-ML integrates SHAP (SHapley Additive exPlanations) for model interpretability. SHAP assigns each feature an importance value for a particular prediction.

```python
from cyberthreat_ml.explain import explain_prediction, plot_shap_summary

# Explain a prediction
explanation = explain_prediction(model, sample, feature_names)

# Get the most influential features
for feature, importance in explanation['feature_importance'][:5]:
    print(f"{feature}: {importance:.4f}")
```

### 2. ThreatInterpreter Class

For more advanced explainability, use the `ThreatInterpreter` class:

```python
from cyberthreat_ml.interpretability import ThreatInterpreter

# Create an interpreter
interpreter = ThreatInterpreter(
    model=model,
    feature_names=feature_names,
    class_names=class_names
)

# Initialize with background data
interpreter.initialize(background_data)

# Explain a prediction
explanation = interpreter.explain_prediction(
    input_data=sample,
    method="shap",  # Can be "shap", "lime", or "rules"
    target_class=1,  # Class to explain (0 = normal, 1 = attack)
    top_features=5   # Number of top features to include
)

# Visualize the explanation
interpreter.plot_explanation(explanation, plot_type="waterfall")
```

### 3. Visualization Types

The interpreter supports multiple visualization types:

```python
# Bar plot (default)
interpreter.plot_explanation(explanation, plot_type="bar")

# Waterfall plot (shows how each feature contributed)
interpreter.plot_explanation(explanation, plot_type="waterfall")

# Force plot (interactive visualization)
interpreter.plot_explanation(explanation, plot_type="force")
```

### 4. Feature Importance Reports

Generate text reports explaining feature contributions:

```python
# Create a feature importance report
report = interpreter.create_feature_importance_report(explanation)
print(report)
```

Example output:
```
THREAT DETECTION EXPLANATION
============================
Predicted class: Brute Force (confidence: 0.86)

TOP CONTRIBUTING FEATURES:
1. Authentication Failures: +0.432 (increased likelihood of threat)
   This feature showed 35 failures in 60 seconds (95th percentile)

2. Source IP Diversity: -0.217 (decreased likelihood of threat)
   This connection came from a known IP address

3. Connection Time: +0.198 (increased likelihood of threat)
   Connection occurred outside normal business hours (3:27 AM)

...
```

## Zero-Day (Anomaly-Based) Detection Explainability

The anomaly detection module provides specialized explainability features for zero-day threats.

### 1. Anomaly Analysis

When an anomaly is detected, you can get a detailed analysis:

```python
from cyberthreat_ml.anomaly import ZeroDayDetector, get_anomaly_description

# Detect anomalies
predictions, scores = detector.detect(test_data, return_scores=True)

# Analyze an anomaly
for i, pred in enumerate(predictions):
    if pred == -1:  # Anomaly
        analysis = detector.analyze_anomaly(test_data[i], scores[i])
        print(f"Anomaly score: {analysis['anomaly_score']:.4f}")
        print(f"Severity: {analysis['severity_level']} ({analysis['severity']:.4f})")
        print(f"Description: {get_anomaly_description(analysis)}")
```

### 2. Feature Contributions

The analysis includes detailed information about how each feature contributed to the anomaly:

```python
# Get feature deviations
for feature, details in analysis["feature_details"].items():
    deviation = details["deviation"]
    if deviation > 2.0:  # More than 2 standard deviations
        print(f"{feature}: {deviation:.2f} std devs from normal")
        print(f"  Value: {details['value']:.4f}")
        print(f"  Normal range: {details['baseline_mean']:.4f} ± {details['baseline_std']:.4f}")
```

### 3. Human-Readable Descriptions

The `get_anomaly_description()` function generates plain-language descriptions:

```python
from cyberthreat_ml.anomaly import get_anomaly_description

description = get_anomaly_description(analysis)
print(description)
```

Example output:
```
Anomalous Payload Size, Entropy, and Destination Port activity detected with High severity (score: 0.87).
```

### 4. Recommended Actions

The library can recommend actions based on the severity and nature of an anomaly:

```python
from cyberthreat_ml.anomaly import recommend_action

actions = recommend_action(analysis)
print(f"Priority: {actions['priority']}")
print("Recommended actions:")
for action in actions['actions']:
    print(f"  - {action}")
```

Example output:
```
Priority: High
Recommended actions:
  - Alert security team immediately
  - Isolate affected systems if disruption is minimal
  - Collect forensic data for investigation
  - Implement temporary security controls
```

## Combining Signature and Anomaly Explainability

For comprehensive threat analysis, combine both approaches:

```python
def analyze_threat(data):
    # First try signature-based detection
    signature_pred = signature_model.predict(data)
    signature_proba = signature_model.predict_proba(data)
    
    if signature_pred == 1:  # Known threat detected
        # Get threat class and confidence
        class_idx = np.argmax(signature_proba)
        confidence = signature_proba[0, class_idx]
        class_name = class_names[class_idx]
        
        # Get signature-based explanation
        signature_explanation = interpreter.explain_prediction(
            input_data=data,
            target_class=class_idx
        )
        
        return {
            "detection_type": "signature",
            "threat_class": class_name,
            "confidence": confidence,
            "explanation": signature_explanation
        }
    
    # If not detected by signature, try anomaly detection
    anomaly_pred, anomaly_score = zero_day_detector.detect(data, return_scores=True)
    
    if anomaly_pred[0] == -1:  # Anomaly detected
        # Get anomaly explanation
        anomaly_analysis = zero_day_detector.analyze_anomaly(data[0], anomaly_score[0])
        
        return {
            "detection_type": "anomaly",
            "severity": anomaly_analysis["severity_level"],
            "anomaly_score": anomaly_score[0],
            "description": get_anomaly_description(anomaly_analysis),
            "recommendations": recommend_action(anomaly_analysis),
            "analysis": anomaly_analysis
        }
    
    # No threat detected
    return {"detection_type": "normal"}
```

## Visualization of Explanations

### 1. SHAP Summary Plots

```python
from cyberthreat_ml.explain import explain_model, plot_shap_summary

# Explain model globally
explainer, shap_values = explain_model(
    model, 
    X_background, 
    X_explain=X_test,
    feature_names=feature_names
)

# Plot summary
plot_shap_summary(shap_values, feature_names)
```

### 2. SHAP Waterfall Plots

```python
from cyberthreat_ml.explain import plot_shap_waterfall

# Create a waterfall plot for a specific prediction
plot_shap_waterfall(shap_values, sample_idx=0, feature_names=feature_names)
```

### 3. Dashboard Integration

The `ThreatVisualizationDashboard` can incorporate explanation data:

```python
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter

# Create dashboard and interpreter
dashboard = ThreatVisualizationDashboard()
interpreter = ThreatInterpreter(model, feature_names, class_names)
interpreter.initialize(background_data)

# Start the dashboard
dashboard.start()

# When a threat is detected, get explanation and add to dashboard
def on_threat(result):
    # Add the threat to the dashboard
    dashboard.add_threat(result)
    
    # Get explanation
    explanation = interpreter.explain_prediction(
        input_data=result['features'],
        target_class=result['class_idx']
    )
    
    # Plot the explanation
    interpreter.plot_explanation(explanation, 
                                save_path=f"explanation_{result['id']}.png")
```

## Cross-Threat Pattern Analysis

For deeper insights across multiple detections:

```python
from cyberthreat_ml.interpretability import get_threat_pattern_insights

# Collect samples of a specific threat type
ddos_samples = []
for result in detection_results:
    if result['class_name'] == 'DDoS':
        ddos_samples.append(result['features'])

# Generate insights if we have enough samples
if len(ddos_samples) >= 5:
    ddos_samples = np.array(ddos_samples)
    insights = get_threat_pattern_insights(
        interpreter,
        samples=ddos_samples,
        threat_class_id=2,  # Index of DDoS class
        top_features=5
    )
    
    print("Common patterns in DDoS attacks:")
    for feature, importance in insights['common_patterns']:
        print(f"  {feature}: {importance:.4f}")
```

## Best Practices for Explainability

1. **Establish Baselines**: Understand what normal explanations look like for your environment.

2. **Feature Engineering for Explainability**: Design features that are inherently interpretable (e.g., "failed login ratio" is more interpretable than "feature_27").

3. **Combine Methods**: Use both SHAP and anomaly analysis for comprehensive explanations.

4. **Document Patterns**: Maintain a library of explanation patterns for common threats in your environment.

5. **Human Review**: Always have human analysts review and validate explanations for critical alerts.

6. **Feedback Loop**: Use analyst feedback on explanations to improve future detections.

## Conclusion

CyberThreat-ML's explainability features transform what would otherwise be "black box" ML detections into transparent, actionable security intelligence. This not only improves detection accuracy but also enhances the overall security posture by enabling informed decision-making.

By leveraging these explainability features, security teams can:
- Understand why something was flagged as a threat
- Prioritize alerts based on severity and evidence
- Take appropriate remediation actions
- Build institutional knowledge about threat patterns
- Reduce false positives by identifying explainable anomalies
