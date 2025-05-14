"""
Module for model interpretability and explainability for cybersecurity threat detection.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# import SHAP for model explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Try to import LIME as an alternative to SHAP
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from cyberthreat_ml.logger import CyberThreatLogger

# Configure logger
logger = CyberThreatLogger(name="cyberthreat_ml.interpretability").get_logger()

class ThreatInterpreter:
    """
    Class for interpreting and explaining threat detection model predictions.
    """
    
    def __init__(self, model, feature_names=None, class_names=None):
        """
        Initialize the threat interpreter.
        
        Args:
            model (cyberthreat_ml.model.ThreatDetectionModel): The trained model to interpret
            feature_names (list, optional): Names of features used by the model
            class_names (list, optional): Names of classes predicted by the model
        """
        self.model = model
        self.feature_names = feature_names or []
        self.class_names = class_names or []
        
        # Check for available explainers
        self.available_explainers = self._check_available_explainers()
        logger.info(f"Available explainers: {', '.join(self.available_explainers)}")
        
        # Initialize explainers
        self.explainers = {}
        self.initialized = False
        
    def _check_available_explainers(self):
        """
        Check which explainer libraries are available.
        
        Returns:
            list: List of available explainer names
        """
        available = []
        
        if SHAP_AVAILABLE:
            available.append("shap")
            
        if LIME_AVAILABLE:
            available.append("lime")
        
        # Rules-based is always available as a fallback
        available.append("rules")
        
        return available
    
    def initialize(self, background_data):
        """
        Initialize explainers with background data.
        
        Args:
            background_data (numpy.ndarray): Background data for initializing explainers
        """
        if self.initialized:
            return
            
        if not len(background_data):
            logger.warning("Empty background data provided. Explainers may not work correctly.")
            return
            
        # Initialize SHAP explainer
        if "shap" in self.available_explainers:
            try:
                # Check if model keras or sklearn model
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'predict'):
                    # TensorFlow model
                    self.explainers["shap"] = shap.KernelExplainer(
                        self.model.predict_proba, 
                        background_data
                    )
                else:
                    # Generic model
                    self.explainers["shap"] = shap.KernelExplainer(
                        self.model.predict_proba, 
                        background_data
                    )
                logger.info("SHAP explainer initialized")
            except Exception as e:
                logger.error(f"Error initializing SHAP explainer: {e}")
                
        # Initialize LIME explainer
        if "lime" in self.available_explainers:
            try:
                mode = "classification"
                if hasattr(self.model, 'num_classes') and self.model.num_classes > 2:
                    # Multi-class model
                    mode = "classification"
                
                self.explainers["lime"] = lime.lime_tabular.LimeTabularExplainer(
                    background_data,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    mode=mode
                )
                logger.info("LIME explainer initialized")
            except Exception as e:
                logger.error(f"Error initializing LIME explainer: {e}")
                
        # Rules-based explainer doesn't need initialization
        
        self.initialized = True
        
    def explain_prediction(self, input_data, method="auto", target_class=None, top_features=5):
        """
        Explain a prediction for the given input.
        
        Args:
            input_data (numpy.ndarray): Input data to explain (single sample)
            method (str): Method to use for explanation ('auto', 'shap', 'lime', or 'rules')
            target_class (int, optional): Class to explain for multi-class models
            top_features (int): Number of top features to include in explanation
            
        Returns:
            dict: Explanation data including feature importances
        """
        if not self.initialized:
            logger.warning("Explainer not initialized. Call initialize() first.")
            return self._rules_based_explanation(input_data)
            
        # Ensure input_data is 2D
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
            
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_data)
            prediction = self.model.predict(input_data)
        else:
            # Fallback if model doesn't have predict_proba
            prediction = self.model.predict(input_data)
            probabilities = np.zeros((input_data.shape[0], len(self.class_names) or 2))
            probabilities[np.arange(input_data.shape[0]), prediction] = 1
            
        # Extract prediction for the first sample
        if len(prediction.shape) > 0:
            prediction = prediction[0]
        if len(probabilities.shape) > 1:
            probabilities = probabilities[0]
            
        # Determine which class to explain
        if target_class is None:
            target_class = prediction
            
        # Choose method
        if method == "auto":
            for m in ["shap", "lime", "rules"]:
                if m in self.available_explainers:
                    method = m
                    break
                    
        # Generate explanation with the chosen method
        try:
            if method == "shap" and "shap" in self.available_explainers:
                return self._explain_with_shap(input_data, target_class, top_features)
            elif method == "lime" and "lime" in self.available_explainers:
                return self._explain_with_lime(input_data, target_class, top_features)
            else:
                return self._rules_based_explanation(input_data)
        except Exception as e:
            logger.error(f"Error generating explanation with {method}: {e}")
            return self._rules_based_explanation(input_data)
            
    def _explain_with_shap(self, input_data, target_class, top_features):
        """
        Generate explanation using SHAP.
        
        Args:
            input_data (numpy.ndarray): Input data to explain
            target_class (int): Class to explain
            top_features (int): Number of top features to include
            
        Returns:
            dict: Explanation data
        """
        explainer = self.explainers["shap"]
        
        # Generate SHAP values
        shap_values = explainer.shap_values(input_data)
        
        # For multi-class, shap_values is a list of arrays (one per class)
        if isinstance(shap_values, list):
            values = shap_values[target_class][0]
            expected_value = explainer.expected_value[target_class]
        else:
            values = shap_values[0]
            expected_value = explainer.expected_value
            
        # Get feature names
        feature_names = self.feature_names
        if not feature_names:
            feature_names = [f"Feature {i}" for i in range(len(values))]
            
        # Create explanation with feature importances
        feature_importances = [(name, float(value)) for name, value in zip(feature_names, values)]
        sorted_importances = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)
        
        # Limit to top features
        top_importances = sorted_importances[:top_features]
        
        # Create explanation object
        explanation = {
            "method": "shap",
            "expected_value": float(expected_value),
            "top_features": top_importances,
            "all_features": feature_importances,
            "feature_values": input_data[0].tolist(),
            "target_class": int(target_class)
        }
        
        return explanation
        
    def _explain_with_lime(self, input_data, target_class, top_features):
        """
        Generate explanation using LIME.
        
        Args:
            input_data (numpy.ndarray): Input data to explain
            target_class (int): Class to explain
            top_features (int): Number of top features to include
            
        Returns:
            dict: Explanation data
        """
        explainer = self.explainers["lime"]
        
        # Create a prediction function that returns probabilities
        def predict_fn(x):
            return self.model.predict_proba(x)
            
        # Generate LIME explanation
        exp = explainer.explain_instance(
            input_data[0],
            predict_fn,
            num_features=top_features,
            labels=[target_class]
        )
        
        # Extract feature importances from explanation
        feature_weights = exp.as_list(label=target_class)
        feature_importances = []
        
        for fw in feature_weights:
            # Extract feature name and value from LIME's string format
            if " = " in fw[0]:
                parts = fw[0].split(" = ")
                feature_name = parts[0]
                feature_value = float(parts[1])
            else:
                feature_name = fw[0]
                feature_value = None
                
            feature_importances.append((feature_name, float(fw[1])))
            
        # Create explanation object
        explanation = {
            "method": "lime",
            "top_features": feature_importances,
            "all_features": feature_importances,  # LIME already limits to top features
            "feature_values": input_data[0].tolist(),
            "target_class": int(target_class)
        }
        
        return explanation
        
    def _rules_based_explanation(self, input_data):
        """
        Generate a simple rules-based explanation when other methods are unavailable.
        
        Args:
            input_data (numpy.ndarray): Input data to explain
            
        Returns:
            dict: Explanation data
        """
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_data)
            prediction = self.model.predict(input_data)
        else:
            # Fallback if model doesn't have predict_proba
            prediction = self.model.predict(input_data)
            probabilities = np.zeros((input_data.shape[0], len(self.class_names) or 2))
            probabilities[np.arange(input_data.shape[0]), prediction] = 1
            
        # Extract prediction for the first sample
        if len(prediction.shape) > 0:
            prediction = prediction[0]
        if len(probabilities.shape) > 1:
            probabilities = probabilities[0]
            
        # Determine feature importances based on statistical properties
        feature_values = input_data[0]
        
        # Get feature names
        feature_names = self.feature_names
        if not feature_names:
            feature_names = [f"Feature {i}" for i in range(len(feature_values))]
            
        # Calculate basic statistical importances
        importances = []
        for i, (name, value) in enumerate(zip(feature_names, feature_values)):
            # Assign importance based on deviation from zero and feature index
            # (simple heuristic for demonstration)
            importance = abs(value) * (1 - 0.5 * (i / len(feature_values)))
            importances.append((name, float(importance), float(value)))
            
        # Sort by absolute importance
        sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
        
        # Create top features list (name, importance)
        top_features = [(name, imp) for name, imp, val in sorted_importances[:5]]
        
        # Create explanation object
        explanation = {
            "method": "rules",
            "description": "Simple statistical explanation (fallback method)",
            "top_features": top_features,
            "all_features": [(name, imp) for name, imp, val in sorted_importances],
            "feature_values": feature_values.tolist(),
            "target_class": int(prediction)
        }
        
        return explanation
        
    def plot_explanation(self, explanation, plot_type="bar", save_path=None):
        """
        Plot a visual explanation of a prediction.
        
        Args:
            explanation (dict): Explanation data from explain_prediction
            plot_type (str): Type of plot ('bar', 'waterfall', or 'force')
            save_path (str, optional): Path to save the plot image
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Set matplotlib to use non-interactive backend (important for headless environments)
        import matplotlib
        matplotlib.use('Agg')
        
        if explanation["method"] == "rules" and plot_type != "bar":
            logger.warning(f"Plot type '{plot_type}' not supported for rules-based explanation. Using 'bar' instead.")
            plot_type = "bar"
            
        if plot_type == "bar":
            return self._plot_bar_explanation(explanation, save_path)
        elif plot_type == "waterfall" and explanation["method"] == "shap":
            return self._plot_waterfall_explanation(explanation, save_path)
        elif plot_type == "force" and explanation["method"] == "shap":
            return self._plot_force_explanation(explanation, save_path)
        else:
            logger.warning(f"Unsupported plot type '{plot_type}' or method '{explanation['method']}'. Using 'bar' instead.")
            return self._plot_bar_explanation(explanation, save_path)
            
    def _plot_bar_explanation(self, explanation, save_path=None):
        """
        Create a bar plot visualization of feature importances.
        
        Args:
            explanation (dict): Explanation data
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Extract data
        top_features = explanation["top_features"]
        target_class = explanation["target_class"]
        
        # Get class name if available
        if self.class_names and target_class < len(self.class_names):
            class_name = self.class_names[target_class]
        else:
            class_name = f"Class {target_class}"
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate positive and negative contributions
        pos_features = [(name, value) for name, value in top_features if value > 0]
        neg_features = [(name, value) for name, value in top_features if value < 0]
        
        # Sort by absolute value
        pos_features.sort(key=lambda x: x[1], reverse=True)
        neg_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Plot positive features
        if pos_features:
            names, values = zip(*pos_features)
            y_pos = range(len(pos_features))
            ax.barh(y_pos, values, align='center', color='#66bb6a', alpha=0.8, label='Increases likelihood')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            
        # Plot negative features
        if neg_features:
            n_pos = len(pos_features)
            names, values = zip(*neg_features)
            y_pos = range(n_pos, n_pos + len(neg_features))
            ax.barh(y_pos, values, align='center', color='#ef5350', alpha=0.8, label='Decreases likelihood')
            ax.set_yticks(list(ax.get_yticks()) + y_pos)
            ax.set_yticklabels(list(ax.get_yticklabels()) + list(names))
            
        # Set plot title and labels
        title = f"Features contributing to {class_name} prediction"
        if explanation["method"] != "rules":
            title += f" ({explanation['method'].upper()})"
        ax.set_title(title)
        ax.set_xlabel('Feature Contribution')
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add legend
        if pos_features and neg_features:
            ax.legend()
            
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation plot saved to {save_path}")
            
        return fig
        
    def _plot_waterfall_explanation(self, explanation, save_path=None):
        """
        Create a waterfall plot for SHAP explanations.
        
        Args:
            explanation (dict): Explanation data (must be from SHAP)
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for waterfall plot")
            return self._plot_bar_explanation(explanation, save_path)
            
        try:
            # Extract data
            feature_names = [name for name, _ in explanation["all_features"]]
            feature_values = np.array([val for _, val in explanation["all_features"]])
            expected_value = explanation.get("expected_value", 0)
            target_class = explanation["target_class"]
            
            # Get class name if available
            if self.class_names and target_class < len(self.class_names):
                class_name = self.class_names[target_class]
            else:
                class_name = f"Class {target_class}"
                
            # Create SHAP explanation object
            shap_values = shap.Explanation(
                values=feature_values, 
                base_values=expected_value,
                data=explanation["feature_values"],
                feature_names=feature_names
            )
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot waterfall
            shap.plots.waterfall(shap_values, max_display=10, show=False)
            
            # Update title
            plt.title(f"SHAP Waterfall Plot for {class_name} Prediction")
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Waterfall plot saved to {save_path}")
                
            return fig
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {e}")
            return self._plot_bar_explanation(explanation, save_path)
            
    def _plot_force_explanation(self, explanation, save_path=None):
        """
        Create a force plot for SHAP explanations.
        
        Args:
            explanation (dict): Explanation data (must be from SHAP)
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object or None for HTML plots
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for force plot")
            return self._plot_bar_explanation(explanation, save_path)
            
        try:
            # Extract data
            feature_names = [name for name, _ in explanation["all_features"]]
            feature_values = np.array([val for _, val in explanation["all_features"]])
            expected_value = explanation.get("expected_value", 0)
            
            # Create SHAP explanation object
            shap_values = shap.Explanation(
                values=feature_values, 
                base_values=expected_value,
                data=explanation["feature_values"],
                feature_names=feature_names
            )
            
            # Create a matplotlib figure for the force plot
            fig, ax = plt.subplots(figsize=(12, 3))
            
            # Plot force plot
            shap.plots.force(shap_values, matplotlib=True, show=False, ax=ax)
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Force plot saved to {save_path}")
                
            return fig
        except Exception as e:
            logger.error(f"Error creating force plot: {e}")
            return self._plot_bar_explanation(explanation, save_path)
            
    def create_feature_importance_report(self, explanation, output_path=None):
        """
        Create a text report explaining the feature importances.
        
        Args:
            explanation (dict): Explanation data from explain_prediction
            output_path (str, optional): Path to save the report
            
        Returns:
            str: The text report
        """
        # Extract data
        top_features = explanation["top_features"]
        method = explanation["method"]
        target_class = explanation["target_class"]
        
        # Get class name if available
        if self.class_names and target_class < len(self.class_names):
            class_name = self.class_names[target_class]
        else:
            class_name = f"Class {target_class}"
            
        # Create report header
        lines = [
            "=================================================",
            f"THREAT DETECTION EXPLANATION REPORT ({method.upper()})",
            "=================================================",
            "",
            f"Prediction: {class_name} (Class ID: {target_class})",
            f"Explanation Method: {method}",
            ""
        ]
        
        # Add expected value for SHAP
        if method == "shap" and "expected_value" in explanation:
            lines.append(f"Base prediction value: {explanation['expected_value']:.4f}")
            lines.append("")
            
        # Add top features section
        lines.append("TOP CONTRIBUTING FEATURES")
        lines.append("-----------------------")
        
        for i, (name, value) in enumerate(top_features):
            # Format the contribution
            sign = "+" if value > 0 else ""
            lines.append(f"{i+1}. {name}: {sign}{value:.4f}")
            
        # Add interpretation section
        lines.append("")
        lines.append("INTERPRETATION")
        lines.append("-------------")
        
        # Add positive features
        pos_features = [(name, value) for name, value in top_features if value > 0]
        if pos_features:
            lines.append("Features increasing the likelihood of this threat:")
            for name, value in pos_features:
                lines.append(f"- {name} (contribution: +{value:.4f})")
                
        # Add negative features
        neg_features = [(name, value) for name, value in top_features if value < 0]
        if neg_features:
            lines.append("")
            lines.append("Features decreasing the likelihood of this threat:")
            for name, value in neg_features:
                lines.append(f"- {name} (contribution: {value:.4f})")
                
        # Add method-specific notes
        lines.append("")
        lines.append("NOTES")
        lines.append("-----")
        if method == "shap":
            lines.append("SHAP (SHapley Additive exPlanations) values represent the contribution")
            lines.append("of each feature to the prediction compared to the average prediction.")
        elif method == "lime":
            lines.append("LIME (Local Interpretable Model-agnostic Explanations) creates a local")
            lines.append("surrogate model to explain individual predictions.")
        else:
            lines.append("This is a simple statistical explanation based on feature values.")
            lines.append("More sophisticated explanation methods are available with optional dependencies.")
            
        # Join lines into a single string
        report = "\n".join(lines)
        
        # Save report if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report)
                logger.info(f"Explanation report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving explanation report: {e}")
                
        return report
        
# Function to get interpretability insights for a specific threat type
def get_threat_pattern_insights(interpreter, samples, threat_class_id, top_features=5, method="auto"):
    """
    Generate interpretability insights for a specific threat class.
    
    Args:
        interpreter (ThreatInterpreter): Initialized interpreter
        samples (numpy.ndarray): Sample data for the threat class
        threat_class_id (int): ID of the threat class to analyze
        top_features (int): Number of top features to include
        method (str): Method to use for explanations
        
    Returns:
        dict: Insights about the threat class
    """
    if not interpreter.initialized:
        logger.warning("Interpreter not initialized")
        return None
        
    if len(samples) == 0:
        logger.warning("No samples provided for threat insights")
        return None
        
    # Get class name
    if interpreter.class_names and threat_class_id < len(interpreter.class_names):
        class_name = interpreter.class_names[threat_class_id]
    else:
        class_name = f"Class {threat_class_id}"
        
    # Generate explanations for each sample
    explanations = []
    for i in range(min(len(samples), 10)):  # Limit to 10 samples for efficiency
        explanation = interpreter.explain_prediction(
            samples[i:i+1],
            method=method,
            target_class=threat_class_id,
            top_features=top_features
        )
        explanations.append(explanation)
        
    # Aggregate feature importances across samples
    feature_importances = defaultdict(list)
    
    for exp in explanations:
        for feature, importance in exp["top_features"]:
            feature_importances[feature].append(importance)
            
    # Calculate average importance for each feature
    avg_importances = {}
    for feature, values in feature_importances.items():
        avg_importances[feature] = sum(values) / len(values)
        
    # Sort features by average importance
    sorted_importances = sorted(
        avg_importances.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Create insights object
    insights = {
        "threat_class": {
            "id": threat_class_id,
            "name": class_name
        },
        "method": method,
        "sample_count": len(explanations),
        "key_features": sorted_importances[:top_features],
        "consistency": {}
    }
    
    # Calculate consistency of feature importance
    for feature, values in feature_importances.items():
        if len(values) >= 2:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            coefficient_variation = abs(std_dev / mean) if mean != 0 else float('inf')
            
            insights["consistency"][feature] = {
                "mean": mean,
                "std_dev": std_dev,
                "coefficient_variation": coefficient_variation
            }
            
    return insights
