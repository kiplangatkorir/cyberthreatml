"""
Minimal example for the CyberThreat-ML library.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the cyberthreat_ml package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    """
    Minimal example that doesn't require external libraries.
    This is used for demonstration when proper environment isn't available.
    """
    print("CyberThreat-ML Minimal Example")
    print("------------------------------------")
    
    print("\nChecking Python environment and dependencies...")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    print("\nChecking CyberThreat-ML package structure...")
    base_dir = Path(__file__).resolve().parent.parent
    cyberthreat_ml_dir = base_dir / "cyberthreat_ml"
    
    if os.path.isdir(cyberthreat_ml_dir):
        print(f"✓ Found CyberThreat-ML package at: {cyberthreat_ml_dir}")
        
        # List the module files
        print("\nModules found in CyberThreat-ML package:")
        for module_file in os.listdir(cyberthreat_ml_dir):
            if module_file.endswith(".py"):
                print(f"  - {module_file}")
    else:
        print(f"✗ CyberThreat-ML package not found at: {cyberthreat_ml_dir}")
    
    print("\nChecking examples directory...")
    examples_dir = base_dir / "examples"
    if os.path.isdir(examples_dir):
        print(f"✓ Found examples directory at: {examples_dir}")
        
        # List the example scripts
        print("\nExample scripts found:")
        for example_file in os.listdir(examples_dir):
            if example_file.endswith(".py"):
                print(f"  - {example_file}")
    else:
        print(f"✗ Examples directory not found at: {examples_dir}")
    
    print("\nChecking documentation...")
    docs_dir = base_dir / "docs"
    if os.path.isdir(docs_dir):
        print(f"✓ Found documentation directory at: {docs_dir}")
        
        # List the documentation files
        print("\nDocumentation files found:")
        for doc_file in os.listdir(docs_dir):
            if doc_file.endswith(".md") or doc_file.endswith(".txt"):
                print(f"  - {doc_file}")
    else:
        print(f"✗ Documentation directory not found at: {docs_dir}")
    
    # Simulate a simple threat detection
    print("\nSimulating a basic threat detection...")
    threat_detected = True
    confidence = 0.87
    
    if threat_detected:
        print(f"⚠️ Threat detected with {confidence:.2f} confidence!")
        print("Threat type: Brute Force Attack")
        print("Source IP: 192.168.1.100")
        print("Target IP: 10.0.0.5")
        print("Timestamp: 2025-03-11 09:42:13")
        print("Severity: High")
    else:
        print("No threats detected.")
    
    print("\nMinimal example completed successfully!")

if __name__ == "__main__":
    main()