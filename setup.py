from setuptools import setup, find_packages

setup(
    name="cyberthreat_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.24.0",
        "shap>=0.40.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0"
    ],
    author="Cyberthreat ML Team",
    description="A machine learning framework for cyberthreat detection",
    python_requires=">=3.7"
)
