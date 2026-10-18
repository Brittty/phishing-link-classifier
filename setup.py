from setuptools import setup, find_packages

setup(
    name="phishing-link-classifier",
    version="0.1.0",
    description="Offline phishing URL classifier using scikit-learn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
        "tldextract>=5.0.0",
        "idna>=3.7",
    ],
    entry_points={
        "console_scripts": [
            "phish-clf=src.cli:main",
        ],
    },
)