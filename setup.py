from setuptools import setup, find_packages

setup(
    name="signal_modulation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "torch>=1.8.0",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.50.0",
        "matplotlib>=3.3.0",
        "pathlib>=1.0.1"
    ]
) 