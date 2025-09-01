"""
Setup configuration for PyTorch Mastery Hub
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pytorch-mastery-hub",
    version="1.0.0",
    author="PyTorch Mastery Hub Contributors",
    author_email="your-email@example.com",
    description="Comprehensive PyTorch learning resource with hands-on examples",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SatvikPraveen/pytorch-mastery-hub",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
        ],
        "advanced": [
            "transformers>=4.20.0",
            "datasets>=2.0.0",
            "optuna>=2.10.0",
            "wandb>=0.12.0",
            "lightning>=2.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "pytorch-hub=src.utils.cli:main",
        ],
    },
    keywords=[
        "pytorch", "machine-learning", "deep-learning", "education",
        "tutorials", "neural-networks", "computer-vision", "nlp"
    ],
    project_urls={
        "Bug Reports": "https://github.com/SatvikPraveen/pytorch-mastery-hub/issues",
        "Source": "https://github.com/SatvikPraveen/pytorch-mastery-hub",
        "Documentation": "https://pytorch-mastery-hub.readthedocs.io/",
    },
)