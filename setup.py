#!/usr/bin/env/python3
import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "QuickClus",
    version = "0.1.0",
    author = "Aitor Porcel Laburu",
    description = "UMAP + HDBSCAN for numeric and/or categorical variables",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/aitorporcel/QuickClus",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.7",
    license = "MIT",
    install_requires = [
        "umap_learn>=0.5.1",
        "numpy>=1.20.2",
        "hdbscan>=0.8.28",
        "numba>=0.51.2",
        "pandas>=1.3.4",
        "scikit_learn>=1.0.2",
        "matplotlib>=3.5.0",
        "optuna>=2.10.0",
        "seaborn>=0.11.2",
    ],
)

