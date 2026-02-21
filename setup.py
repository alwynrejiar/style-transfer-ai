from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Advanced stylometry analysis system with modular architecture for analyzing writing styles using AI models."

setup(
    name="style-transfer-ai",
    version="1.3.0",
    description="Advanced stylometry analysis system with modular architecture",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Alwyn Rejicser",
    author_email="alwynrejicser@gmail.com",
    url="https://github.com/alwynrejicser/style-transfer-ai",
    download_url="https://github.com/alwynrejicser/style-transfer-ai/archive/v1.0.0.tar.gz",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    data_files=[
        ('', ['check_config.py']),
    ],
    install_requires=[
        "requests>=2.25.0",
        "spacy>=3.5.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "gemini": ["google-generativeai"],
        "all": [
            "openai>=1.0.0",
            "google-generativeai",
        ],
    },
    entry_points={
        "console_scripts": [
            "style-transfer-ai=src.main:cli_entry_point",
            "style-transfer-ai-config=check_config:main"
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="stylometry, text analysis, writing style, natural language processing, AI",
    project_urls={
        "Documentation": "https://github.com/alwynrejicser/style-transfer-ai/wiki",
        "Source": "https://github.com/alwynrejicser/style-transfer-ai",
        "Tracker": "https://github.com/alwynrejicser/style-transfer-ai/issues",
    },
)