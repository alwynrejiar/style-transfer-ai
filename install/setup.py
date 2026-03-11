from setuptools import setup, find_packages

setup(
    name="style-transfer-ai",
    version="1.4.0",
    description="Advanced stylometry analysis system with modular architecture",
    author="Style Transfer AI Team",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        # Optional dependencies (users can install as needed)
        # "openai>=1.0.0",          # For OpenAI API
        # "google-generativeai",    # For Gemini API
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "gemini": ["google-generativeai"],
        "all": [
            "openai>=1.0.0", 
            "google-generativeai"
        ]
    },
    entry_points={
        "console_scripts": [
            "style-transfer-ai=src.main:cli_entry_point"
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
        "Documentation": "https://github.com/yourusername/style-transfer-ai/wiki",
        "Source": "https://github.com/yourusername/style-transfer-ai",
        "Tracker": "https://github.com/yourusername/style-transfer-ai/issues",
    },
)