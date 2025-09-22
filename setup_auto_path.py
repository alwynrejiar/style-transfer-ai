"""
Setup script with automatic PATH configuration for Windows.
This ensures style-transfer-ai CLI is globally accessible after pip install.
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import sys
import platform


class PostInstallCommand(install):
    """Post-installation for Windows PATH setup."""
    
    def run(self):
        # Run the normal installation
        install.run(self)
        
        # Add to PATH on Windows
        if platform.system() == 'Windows':
            self.add_to_windows_path()
    
    def add_to_windows_path(self):
        """Automatically add Python user Scripts to Windows PATH."""
        try:
            import winreg
            
            # Get user Scripts directory
            user_scripts = os.path.join(
                os.path.expanduser("~"), 
                "AppData", "Roaming", "Python", 
                f"Python{sys.version_info.major}{sys.version_info.minor}", 
                "Scripts"
            )
            
            # Open user environment variables registry key
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS) as key:
                try:
                    # Get current PATH
                    current_path, _ = winreg.QueryValueEx(key, "PATH")
                except FileNotFoundError:
                    current_path = ""
                
                # Check if already in PATH
                if user_scripts.lower() not in current_path.lower():
                    # Add to PATH
                    new_path = f"{current_path};{user_scripts}" if current_path else user_scripts
                    winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
                    
                    print(f"\n✅ Added to PATH: {user_scripts}")
                    print("🔄 Restart your terminal to use 'style-transfer-ai' globally!")
                else:
                    print(f"✅ Already in PATH: {user_scripts}")
                    
        except Exception as e:
            print(f"\n⚠️  Could not automatically add to PATH: {e}")
            print(f"Manual setup: Add {user_scripts} to your PATH")
            print("Or run our setup script: add_to_path.ps1")


# Read long description from README
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Advanced stylometry analysis system with modular architecture for analyzing writing styles using AI models."


setup(
    name="style-transfer-ai",
    version="1.2.0",  # Increment for auto-PATH feature
    description="Advanced stylometry analysis system with automatic PATH setup",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Alwyn Rejicser",
    author_email="alwynrejicser@gmail.com",
    url="https://github.com/alwynrejicser/style-transfer-ai",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "requests>=2.25.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "gemini": ["google-generativeai"],
        "all": ["openai>=1.0.0", "google-generativeai"]
    },
    entry_points={
        "console_scripts": [
            "style-transfer-ai=src.main:cli_entry_point",
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.7",
    keywords="stylometry, writing analysis, AI, natural language processing, text analysis",
)