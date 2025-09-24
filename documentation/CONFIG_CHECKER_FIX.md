# Configuration Checker Fix Guide

## Issue
When installing Style Transfer AI via `pip install`, the configuration checker fails with:
```
Error: Configuration checker not found at /usr/local/lib/python3.12/dist-packages/check_config.py
```

## Solution
This issue has been fixed in the latest version with multiple fallback mechanisms:

### 1. Integrated Configuration Check
The configuration checker is now integrated directly into the main application and no longer relies on external files.

### 2. Multiple Fallback Methods
The system now tries three methods in order:
1. **Integrated Checker**: Uses the built-in `check_system_requirements()` function
2. **Standalone Script**: Falls back to `check_config.py` if available
3. **Basic Manual Check**: Performs basic system validation as last resort

### 3. Update Your Installation

#### Option A: Update Existing Installation
```bash
pip install --upgrade --force-reinstall style-transfer-ai
```

#### Option B: Fresh Installation
```bash
pip uninstall style-transfer-ai
pip install style-transfer-ai
```

#### Option C: Development Installation
```bash
git clone https://github.com/alwynrejicser/style-transfer-ai.git
cd style-transfer-ai
pip install -e .
```

### 4. Verify the Fix
After updating, run the application and try option 9 (Check Configuration). You should see:

```
Running configuration check...
✅ Python version: 3.x.x
✅ requests: Available
✅ Directory: src/config
✅ Directory: src/utils
...
Configuration check completed successfully!
```

### 5. What Was Fixed

1. **Removed External Dependency**: No longer looks for `check_config.py` file
2. **Added Integrated Checker**: Built directly into `src/main.py`
3. **Enhanced Error Handling**: Multiple fallback mechanisms
4. **Better User Feedback**: Clear status messages and recommendations
5. **Ollama Detection**: Automatic detection of local Ollama server
6. **Dependency Validation**: Checks for required and optional packages

### 6. Additional Configuration Options

You can also run the configuration checker independently:
```bash
# If installed via pip
style-transfer-ai-config

# Or directly
python check_config.py
```

### 7. Troubleshooting

If you still experience issues:

1. **Check Python Version**: Ensure Python 3.7 or higher
2. **Verify Installation**: `pip list | grep style-transfer-ai`
3. **Check Dependencies**: `pip install requests`
4. **Reinstall**: Use the commands in section 3

The configuration checker now provides comprehensive feedback about your system setup and will guide you through any remaining installation issues.