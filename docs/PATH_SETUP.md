# 🚀 Style Transfer AI - Automatic PATH Setup

This directory contains scripts to automatically add the Style Transfer AI CLI to your system PATH, making it globally accessible from any terminal.

## 🔧 Automatic PATH Setup Options

### Option 1: PowerShell Script (Recommended)
```powershell
# Run from this directory
PowerShell -ExecutionPolicy Bypass -File "add_to_path.ps1"
```

### Option 2: Batch File (Windows)
```cmd
# Double-click or run from command prompt
setup_path.bat
```

### Option 3: Manual PowerShell Command
```powershell
# Add Python user Scripts to PATH
$userScriptsPath = "$env:APPDATA\Python\Python313\Scripts"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
[Environment]::SetEnvironmentVariable("PATH", "$currentPath;$userScriptsPath", "User")
```

## 📍 Installation Locations

When you run `pip install style-transfer-ai`, the package installs to:

| Component | Location |
|-----------|----------|
| **Package Code** | `%APPDATA%\Python\Python313\site-packages\` |
| **CLI Executable** | `%APPDATA%\Python\Python313\Scripts\style-transfer-ai.exe` |

## ✅ After PATH Setup

Once PATH is configured, you can run from anywhere:

```bash
# From any directory
style-transfer-ai

# With arguments
style-transfer-ai --analyze sample.txt

# Interactive mode
style-transfer-ai --interactive
```

## 🔄 Refresh Environment

After running the setup script:

### In Current Terminal:
```powershell
# PowerShell
$env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'User') + ';' + [Environment]::GetEnvironmentVariable('PATH', 'Machine')

# Command Prompt
refreshenv
```

### Or Simply:
- **Restart your terminal**
- **Open a new terminal window**

## 🧪 Test Installation

Verify the CLI is working:
```bash
# Check version
style-transfer-ai --version

# View help
style-transfer-ai --help

# Test from different directory
cd C:\ && style-transfer-ai
```

## 🛠️ Troubleshooting

### CLI Not Found After Setup:
1. **Restart terminal** or run `refreshenv`
2. **Check PATH**: `echo $env:PATH` (PowerShell) or `echo %PATH%` (CMD)
3. **Verify installation**: `pip show style-transfer-ai`
4. **Run setup script again** if needed

### Permission Issues:
- Run PowerShell/Command Prompt **as Administrator**
- Or use the manual command method

### Different Python Version:
- Update scripts to match your Python version (e.g., `Python312`, `Python311`)
- Check with: `python --version`

## 📦 Complete Installation Flow

```bash
# 1. Install package
pip install style-transfer-ai

# 2. Setup PATH (choose one method)
PowerShell -ExecutionPolicy Bypass -File "add_to_path.ps1"
# OR
setup_path.bat

# 3. Refresh environment
refreshenv
# OR restart terminal

# 4. Use globally
style-transfer-ai
```

Now Style Transfer AI is ready for global use! 🎉