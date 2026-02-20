# 🛠️ **Installation Guide - For Your Friend**

## 📥 **Download & Install (Step-by-Step)**

Hey! Here's how to get Style Transfer AI running on your machine:

### **Prerequisites**
- **Python 3.7+** ([Download here](https://python.org/downloads/))
- **Git** ([Download here](https://git-scm.com/downloads))

---

## 🚀 **Method 1: PyPI Install (EASIEST - NEW!)**

### **1. Install from PyPI**
```bash
# One command installation
pip install style-transfer-ai
```

### **2. Setup Global CLI Access (Windows)**
```bash
# Download PATH setup script
curl -O https://raw.githubusercontent.com/alwynrejicser/style-transfer-ai/main/add_to_path.ps1

# Run automatic PATH setup
PowerShell -ExecutionPolicy Bypass -File "add_to_path.ps1"

# OR use our batch file
curl -O https://raw.githubusercontent.com/alwynrejicser/style-transfer-ai/main/setup_path.bat
setup_path.bat
```

### **3. Use Globally**
```bash
# Now works from any directory!
style-transfer-ai

# With arguments
style-transfer-ai --analyze your-text.txt
```

---

## 🚀 **Method 2: Simple Install (RECOMMENDED for Development)**

### **1. Clone the Repository**
```bash
git clone https://github.com/alwynrejicser/style-transfer-ai.git
cd style-transfer-ai
```

### **2. Install Dependencies Only (Safest Method)**
```bash
# Install just the required packages
pip install requests

# Optional: Install all features
pip install requests openai google-generativeai
```

### **3. Run the Application**
```bash
# This always works
python scripts/run.py
```

### **4. Test It Works**
If you see the main menu, you're ready! 🎉

---

## 🛠️ **Method 2: Advanced Install (Optional)**

### **1. Clone Repository**
```bash
git clone https://github.com/alwynrejicser/style-transfer-ai.git
cd style-transfer-ai
```

### **2. Create Virtual Environment (Recommended)**
```bash
python -m venv .venv

# Activate it:
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### **3. Install Dependencies**
```bash
# Basic method (always works)
pip install requests

# Or install from requirements file
pip install -r install/requirements.txt

# Only if basic method works, try package install:
pip install -e .
```

### **4. Run the Application**
```bash
# Method 1: Direct run (always works)
python scripts/run.py

# Method 2: CLI command (only if pip install -e . worked)
style-transfer-ai
```

---

## 🤖 **AI Model Setup (Choose One)**

### **Option A: Local Models (Free, Private, Recommended)**

1. **Install Ollama**:
   - Go to [ollama.ai](https://ollama.ai/download)
   - Download and install for your OS

2. **Pull Models**:
   ```bash
   # Fast, lightweight model (recommended for testing)
   ollama pull gemma2:2b
   
   # Better quality model (if you have more RAM)
   ollama pull llama3.1:8b
   ```

3. **Start Ollama Server**:
   ```bash
   ollama serve
   ```

### **Option B: Cloud APIs (Paid)**

1. **Get API Keys**:
   - **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - **Gemini**: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

2. **Configure Environment**:
   ```bash
   # Copy template
   cp .env.example .env
   
   # Edit .env and add your API keys
   ```

---

## 🧪 **Test Your Installation**

### **Quick Test**
```bash
# Run the application
python scripts/run.py

# Or if installed as package:
style-transfer-ai
```

You should see a menu like:
```
=====================================================
STYLE TRANSFER AI - ADVANCED STYLOMETRY ANALYSIS
=====================================================
Current Model: None (will be selected before analysis)
-----------------------------------------------------
STYLE ANALYSIS:
1. Analyze Writing Style (Complete Analysis)
2. Quick Style Analysis (Statistical Only)
...
```

### **Test with Sample Files**
The app comes with sample text files in `data/samples/` folder - perfect for testing!

---

## 🔧 **Troubleshooting**

### **Your Friend's Error: "Not a valid editable requirement"**

This happens when pip can't install the package. **Solution:**

```bash
# Don't use pip install -e .
# Instead, just install dependencies:
pip install requests

# Then run directly:
python scripts/run.py
```

### **Other Common Issues & Solutions**

**❌ "Python not found"**
- Install Python from [python.org](https://python.org/downloads/)
- Make sure to check "Add to PATH" during installation

**❌ "pip not found"**
- Python 3.7+ includes pip automatically
- Try: `python -m pip install requests` instead of `pip install requests`

**❌ "Module not found"**
- Make sure you're in the correct directory (`cd style-transfer-ai`)
- Use `python scripts/run.py` instead of `style-transfer-ai`

**❌ "No such file or directory"**
- Make sure you cloned the repository correctly
- Check you're in the right folder: `ls` (Linux/Mac) or `dir` (Windows)

**❌ "Ollama connection failed"**
- This is OK! The app works without Ollama
- Choose "Statistical Analysis" or install Ollama later

---

## 📁 **Project Structure (What You Downloaded)**

```
style-transfer-ai/
├── src/                     # Main application code
├── install/                 # Installation scripts
├── data/                    # Data files
│   └── samples/            # Sample text files for testing
├── docs/                    # Detailed guides
├── config/                  # Configuration templates
├── scripts/                 # Entry points and helpers
│   └── run.py               # Quick start script
├── setup.py                 # Package installation
└── README.md               # Project overview
```

---

## 🎯 **You're Ready!**

Once installed, you can:
- ✅ Analyze your writing style with 25-point framework
- ✅ Generate content in your style
- ✅ Transfer writing to different styles
- ✅ Save results locally
- ✅ Work with virtually any text file format

**Need help?** Check the `docs/` folder for detailed guides!

---

## 🔗 **Quick Links**

- **Main Repository**: https://github.com/alwynrejicser/style-transfer-ai
- **Issues/Support**: https://github.com/alwynrejicser/style-transfer-ai/issues
- **Ollama Download**: https://ollama.ai/download
- **OpenAI API**: https://platform.openai.com/api-keys
- **Gemini API**: https://makersuite.google.com/app/apikey

**Enjoy analyzing your writing style!** 🎨📝