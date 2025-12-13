# Style Transfer AI - Enhanced Deep Stylometry Analyzer v1.1.0

🎯  Advanced stylometry analysis system with personalized linguistic fingerprinting and privacy-first local processing**

![State-Level Buildathon Qualification](assets/image.png)

## 🚀 Quick Start

### Method 1: One-Line Installation (Recommended)
```powershell
# Complete installation + PATH setup (PowerShell)
iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/alwynrejicser/style-transfer-ai/main/install_one_line.ps1'))
```

### Method 2: Standard PyPI Installation
```bash
# Install from PyPI
pip install style-transfer-ai

# Add to PATH (Windows PowerShell)
$p="$env:APPDATA\Python\Python313\Scripts";$c=[Environment]::GetEnvironmentVariable("PATH","User");if($c -notlike "*$p*"){[Environment]::SetEnvironmentVariable("PATH","$c;$p","User");Write-Host "✅ PATH configured! Restart terminal."}else{Write-Host "✅ Already configured!"}

# Run globally
style-transfer-ai
```

### Method 3: Development Installation
```bash
# Clone and run directly
git clone https://github.com/alwynrejicser/style-transfer-ai.git
cd style-transfer-ai
pip install requests
python run.py            # CLI mode
python run_gui.py        # Desktop GUI
```

**📋 Quick Setup Notes:**
- **No dependencies required** - Package installs everything automatically
- **Local processing** - Works offline with Ollama models (optional)
- **Privacy-first** - No cloud dependencies, no Firebase
- **Global CLI** - Use `style-transfer-ai` from anywhere after installation

## Features

✅ **🔒 Privacy-First Architecture**:
- **Local processing**: Complete analysis without internet (Ollama models)
- **No cloud dependencies**: Firebase completely removed from v1.1.0
- **Zero data sharing**: Your text never leaves your machine
- **Optional cloud models**: OpenAI/Gemini support when needed

✅ **🏗️ Modular Architecture**:
- **Clean separation**: Feature-based modules for maintainability
- **Scalable design**: Easy to extend with new models or features
- **Professional structure**: Industry-standard Python package organization
- **PyPI distribution**: Simple `pip install style-transfer-ai` installation

✅ **Personalized Stylometric Fingerprints**:
- **Name-based file organization**: Files saved as `{name}_stylometric_profile_{timestamp}`
- **Personal identity integration**: Your name prominently displayed in analysis
- **Stylometric fingerprint concept**: Treats writing analysis as unique personal identifiers
- **Safe filename handling**: Automatic sanitization for filesystem compatibility

✅ **Performance Optimization**:
- **Multi-model support**: Local Ollama + Cloud APIs (OpenAI, Gemini)
- **Intelligent processing**: Statistical-only or full deep analysis modes
- **Resource-aware processing**: Optimized for different analysis depths
- **One-line installation**: Complete setup with single PowerShell command

✅ **Hierarchical Model Selection**:
- **Local Processing**: Ollama models (privacy-first, free)
- **Cloud Processing**: OpenAI GPT-3.5-turbo, Google Gemini-1.5-flash
- **Automatic fallback**: Graceful degradation when models unavailable
- **Intuitive navigation**: Main menu → Sub-menus with back navigation
- **Professional interface**: Clean, emoji-free design for serious analysis

✅ **🖥️ Desktop GUI (CustomTkinter)**:
- **Multi-view UI**: Dashboard, Generation Studio, Profiles Hub, Settings
- **Live charts**: Radar visualization + readability bars
- **Non-blocking**: Background threads keep UI responsive during model calls
- **One-click runs**: `python run_gui.py` launches the full experience

✅ **Enhanced Deep Analysis**:
- **25-point stylometric framework** (upgraded from 15-point)
- **Dual output formats**: JSON (machine-readable) + TXT (human-readable)
- **Advanced metrics**: Readability scores, lexical diversity, psychological profiling
- **Statistical analysis**: Word frequencies, punctuation patterns, complexity indices
- **Local storage**: Secure local file storage with timestamped profiles
- Individual file analysis + consolidated profiling

✅ **Flexible Input Options**:
- **File-based analysis**: Traditional text file processing
- **Custom text input**: Direct text entry without file management
- **Sample file support**: Built-in test files for quick evaluation
- **Smart validation**: Automatic text length checking and user guidance

✅ **Privacy & Flexibility**:
- Local processing for confidential content
- Multiple cloud API options (OpenAI, Gemini)
- Flexible API key management
- No data sharing with local models
- Production-ready architecture

## Quick Start

### 1. Install Style Transfer AI

**Option A: One-Line Complete Setup (Recommended)**
```powershell
# PowerShell one-liner - installs everything + configures PATH
iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/alwynrejicser/style-transfer-ai/main/install_one_line.ps1'))
```

**Option B: Standard Installation**
```bash
# Install from PyPI
pip install style-transfer-ai

# Configure PATH (Windows) - restart terminal after this
$p="$env:APPDATA\Python\Python313\Scripts";$c=[Environment]::GetEnvironmentVariable("PATH","User");if($c -notlike "*$p*"){[Environment]::SetEnvironmentVariable("PATH","$c;$p","User");Write-Host "✅ PATH configured! Restart terminal."}else{Write-Host "✅ Already configured!"}
```

### 2. Optional: Local Models for Privacy (Recommended)

#### For Local Processing (Privacy-First)
```bash
# Install Ollama from https://ollama.ai/download
# Then pull the models:
ollama pull gpt-oss:20b      # Advanced model
ollama pull gemma3:1b        # Fast model

# Start Ollama server
ollama serve
```

### 3. Optional: Cloud APIs (If Needed)

#### OpenAI API (Optional)
- Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- Enter when prompted by the application

#### Google Gemini API (Optional)
- Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Enter when prompted by the application

### 4. Run Analysis
```bash
# Run from anywhere (after PATH setup)
style-transfer-ai

# Or in development mode
python run.py
python run_gui.py  # Launch the desktop GUI
```

**🎯 No additional dependencies required!** The package automatically installs all necessary components.

## CLI Installation & Usage

### One-Line Installation (Recommended)

**Complete Setup:**
```powershell
# PowerShell - installs package + configures PATH automatically
iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/alwynrejicser/style-transfer-ai/main/install_one_line.ps1'))
```

**PATH-Only Setup (after manual pip install):**
```powershell
# If you already ran: pip install style-transfer-ai
$p="$env:APPDATA\Python\Python313\Scripts";$c=[Environment]::GetEnvironmentVariable("PATH","User");if($c -notlike "*$p*"){[Environment]::SetEnvironmentVariable("PATH","$c;$p","User");Write-Host "✅ PATH configured! Restart terminal."}else{Write-Host "✅ Already configured!"}
```

### Manual Installation

**Standard PyPI Installation:**
```bash
# Install the package
pip install style-transfer-ai

# For global access, restart terminal after PATH setup above
style-transfer-ai
```

**Development Installation:**
```bash
# From project root directory
pip install -e .

# Use globally
style-transfer-ai
```

**Post-Installation:**
- ✅ Restart command prompt/terminal for PATH changes
- ✅ Use `style-transfer-ai` from any directory
- ✅ No additional dependencies needed
- ✅ Local processing ready (add Ollama models for privacy)

### CLI Usage Examples

#### Interactive Mode (Default)
```bash
# Run interactive menu (same as python style_analyzer_enhanced.py)
style-transfer-ai
style-transfer-ai --interactive
```

#### Batch Analysis
```bash
# Analyze single file
style-transfer-ai --analyze sample.txt

# Analyze multiple files
style-transfer-ai --analyze file1.txt file2.txt file3.txt

# Analyze with specific model
style-transfer-ai --analyze sample.txt --model gpt-oss:20b

# Force local processing
style-transfer-ai --analyze sample.txt --local

# Force cloud processing
style-transfer-ai --analyze sample.txt --cloud
```

#### Custom Output
```bash
# Custom output filename base
style-transfer-ai --analyze sample.txt --output "my_analysis"
```

### CLI Options Reference

| Option | Description | Example |
|--------|-------------|---------|
| `--interactive` | Run in interactive menu mode (default) | `style-transfer-ai --interactive` |
| `--analyze FILE [FILE ...]` | Analyze one or more text files | `style-transfer-ai --analyze text1.txt text2.txt` |
| `--model MODEL` | Specify model (gpt-oss:20b, gemma3:1b, openai, gemini) | `style-transfer-ai --analyze file.txt --model gemma3:1b` |
| `--local` | Force use of local Ollama models | `style-transfer-ai --analyze file.txt --local` |
| `--cloud` | Force use of cloud models (OpenAI/Gemini) | `style-transfer-ai --analyze file.txt --cloud` |
| `--output NAME` | Base name for output files (no extension) | `style-transfer-ai --analyze file.txt --output my_profile` |
| `--help` | Show help message and exit | `style-transfer-ai --help` |

## GUI Usage (Desktop)

```bash
python run_gui.py
```

- Choose a model (Ollama local by default) and toggle Turbo if you want faster statistical-only runs
- Load a `.txt` file or paste text, then click **Analyze Text** for deep stylometry
- View radar chart + readability bars, and save outputs from the dashboard
- Use **Generation Studio** to load a saved profile and generate on-brand content
- Manage saved profiles in **Profiles**, set API keys / cleanup in **Settings**

### Advanced CLI Workflows

#### Research Pipeline
```bash
# Analyze academic papers with cloud processing
style-transfer-ai --analyze paper1.txt paper2.txt --cloud --output "academic_style"

# Quick analysis with local fast model
style-transfer-ai --analyze draft.txt --local --model gemma3:1b
```

#### Batch Processing
```bash
# Analyze all text files in current directory (Windows PowerShell)
Get-ChildItem *.txt | ForEach-Object { style-transfer-ai --analyze $_.Name --output $_.BaseName }

# Analyze with privacy mode (local only)
style-transfer-ai --analyze sensitive.txt --local
```

#### Development & Testing
```bash
# Test different models on same content
style-transfer-ai --analyze test.txt --model gpt-oss:20b --output "test_advanced"
style-transfer-ai --analyze test.txt --model gemma3:1b --output "test_fast"
```

### CLI vs Interactive Mode

| Feature | CLI Mode | Interactive Mode |
|---------|----------|------------------|
| **Speed** | Fast, direct analysis | Menu navigation required |
| **Automation** | Perfect for scripts/batch | Manual operation |
| **Customization** | Command-line arguments | Interactive prompts |
| **User Experience** | Technical users | User-friendly menus |
| **Integration** | CI/CD, automation tools | Stand-alone usage |

### Troubleshooting CLI

**Command not found after installation:**
```bash
# Verify installation
pip list | grep style-transfer-ai

# Reinstall in development mode
pip uninstall style-transfer-ai
pip install -e .
```

**Permission errors:**
```bash
# Use user installation
pip install --user -e .

# Or with elevated permissions (Windows)
# Run PowerShell as Administrator, then install
```

**Path issues:**
```bash
# Check if Python Scripts directory is in PATH
python -m site --user-base

# Add to PATH if needed (Windows PowerShell)
$env:PATH += ";$(python -m site --user-base)\Scripts"
```

## Key Workflow

### Personal Profile Setup
1. **Name Collection**: Your name is collected first as stylometric profiles are personal fingerprints
2. **Cultural Context**: Language background, nationality, education details
3. **Writing Experience**: Frequency and background information
4. **Streamlined Process**: Only essential information collected for meaningful analysis

### Processing Modes (GPT-OSS)
When using GPT-OSS models, choose your processing mode:
- **Turbo Mode**: Faster analysis (2000 tokens, 120s timeout) for quick insights
- **Normal Mode**: Thorough analysis (3000 tokens, 180s timeout) for comprehensive profiling

## Usage

### Model Selection
The analyzer features a **hierarchical menu system** for intuitive model selection:

**Main Menu:**
1. **Local Processing** (Privacy-focused)
2. **Online Processing** (Cloud-based)

**Local Models Sub-menu:**
- **GPT-OSS 20B** - Advanced comprehensive analysis
- **Gemma 3:1B** - Fast efficient processing

**Online Models Sub-menu:**
- **OpenAI GPT-3.5-turbo** - Cloud-based analysis
- **Google Gemini-1.5-flash** - Google's latest language model

**Navigation:** Use '0' to go back to the previous menu or exit the application.

### Input Options
Choose from **three flexible input methods** for text analysis:

**1. Sample Files (Recommended for Testing)**
- Use built-in test files for immediate evaluation
- Perfect for first-time users and feature testing
- Pre-validated content ensures reliable analysis

**2. Custom File Paths**
- Specify your own text files for analysis
- Supports multiple files for comprehensive profiling
- Automatic file validation and error handling

**3. Direct Text Input (NEW)**
- **No files needed**: Enter text directly into the application
- **Copy & paste**: Analyze content from any source (emails, documents, web pages)
- **Flexible length**: From single paragraphs to full documents
- **Smart validation**: Automatic length checking and user guidance
- **Perfect for**: Quick analysis without file management

**Example Direct Text Usage:**
```
Options:
1. Use sample files (recommended for testing)
2. Specify your own file paths  
3. Enter custom text directly (no files needed)

Enter your choice (1-3): 3

Enter your text (press Enter twice to finish):
This is my writing sample for analysis...
[Continue entering text]
[Press Enter twice to complete]

✓ Text captured: 125 words, 650 characters
```

### Enhanced Output
The analyzer generates **personalized stylometric fingerprints**:
- **Individual analyses** for each text file with 25-point deep analysis
- **Consolidated style profile** combining all samples
- **Personalized file naming**: `{your_name}_stylometric_profile_{timestamp}`
  - Example: `John_Doe_stylometric_profile_20250915_123456.json`
- **Dual format outputs**:
  - **JSON file** - Machine-readable data for further processing
  - **TXT file** - Human-readable report with your name prominently displayed
- **Writer Identity Section**: Your name featured prominently in analysis header
- **Advanced metrics**: Readability scores, lexical diversity, statistical analysis
- **Comprehensive metadata** with timestamps and detailed file information
- **Automatic cleanup**: Option to remove old reports with both naming patterns

## API Key Configuration

### Method 1: Direct Code Modification
Replace the placeholders in `style_analyzer_enhanced.py`:

```python
OPENAI_API_KEY = "your-actual-openai-api-key-here"
GEMINI_API_KEY = "your-actual-gemini-api-key-here"
```

### Method 2: Interactive Setup (Recommended)
The analyzer will automatically:
1. Detect existing API keys for both OpenAI and Gemini
2. Prompt for new keys if needed based on your model choice
3. Validate key format and provide helpful URLs
4. Handle setup cancellation gracefully
5. Support switching between different API providers

## Project Structure

```
style-transfer-ai/
├── src/                                 # Main package source
│   ├── main.py                         # CLI entry point
│   ├── analysis/                       # Analysis modules
│   ├── models/                         # AI model clients
│   ├── menu/                          # Interactive menu system
│   ├── config/                        # Configuration management
│   ├── storage/                       # Local storage only
│   ├── utils/                         # Utility functions
│   └── gui/                           # CustomTkinter desktop UI
├── install/                           # Installation scripts
│   ├── install_cli.bat               # Windows batch installer
│   ├── quick_install.bat             # Quick setup
│   ├── requirements.txt              # Dependencies
│   └── setup.py                      # Package configuration
├── install_one_line.ps1              # One-line PowerShell installer
├── path_one_line.txt                 # PATH-only setup command
├── style_analyzer_enhanced.py        # Legacy analyzer (still functional)
├── run.py                            # Development CLI entry point
├── run_gui.py                        # Desktop GUI entry point
├── setup.py                          # Main package setup
├── README.md                         # This file
├── default text/                     # Sample text files
│   ├── about_my_pet.txt             # Sample analysis file
│   └── about_my_pet_1.txt           # Additional samples
├── documentation/                    # Technical documentation
└── {name}_stylometric_profile_*.json # Your personalized analysis output
└── {name}_stylometric_profile_*.txt  # Human-readable analysis output
```

**Key Changes in v1.1.0:**
- ✅ **Firebase completely removed** - No cloud storage dependencies
- ✅ **Local storage only** - All data stays on your machine
- ✅ **One-line installers** - Simplified deployment
- ✅ **Clean PyPI package** - No unnecessary dependencies

## What's New

🆕 **Personalized Stylometric Fingerprints**:
- Your name is now collected and used for file naming
- Files saved as `{Name}_stylometric_profile_{timestamp}`
- Writer identity prominently displayed in analysis
- Automatic filename sanitization for safe storage

⚡ **GPT-OSS Performance Modes**:
- **Turbo Mode**: Quick analysis with optimized parameters
- **Normal Mode**: Comprehensive deep analysis
- User choice for processing speed vs thoroughness

🏗️ **Enhanced Architecture**:
- Streamlined user profile collection (8 essential fields)
- Professional interface without emoji clutter
- Improved error handling and connection validation
- Updated cleanup functionality for all naming patterns

## Enhanced Stylometric Analysis Framework

The analyzer evaluates **25 comprehensive dimensions** across 7 categories:

### Part 1: Linguistic Architecture
1. **Sentence Structure Mastery**: Exact averages, complexity ratios with percentages
2. **Clause Choreography**: Subordinate clause frequency, coordination patterns
3. **Punctuation Symphony**: Complete punctuation analysis with frequencies
4. **Syntactic Sophistication**: Sentence variety index, grammatical complexity scoring

### Part 2: Lexical Intelligence  
5. **Vocabulary Sophistication**: Word complexity levels, formal/informal ratios
6. **Semantic Field Preferences**: Domain categorization (abstract/concrete, emotional/logical)
7. **Lexical Diversity Metrics**: Type-token ratio, vocabulary richness index
8. **Register Flexibility**: Formality spectrum analysis, colloquialism detection

### Part 3: Stylistic DNA
9. **Tone Architecture**: Confidence indicators, emotional markers with examples
10. **Voice Consistency**: Person preference analysis, active/passive voice ratios
11. **Rhetorical Weaponry**: Metaphor counting, parallel structures, repetition patterns
12. **Narrative Technique**: Point of view consistency, storytelling vs explanatory modes

### Part 4: Cognitive Patterns
13. **Logical Flow Design**: Argument structure, cause-effect pattern analysis
14. **Transition Mastery**: Transition word categorization, coherence mechanisms
15. **Emphasis Engineering**: Key point highlighting strategies, linguistic intensity
16. **Information Density**: Concept-to-word ratios, information packaging efficiency

### Part 5: Psychological Markers
17. **Cognitive Processing Style**: Linear vs circular thinking, analytical patterns
18. **Emotional Intelligence**: Empathy markers, emotional vocabulary richness
19. **Authority Positioning**: Hedging language, assertiveness markers, expertise indicators
20. **Risk Tolerance**: Certainty language analysis, qualification usage patterns

### Part 6: Structural Genius
21. **Paragraph Architecture**: Length variance, topic development patterns
22. **Coherence Engineering**: Text cohesion measurement, referential chains
23. **Temporal Dynamics**: Tense usage patterns, time reference preferences
24. **Modal Expression**: Modal verb counting, probability vs obligation language

### Part 7: Unique Fingerprint
25. **Personal Signature Elements**: Unique phrases, idiosyncratic expressions, personal habits

### Stylistic Markers
7. **Tone Indicators**: Confidence, emotional markers, certainty
8. **Narrative Voice**: Person preference, active/passive ratio
9. **Rhetorical Devices**: Metaphors, repetition, parallel structure

### Structural Preferences
10. **Paragraph Organization**: Length, transition methods
11. **Flow Patterns**: Idea connection, progression style
12. **Emphasis Techniques**: Highlighting methods

### Personal Markers
13. **Idiomatic Expressions**: Unique phrases, expressions
14. **Cultural References**: Reference types and patterns
15. **Formality Range**: Casual to formal adaptability

## Security & Best Practices

🔐 **API Key Security**:
- Never commit real API keys to version control
- Use environment variables for production
- Rotate keys regularly
- Monitor usage and costs

🛡️ **Privacy**:
- Use local models for sensitive content
- Local processing keeps data on your machine
- No internet connection required for Ollama models

## Troubleshooting

### Common Issues

**Ollama Connection Failed**:
```bash
# Make sure Ollama is running
ollama serve

# Check if models are installed
ollama list

# Pull missing models
ollama pull gpt-oss:20b
ollama pull gemma3:1b
```

**GPT-OSS Performance Issues**:
- Try **Turbo Mode** for faster processing
- Switch to **Normal Mode** for detailed analysis
- Check system resources during processing
- Verify model is fully loaded in Ollama

**OpenAI API Errors**:
- Check API key validity
- Verify account has sufficient credits
- Ensure proper key format (starts with 'sk-')

**File Naming Issues**:
- Special characters in names are automatically sanitized
- Long names are truncated to 50 characters
- Empty names default to "Anonymous_User"
- Files are timestamped for unique identification

**Configuration Checker Error**:
```
Error: Configuration checker not found at /usr/local/lib/python3.12/dist-packages/check_config.py
```
This has been fixed in v1.1.0+ with integrated checking:
```bash
# Update to latest version
pip install --upgrade --force-reinstall style-transfer-ai

# Or fresh installation
pip uninstall style-transfer-ai && pip install style-transfer-ai
```

**File Not Found**:
- Verify text files exist in project directory
- Check file names match exactly
- Ensure proper file encoding (UTF-8)

## Contributing

Contributions welcome! Please ensure:
- No real API keys in commits
- Update documentation for new features
- Test with all model types and processing modes
- Follow existing code style and naming conventions
- Include personalization features in new developments

## Version History

- **v1.1.0** (Current): **Firebase-Free Local Edition**
  - ✅ Complete Firebase removal for privacy-first architecture
  - ✅ PyPI distribution with simplified installation
  - ✅ One-line PowerShell installers for Windows
  - ✅ Automatic PATH configuration
  - ✅ Local-only storage with no cloud dependencies
  - ✅ Clean package structure without unnecessary dependencies

- **v1.0.0**: **Enhanced Production Edition**
  - ✅ Personalized stylometric fingerprints, GPT-OSS performance modes
  - ✅ Enhanced deep analysis framework (25-point system)
  - ✅ Hierarchical model selection with local/online options
  - ✅ Dual output formats (JSON + TXT)
  - ✅ Modular architecture with professional CLI

## License

MIT License
