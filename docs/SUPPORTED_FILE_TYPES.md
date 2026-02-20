# File Types Supported by Style Transfer AI

## 📄 **Supported File Types**

Style Transfer AI can read **any text-based file** that contains readable text content. The system is designed to be **file format agnostic** and focuses on processing text content rather than specific file types.

## ✅ **Confirmed Supported Formats**

### **Plain Text Files**
- **`.txt`** - Primary recommended format
- **`.text`** - Alternative plain text extension
- **No extension** - Raw text files without extensions

### **Markup and Documentation**
- **`.md`** - Markdown files
- **`.markdown`** - Markdown files (alternative extension)
- **`.rst`** - reStructuredText files
- **`.html`** - HTML files (text content will be processed)
- **`.xml`** - XML files (text content will be processed)

### **Programming and Config Files**
- **`.py`** - Python source files
- **`.js`** - JavaScript files
- **`.java`** - Java source files
- **`.cpp`** - C++ source files
- **`.c`** - C source files
- **`.cs`** - C# source files
- **`.php`** - PHP files
- **`.json`** - JSON files
- **`.yaml`** - YAML files
- **`.yml`** - YAML files
- **`.ini`** - Configuration files
- **`.cfg`** - Configuration files
- **`.log`** - Log files

### **Web and Data Files**
- **`.csv`** - CSV files (text content)
- **`.tsv`** - Tab-separated files
- **`.sql`** - SQL script files
- **`.sh`** - Shell script files
- **`.bat`** - Batch files
- **`.ps1`** - PowerShell script files

## 🔧 **How File Reading Works**

### **Text Processing Pipeline**
```python
def read_text_file(file_path):
    for encoding in SUPPORTED_ENCODINGS:  # ['utf-8', 'latin-1']
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                return content
        except UnicodeDecodeError:
            continue  # Try next encoding
```

### **Encoding Support**
- **Primary**: `UTF-8` (Universal Unicode)
- **Fallback**: `Latin-1` (ISO-8859-1)
- **Automatic Detection**: System tries UTF-8 first, falls back to Latin-1 if needed

### **File Validation**
- **Existence Check**: File must exist and be accessible
- **Readability Check**: File must have read permissions
- **Content Check**: File must not be empty
- **No Extension Filtering**: System doesn't filter by file extension

## 🚫 **Unsupported File Types**

### **Binary Formats**
- **`.pdf`** - PDF files (binary format)
- **`.doc`** - Microsoft Word (binary format)
- **`.docx`** - Microsoft Word (compressed binary)
- **`.rtf`** - Rich Text Format (requires special parsing)
- **`.odt`** - OpenDocument Text (compressed format)

### **Image and Media Files**
- **`.jpg/.jpeg`** - Image files
- **`.png`** - Image files
- **`.gif`** - Image files
- **`.mp3`** - Audio files
- **`.mp4`** - Video files
- **`.wav`** - Audio files

### **Compressed Archives**
- **`.zip`** - Compressed archives
- **`.rar`** - Compressed archives
- **`.tar`** - Archive files
- **`.gz`** - Compressed files

## 💡 **Best Practices for File Input**

### **Recommended File Types**
1. **`.txt`** - Best compatibility and performance
2. **`.md`** - Good for formatted text with minimal markup
3. **Programming files** - Great for analyzing coding style

### **File Size Considerations**
- **Minimum**: 500+ words recommended for meaningful analysis
- **Maximum**: No hard limit, but very large files may take longer to process
- **Optimal**: 1,000-10,000 words per file for best analysis quality

### **Content Quality Guidelines**
- **Pure Text Content**: Best results with text-only content
- **Minimal Formatting**: Less markup/code = better stylometric analysis
- **Coherent Writing**: Continuous prose works better than fragmented text
- **Single Author**: Files should contain writing from one author only

## 🔍 **File Validation Process**

### **Automatic Checks**
1. **File Exists**: Verifies file path is valid
2. **File Readable**: Checks read permissions
3. **File Not Empty**: Ensures file contains content
4. **Encoding Compatible**: Tests UTF-8 and Latin-1 encoding
5. **Text Content**: Verifies file contains readable text

### **Error Handling**
```
✓ Added: my_writing_sample.txt
✗ File not found: nonexistent.txt
✗ File is empty: empty_file.txt
✗ Cannot read file with any supported encoding
```

## 📋 **Usage Examples**

### **Style Analysis Input**
```
Enter file paths (one per line, empty line to finish):
File path: my_essays/personal_essay.txt
✓ Added: personal_essay.txt
File path: writing_samples/article_draft.md
✓ Added: article_draft.md  
File path: code_comments/documentation.py
✓ Added: documentation.py
File path: [empty line to finish]
```

### **Content Generation Sources**
- **Personal Writing**: Essays, journal entries, blog posts
- **Professional Writing**: Reports, emails, documentation
- **Creative Writing**: Stories, poems, scripts
- **Academic Writing**: Papers, research notes, thesis drafts
- **Technical Writing**: Documentation, code comments, tutorials

### **Style Transfer Sources**
- **Any readable text file** can be used as source content
- **Various formats** can be mixed in a single analysis
- **Consistent encoding** across files is automatically handled

## ⚠️ **Important Notes**

### **File Format Independence**
- **Extension Agnostic**: System reads ANY file as text if possible
- **Content Focus**: Analyzes text content regardless of original format
- **No Format Parsing**: Doesn't parse format-specific elements (HTML tags, Markdown syntax, etc.)

### **Text Extraction**
- **Raw Text Only**: Extracts and analyzes only the text content
- **No Formatting Preservation**: Ignores fonts, colors, layouts
- **Structure Ignored**: Doesn't preserve document structure or hierarchy

### **Multiple File Support**
- **Batch Processing**: Can analyze multiple files simultaneously
- **Mixed Formats**: Can combine different file types in one analysis
- **Encoding Consistency**: Automatically handles different encodings

## 🎯 **Summary**

**Style Transfer AI can read virtually ANY text-based file format** including:

- ✅ **Plain text files** (`.txt`, `.text`)
- ✅ **Markdown files** (`.md`, `.markdown`)
- ✅ **Source code files** (`.py`, `.js`, `.java`, `.cpp`, etc.)
- ✅ **Configuration files** (`.json`, `.yaml`, `.ini`, etc.)
- ✅ **Web files** (`.html`, `.xml`, `.css`)
- ✅ **Data files** (`.csv`, `.sql`, `.log`)
- ✅ **Script files** (`.sh`, `.bat`, `.ps1`)
- ✅ **Any file containing readable text**

**The key requirement is that the file contains TEXT CONTENT that can be read with UTF-8 or Latin-1 encoding.**

For best results, use **plain text files (`.txt`)** or **markdown files (`.md`)** with coherent, well-written content of at least 500 words.