# Microsoft Store Release Guide — Stylomex

This guide walks you through packaging and publishing **Stylomex** (Style Transfer AI) on the Microsoft Store.

---

## Overview

| Component | Tool / Format |
|---|---|
| GUI Framework | CustomTkinter (desktop) |
| Bundler | PyInstaller → standalone `.exe` |
| Package Format | MSIX (required for MS Store) |
| Packaging Tool | MakeAppx.exe (Windows SDK) |
| Submission Portal | [Microsoft Partner Center](https://partner.microsoft.com/dashboard) |

---

## Prerequisites

### 1. Microsoft Partner Center Account
- **Sign up**: https://developer.microsoft.com/en-us/microsoft-store/register/
- **Cost**: One-time **$19 USD** registration fee (individual), or **$99** (company)
- You'll receive a **Publisher ID** (CN=...) after registration

### 2. Windows 10/11 SDK
Required for `MakeAppx.exe` and `SignTool.exe`:
- **Download**: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
- Install the **"Windows App Certification Kit"** and **"Signing Tools"** components
- After install, `MakeAppx.exe` will be at:
  ```
  C:\Program Files (x86)\Windows Kits\10\bin\<version>\x64\makeappx.exe
  ```

### 3. Python Build Dependencies
```powershell
pip install pyinstaller pillow
```

### 4. App Dependencies
Ensure all runtime dependencies are installed in your venv:
```powershell
pip install -r install\requirements.txt
python -m spacy download en_core_web_sm
```

---

## File Structure

```
msstore/
├── AppxManifest.xml        # MSIX package manifest
├── build_msstore.ps1       # Automated build script
├── generate_icons.py       # Icon generator for all required sizes
├── packaging.ini           # Packaging configuration
├── stylomex.spec           # PyInstaller spec file
├── version_info.txt        # Windows version resource
├── MS_STORE_RELEASE_GUIDE.md  # This file
└── icons/
    ├── app.ico             # Multi-resolution .ico (generated)
    └── msix/               # MSIX tile/logo assets (generated)
        ├── Square44x44Logo.png
        ├── Square71x71Logo.png
        ├── Square150x150Logo.png
        ├── Square310x310Logo.png
        ├── Wide310x150Logo.png
        ├── StoreLogo.png
        └── SplashScreen.png
```

---

## Step-by-Step Build Process

### Step 1: Configure Your Publisher Identity

Edit `msstore/AppxManifest.xml` and replace `CN=YourPublisherId` with your actual Publisher ID from Partner Center:

```xml
<Identity
    Name="AlwynRejicser.Stylomex"
    Publisher="CN=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
    ...
```

> You can find your Publisher ID in Partner Center → Account Settings → Organization Profile.

### Step 2: Generate Icons

**Option A — Placeholders (for testing):**
```powershell
python msstore\generate_icons.py
```

**Option B — From your logo:**
```powershell
python msstore\generate_icons.py path\to\your_logo.png
```

The script generates all required MSIX tile sizes + a multi-resolution `.ico`.

### Step 3: Build Everything

Run the automated build script:
```powershell
.\msstore\build_msstore.ps1
```

This will:
1. Generate icons (if not skipping)
2. Bundle the app with PyInstaller → `dist/Stylomex/`
3. Create MSIX layout → `dist/MsixLayout/`
4. Package into `dist/Stylomex.msix` (if Windows SDK is installed)

### Step 4: Test Locally

**Option A — Run the .exe directly:**
```powershell
.\dist\Stylomex\Stylomex.exe
```

**Option B — Install the MSIX (sideload):**
1. Enable Developer Mode: Settings → For Developers → Developer Mode
2. Double-click `dist/Stylomex.msix` to install
3. Find "Stylomex" in Start Menu

**Option C — Sign for sideloading (if MSIX won't install unsigned):**
```powershell
.\msstore\build_msstore.ps1 -SkipBuild -SignForSideload
```

### Step 5: Run Windows App Certification Kit (WACK)

Before submitting, validate your package:
1. Open **Windows App Cert Kit** (installed with the SDK)
2. Select "Validate Store App"
3. Point to your `.msix` file
4. Fix any reported issues

### Step 6: Submit to Microsoft Store

1. Go to https://partner.microsoft.com/dashboard
2. Click **"Create a new app"**
3. Reserve the name **"Stylomex"**
4. Fill in the submission details:

#### Properties
- **Category**: Developer Tools > Development
- **Privacy policy URL**: *(required — host one on your GitHub Pages or website)*
- **Support contact**: your email

#### Store Listing
- **Description**: 
  > Stylomex is an advanced stylometry analysis system that extracts deep linguistic patterns from writing samples. Analyze writing styles with 25-point deep stylometry, generate content in any style, and compare stylometric profiles — all powered by AI with local-first privacy.
- **Screenshots**: Take 1366x768 or 1920x1080 screenshots of the app
- **Search terms**: stylometry, writing analysis, style transfer, NLP, AI writing
- **Short description**: Deep stylometry analysis and AI-powered style transfer

#### Packages
- Upload `dist/Stylomex.msix`
- Architecture: x64
- Min OS: Windows 10 version 1809 (build 17763)

#### Pricing
- Set to **Free** or choose a price tier

5. Click **Submit for certification**

---

## Store Listing Assets Checklist

| Asset | Size | Required |
|---|---|---|
| App icon | 300x300 PNG | Yes |
| Screenshot (desktop) | 1366x768 or 1920x1080 | Yes (min 1) |
| Hero image | 1920x1080 | Recommended |
| Feature graphic | 1024x500 | Recommended |
| Privacy policy URL | — | Yes |

---

## Updating the App

To publish an update:

1. Increment the version in three places:
   - `src/config/settings.py` — `VERSION = "1.4.0"`
   - `msstore/AppxManifest.xml` — `Version="1.4.0.0"`
   - `msstore/version_info.txt` — `filevers` and `prodvers`

2. Rebuild:
   ```powershell
   .\msstore\build_msstore.ps1
   ```

3. Upload the new `.msix` to Partner Center as an update submission.

---

## Troubleshooting

### PyInstaller build fails with missing modules
Add the module to `hiddenimports` in `msstore/stylomex.spec`.

### "MakeAppx.exe not found"
Install the Windows 10/11 SDK or add its `bin` directory to your PATH:
```powershell
$env:PATH += ";C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64"
```

### MSIX won't install (untrusted)
Enable Developer Mode or sign the package:
```powershell
.\msstore\build_msstore.ps1 -SkipBuild -SignForSideload
```

### App crashes on launch after packaging
1. Run the `.exe` from a terminal to see error output:
   ```powershell
   .\dist\Stylomex\Stylomex.exe
   ```
2. Common causes:
   - Missing data files → add to `datas` in the `.spec`
   - Missing hidden imports → add to `hiddenimports`
   - Path issues → use `sys._MEIPASS` for bundled resource paths

### spaCy model not found in packaged app
Ensure `en_core_web_sm` is installed in your venv before building:
```powershell
python -m spacy download en_core_web_sm
```

---

## Privacy Policy

Since the app can connect to AI APIs (Ollama, OpenAI, Gemini), the MS Store **requires** a privacy policy. Create one covering:

- What data is collected (text for analysis, API keys stored locally)
- What data is sent externally (text to AI model APIs by user choice)
- Local-first processing (Ollama runs locally, no data leaves the device by default)
- No telemetry or analytics collected
- User controls over API usage

Host it at a public URL (e.g., GitHub Pages) and link it in your Store submission.

---

## Cost Summary

| Item | Cost |
|---|---|
| Partner Center registration | $19 (one-time, individual) |
| Code signing for MS Store | Free (Microsoft signs during submission) |
| Windows SDK | Free |
| PyInstaller | Free (open source) |
| **Total** | **$19** |
