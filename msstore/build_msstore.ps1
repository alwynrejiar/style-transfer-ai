<#
.SYNOPSIS
    Build Stylomex for Microsoft Store distribution.

.DESCRIPTION
    This script:
      1. Generates placeholder icons (or uses existing ones)
      2. Builds the PyInstaller .exe bundle
      3. Packages the result into an MSIX for Store submission

.PARAMETER SkipIcons
    Skip icon generation if they already exist.

.PARAMETER SkipBuild
    Skip PyInstaller build (useful when iterating on MSIX packaging only).

.PARAMETER SignForSideload
    Sign the MSIX with a self-signed certificate for local testing.
    Not needed for MS Store submission (Microsoft signs it).

.EXAMPLE
    .\msstore\build_msstore.ps1
    .\msstore\build_msstore.ps1 -SkipIcons -SignForSideload
#>

param(
    [switch]$SkipIcons,
    [switch]$SkipBuild,
    [switch]$SignForSideload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MsStoreDir  = Join-Path $ProjectRoot "msstore"
$DistDir     = Join-Path $ProjectRoot "dist"
$StylomexDir = Join-Path $DistDir "Stylomex"
$MsixOutput  = Join-Path $DistDir "Stylomex.msix"
$MappingFile = Join-Path $MsStoreDir "mapping.txt"

Set-Location $ProjectRoot

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Stylomex — MS Store Build Pipeline"  -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# ───────────────────────────────────────────────
# Step 0: Activate venv if present
# ───────────────────────────────────────────────
$venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "[0/4] Activating virtual environment..." -ForegroundColor Yellow
    & $venvActivate
}

# ───────────────────────────────────────────────
# Step 1: Generate icons
# ───────────────────────────────────────────────
if (-not $SkipIcons) {
    Write-Host "[1/4] Generating icons..." -ForegroundColor Yellow
    $iconScript = Join-Path $MsStoreDir "generate_icons.py"
    python $iconScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Icon generation failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Icons generated successfully." -ForegroundColor Green
} else {
    Write-Host "[1/4] Skipping icon generation." -ForegroundColor DarkGray
}

# Verify icons exist
$requiredIco = Join-Path $MsStoreDir "icons\app.ico"
if (-not (Test-Path $requiredIco)) {
    Write-Host "ERROR: app.ico not found at $requiredIco" -ForegroundColor Red
    Write-Host "Run: python msstore\generate_icons.py" -ForegroundColor Yellow
    exit 1
}

# ───────────────────────────────────────────────
# Step 2: Build with PyInstaller
# ───────────────────────────────────────────────
if (-not $SkipBuild) {
    Write-Host "[2/4] Building with PyInstaller..." -ForegroundColor Yellow

    # Ensure PyInstaller is installed
    python -m pip install pyinstaller --quiet 2>$null

    $specFile = Join-Path $MsStoreDir "stylomex.spec"
    pyinstaller $specFile --noconfirm --clean
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: PyInstaller build failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "  PyInstaller build complete: $StylomexDir" -ForegroundColor Green
} else {
    Write-Host "[2/4] Skipping PyInstaller build." -ForegroundColor DarkGray
}

# Verify exe exists
$exePath = Join-Path $StylomexDir "Stylomex.exe"
if (-not (Test-Path $exePath)) {
    Write-Host "ERROR: Stylomex.exe not found at $exePath" -ForegroundColor Red
    exit 1
}

# ───────────────────────────────────────────────
# Step 3: Prepare MSIX layout & package
# ───────────────────────────────────────────────
Write-Host "[3/4] Creating MSIX package..." -ForegroundColor Yellow

$msixLayoutDir = Join-Path $DistDir "MsixLayout"
if (Test-Path $msixLayoutDir) {
    Remove-Item $msixLayoutDir -Recurse -Force
}
New-Item -ItemType Directory -Path $msixLayoutDir -Force | Out-Null

# Copy PyInstaller output into layout
Write-Host "  Copying application files..."
Copy-Item -Path "$StylomexDir\*" -Destination $msixLayoutDir -Recurse -Force

# Copy AppxManifest.xml
$manifestSrc = Join-Path $MsStoreDir "AppxManifest.xml"
Copy-Item $manifestSrc -Destination $msixLayoutDir -Force

# Copy MSIX icon assets
$assetsDir = Join-Path $msixLayoutDir "Assets"
New-Item -ItemType Directory -Path $assetsDir -Force | Out-Null
$msixIconsDir = Join-Path $MsStoreDir "icons\msix"
if (Test-Path $msixIconsDir) {
    Copy-Item -Path "$msixIconsDir\*" -Destination $assetsDir -Force
}

# Check if MakeAppx is available
$makeAppx = Get-Command "MakeAppx.exe" -ErrorAction SilentlyContinue
if (-not $makeAppx) {
    # Try to find it in Windows SDK
    $sdkPaths = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\makeappx.exe",
        "${env:ProgramFiles}\Windows Kits\10\bin\*\x64\makeappx.exe"
    )
    foreach ($pattern in $sdkPaths) {
        $found = Get-ChildItem $pattern -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
        if ($found) {
            $makeAppx = $found.FullName
            break
        }
    }
}

if ($makeAppx) {
    $makeAppxPath = if ($makeAppx -is [System.Management.Automation.ApplicationInfo]) { $makeAppx.Source } else { $makeAppx }
    Write-Host "  Using MakeAppx: $makeAppxPath"

    # Remove old MSIX if present
    if (Test-Path $MsixOutput) {
        Remove-Item $MsixOutput -Force
    }

    & $makeAppxPath pack /d $msixLayoutDir /p $MsixOutput /o
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: MakeAppx failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "  MSIX created: $MsixOutput" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  WARNING: MakeAppx.exe not found." -ForegroundColor Yellow
    Write-Host "  Install the Windows 10/11 SDK to get MakeAppx." -ForegroundColor Yellow
    Write-Host "  Download: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Once installed, run:" -ForegroundColor Yellow
    Write-Host "    MakeAppx pack /d `"$msixLayoutDir`" /p `"$MsixOutput`" /o" -ForegroundColor White
    Write-Host ""
    Write-Host "  The MSIX layout is ready at: $msixLayoutDir" -ForegroundColor Green
}

# ───────────────────────────────────────────────
# Step 4: Optional signing for sideload testing
# ───────────────────────────────────────────────
if ($SignForSideload -and (Test-Path $MsixOutput)) {
    Write-Host "[4/4] Signing MSIX for sideloading..." -ForegroundColor Yellow

    $signTool = Get-Command "SignTool.exe" -ErrorAction SilentlyContinue
    if (-not $signTool) {
        $sdkSignPaths = @(
            "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe",
            "${env:ProgramFiles}\Windows Kits\10\bin\*\x64\signtool.exe"
        )
        foreach ($pattern in $sdkSignPaths) {
            $found = Get-ChildItem $pattern -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
            if ($found) {
                $signTool = $found.FullName
                break
            }
        }
    }

    if ($signTool) {
        Write-Host "  Creating self-signed certificate for testing..."
        $cert = New-SelfSignedCertificate `
            -Subject "CN=StylomexTestSigning" `
            -Type CodeSigningCert `
            -CertStoreLocation "Cert:\CurrentUser\My" `
            -NotAfter (Get-Date).AddYears(1)
        
        $thumbprint = $cert.Thumbprint
        $signToolPath = if ($signTool -is [System.Management.Automation.ApplicationInfo]) { $signTool.Source } else { $signTool }

        & $signToolPath sign /fd SHA256 /sha1 $thumbprint /t "http://timestamp.digicert.com" $MsixOutput
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  MSIX signed successfully for sideloading." -ForegroundColor Green
            Write-Host "  Certificate thumbprint: $thumbprint" -ForegroundColor DarkGray
        } else {
            Write-Host "  WARNING: Signing failed — MSIX is unsigned." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  WARNING: SignTool.exe not found. Install Windows SDK." -ForegroundColor Yellow
    }
} else {
    Write-Host "[4/4] Skipping signing (not needed for Store submission)." -ForegroundColor DarkGray
}

# ───────────────────────────────────────────────
# Summary
# ───────────────────────────────────────────────
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Outputs:"
Write-Host "    EXE Bundle:  $StylomexDir" -ForegroundColor White
Write-Host "    MSIX Layout: $msixLayoutDir" -ForegroundColor White
if (Test-Path $MsixOutput) {
    $msixSize = [math]::Round((Get-Item $MsixOutput).Length / 1MB, 1)
    Write-Host "    MSIX Package: $MsixOutput ($msixSize MB)" -ForegroundColor White
}
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "    1. Test locally:   Install the MSIX by double-clicking it"
Write-Host "    2. Submit to Store: Upload the MSIX at https://partner.microsoft.com/dashboard"
Write-Host "    3. See full guide:  msstore\MS_STORE_RELEASE_GUIDE.md"
Write-Host ""
