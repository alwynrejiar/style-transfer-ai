# One-command installation with automatic PATH setup
# Save this as: install_style_transfer_ai.ps1

Write-Host "🚀 Style Transfer AI - Automated Installation" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# Step 1: Install from PyPI
Write-Host "📦 Installing Style Transfer AI from PyPI..." -ForegroundColor Blue
try {
    pip install style-transfer-ai
    Write-Host "✅ Package installed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "❌ Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Step 2: Automatic PATH setup
Write-Host ""
Write-Host "🔧 Setting up global CLI access..." -ForegroundColor Blue

$userScriptsPath = "$env:APPDATA\Python\Python313\Scripts"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

if ($currentPath -like "*$userScriptsPath*") {
    Write-Host "✅ PATH already configured!" -ForegroundColor Green
} else {
    try {
        $newPath = "$currentPath;$userScriptsPath"
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
        Write-Host "✅ PATH configured automatically!" -ForegroundColor Green
    }
    catch {
        Write-Host "⚠️  Could not auto-configure PATH: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Step 3: Test installation
Write-Host ""
Write-Host "🧪 Testing installation..." -ForegroundColor Blue

# Refresh current session PATH
$env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'User') + ';' + [Environment]::GetEnvironmentVariable('PATH', 'Machine')

try {
    $command = Get-Command "style-transfer-ai" -ErrorAction SilentlyContinue
    if ($command) {
        Write-Host "✅ style-transfer-ai CLI is ready!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🎉 Installation Complete!" -ForegroundColor Green
        Write-Host "You can now run: style-transfer-ai" -ForegroundColor White
    } else {
        Write-Host "⚠️  CLI not accessible in current session" -ForegroundColor Yellow
        Write-Host "🔄 Please restart your terminal and try: style-transfer-ai" -ForegroundColor White
    }
}
catch {
    Write-Host "⚠️  Could not test CLI accessibility" -ForegroundColor Yellow
    Write-Host "🔄 Please restart your terminal and try: style-transfer-ai" -ForegroundColor White
}

Write-Host ""
Write-Host "📚 Quick Start:" -ForegroundColor Cyan
Write-Host "  style-transfer-ai                    # Interactive mode" -ForegroundColor White
Write-Host "  style-transfer-ai --analyze file.txt # Analyze a file" -ForegroundColor White
Write-Host "  style-transfer-ai --help             # Show all options" -ForegroundColor White