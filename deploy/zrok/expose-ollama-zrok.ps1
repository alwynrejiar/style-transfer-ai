[CmdletBinding()]
param(
    [string]$OllamaUrl = $(if ($env:OLLAMA_URL) { $env:OLLAMA_URL } else { "http://127.0.0.1:11434" }),
    [string]$BasicUser = $(if ($env:ZROK_BASIC_USER) { $env:ZROK_BASIC_USER } else { "ollama" }),
    [int]$ApiKeyLength = $(if ($env:ZROK_API_KEY_LENGTH) { [int]$env:ZROK_API_KEY_LENGTH } else { 48 }),
    [ValidateSet("auto", "never")]
    [string]$StartOllama = $(if ($env:START_OLLAMA) { $env:START_OLLAMA } else { "auto" })
)

$ErrorActionPreference = "Stop"

$StateRoot = if ($env:OLLAMA_ZROK_STATE_DIR) {
    $env:OLLAMA_ZROK_STATE_DIR
} elseif ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA "ollama-zrok"
} else {
    Join-Path $HOME ".ollama-zrok"
}
$LogDir = Join-Path $StateRoot "logs"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "[ollama-zrok] $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host ""
    Write-Host "WARNING: $Message" -ForegroundColor Yellow
}

function Fail {
    param([string]$Message)
    Write-Host ""
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Get-CommandPath {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { return $null }
    return $cmd.Source
}

function Show-OllamaInstallHelp {
    Write-Host @"

Ollama is not installed or is not on PATH.

Install Ollama:
  Windows:
    Download and run https://ollama.com/download

  After installing, open a new PowerShell window and pull at least one model:
    ollama pull gemma3:1b

"@
}

function Show-ZrokInstallHelp {
    Write-Host @"

zrok is not installed or is not on PATH.

Install zrok:
  1. Download the Windows zrok archive from:
     https://docs.zrok.io/docs/guides/install/windows

  2. Extract zrok.exe into a folder on PATH, for example:
     `$env:USERPROFILE\bin

  3. Open a new PowerShell window and verify:
     zrok version

Then create or open a zrok account, copy your enable token, and rerun this script.

"@
}

function Ensure-StateDirs {
    try {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    } catch {
        Fail "Cannot create state directory '$LogDir'. Set OLLAMA_ZROK_STATE_DIR to a writable folder and rerun."
    }
}

function Test-OllamaApi {
    try {
        $response = Invoke-WebRequest -Uri "$($OllamaUrl.TrimEnd('/'))/api/tags" -UseBasicParsing -TimeoutSec 5
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Wait-ForOllama {
    param([int]$TimeoutSeconds = 30)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-OllamaApi) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Ensure-Ollama {
    $ollamaPath = Get-CommandPath "ollama"
    if (-not $ollamaPath) {
        Show-OllamaInstallHelp
        Fail "Install Ollama, make sure 'ollama' is on PATH, then rerun."
    }

    if (Test-OllamaApi) {
        Write-Step "Ollama is reachable at $($OllamaUrl.TrimEnd('/'))."
        return
    }

    if ($StartOllama -eq "never") {
        Show-OllamaInstallHelp
        Fail "Ollama is installed but not reachable at $($OllamaUrl.TrimEnd('/'))."
    }

    Write-Step "Ollama is installed but not reachable. Starting 'ollama serve' in the background..."
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdout = Join-Path $LogDir "ollama-serve-$stamp.log"
    $stderr = Join-Path $LogDir "ollama-serve-$stamp.err.log"
    $process = Start-Process -FilePath $ollamaPath -ArgumentList @("serve") -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Set-Content -Path (Join-Path $StateRoot "ollama.pid") -Value $process.Id

    if (Wait-ForOllama -TimeoutSeconds 30) {
        Write-Step "Ollama started. Log: $stdout"
        return
    }

    Write-Warn "Ollama did not become reachable within 30 seconds."
    Write-Warn "Ollama logs: $stdout and $stderr"
    Fail "Start Ollama manually, verify $($OllamaUrl.TrimEnd('/'))/api/tags, then rerun."
}

function Test-ZrokEnabled {
    param([string]$ZrokPath)
    try {
        $output = & $ZrokPath status 2>&1
        $text = ($output | Out-String)
        return ($LASTEXITCODE -eq 0 -and $text -match "Account Token\s+<<SET>>|Secret Token\s+<<SET>>|Ziti Identity\s+<<SET>>")
    } catch {
        return $false
    }
}

function ConvertFrom-SecureStringToPlainText {
    param([securestring]$SecureString)
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureString)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

function Ensure-Zrok {
    $zrokPath = Get-CommandPath "zrok"
    if (-not $zrokPath) {
        Show-ZrokInstallHelp
        Fail "Install zrok, make sure 'zrok' is on PATH, then rerun."
    }

    if (Test-ZrokEnabled -ZrokPath $zrokPath) {
        Write-Step "zrok environment is enabled."
        return $zrokPath
    }

    Write-Step "zrok is installed, but this shell is not enabled yet."
    $token = $env:ZROK_ENABLE_TOKEN
    if (-not $token) {
        $secureToken = Read-Host "Paste your zrok enable token. Input is hidden" -AsSecureString
        $token = ConvertFrom-SecureStringToPlainText -SecureString $secureToken
    } else {
        Write-Host "(using ZROK_ENABLE_TOKEN)"
    }

    if (-not $token) {
        Fail "No zrok enable token provided."
    }

    & $zrokPath enable $token | Out-Null
    $token = $null

    if (-not (Test-ZrokEnabled -ZrokPath $zrokPath)) {
        Fail "zrok enable finished, but 'zrok status' does not look enabled."
    }

    Write-Step "zrok environment enabled."
    return $zrokPath
}

function New-ApiKey {
    param([int]$Length)
    if ($Length -lt 32) {
        Fail "ApiKeyLength must be at least 32."
    }

    $chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".ToCharArray()
    $max = [math]::Floor(256 / $chars.Length) * $chars.Length
    $builder = New-Object System.Text.StringBuilder
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        while ($builder.Length -lt $Length) {
            $bytes = New-Object byte[] 64
            $rng.GetBytes($bytes)
            foreach ($byte in $bytes) {
                if ($byte -ge $max) { continue }
                [void]$builder.Append($chars[$byte % $chars.Length])
                if ($builder.Length -ge $Length) { break }
            }
        }
    } finally {
        $rng.Dispose()
    }
    return $builder.ToString()
}

function Get-FirstUrlFromFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    $text = Get-Content -Path $Path -Raw -ErrorAction SilentlyContinue
    if (-not $text) { return $null }
    $match = [regex]::Match($text, 'https?://[^\s")<>]+')
    if ($match.Success) { return $match.Value }
    return $null
}

function Start-ZrokShare {
    param(
        [string]$ZrokPath,
        [string]$ApiKey
    )

    Write-Step "Starting zrok tunnel to $($OllamaUrl.TrimEnd('/'))..."
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdout = Join-Path $LogDir "zrok-share-$stamp.log"
    $stderr = Join-Path $LogDir "zrok-share-$stamp.err.log"
    $combined = Join-Path $LogDir "zrok-share-combined-$stamp.log"
    $auth = "${BasicUser}:$ApiKey"
    $args = @(
        "share",
        "public",
        $OllamaUrl.TrimEnd('/'),
        "--backend-mode",
        "proxy",
        "--headless",
        "--basic-auth",
        $auth
    )

    $process = Start-Process -FilePath $ZrokPath -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Set-Content -Path (Join-Path $StateRoot "zrok.pid") -Value $process.Id

    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        $combinedText = ""
        if (Test-Path $stdout) { $combinedText += Get-Content -Path $stdout -Raw -ErrorAction SilentlyContinue }
        if (Test-Path $stderr) { $combinedText += "`n" + (Get-Content -Path $stderr -Raw -ErrorAction SilentlyContinue) }
        Set-Content -Path $combined -Value $combinedText

        $publicUrl = Get-FirstUrlFromFile -Path $combined
        if ($publicUrl) {
            Set-Content -Path (Join-Path $StateRoot "zrok.url") -Value $publicUrl
            return [pscustomobject]@{
                Url = $publicUrl
                Pid = $process.Id
                Log = $combined
            }
        }

        if ($process.HasExited) {
            Write-Warn "zrok exited before a public URL was printed."
            if (Test-Path $combined) { Get-Content -Path $combined | Write-Host }
            Fail "Unable to start zrok tunnel."
        }

        Start-Sleep -Seconds 1
        $process.Refresh()
    }

    Write-Warn "Timed out waiting for zrok to print a public URL."
    Write-Warn "zrok is still running as PID $($process.Id). Log: $combined"
    Fail "Unable to confirm the public zrok URL."
}

function Show-Success {
    param(
        [string]$PublicUrl,
        [string]$ApiKey,
        [int]$TunnelPid,
        [string]$LogFile
    )

    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Green
    Write-Host "  LOCAL OLLAMA API IS LIVE THROUGH ZROK" -ForegroundColor Green
    Write-Host "========================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Public URL:"
    Write-Host "  $PublicUrl" -ForegroundColor White
    Write-Host ""
    Write-Host "Basic Auth username:"
    Write-Host "  $BasicUser" -ForegroundColor White
    Write-Host ""
    Write-Host "API Key / Basic Auth password:"
    Write-Host "  $ApiKey" -ForegroundColor White
    Write-Host ""
    Write-Host "List installed models:"
    Write-Host "  curl.exe -u `"$BasicUser`:$ApiKey`" `"$PublicUrl/api/tags`"" -ForegroundColor White
    Write-Host ""
    Write-Host "Sample generate request:"
    Write-Host "  curl.exe -u `"$BasicUser`:$ApiKey`" `"$PublicUrl/api/generate`" -H `"Content-Type: application/json`" -d '{`"model`":`"gemma3:1b`",`"prompt`":`"Say hello from Ollama through zrok.`",`"stream`":false}'" -ForegroundColor White
    Write-Host ""
    Write-Host "Tunnel process:"
    Write-Host "  PID $TunnelPid" -ForegroundColor White
    Write-Host ""
    Write-Host "Stop the tunnel:"
    Write-Host "  Stop-Process -Id $TunnelPid" -ForegroundColor White
    Write-Host ""
    Write-Host "Logs:"
    Write-Host "  $LogFile" -ForegroundColor White
    Write-Host ""
    Write-Host "Security notes:"
    Write-Host "  - Do not paste this API key into public source code."
    Write-Host "  - Anyone with the URL and API key can use your local models while the tunnel is running."
    Write-Host "  - On shared machines, local admins or same-user processes may be able to inspect running command arguments. Use a host account you control."
    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Green
    Write-Host ""
}

function Main {
    if ($ApiKeyLength -lt 32) {
        Fail "ApiKeyLength must be at least 32."
    }

    Ensure-StateDirs
    Ensure-Ollama
    $zrokPath = Ensure-Zrok
    $apiKey = New-ApiKey -Length $ApiKeyLength
    $share = Start-ZrokShare -ZrokPath $zrokPath -ApiKey $apiKey
    Show-Success -PublicUrl $share.Url -ApiKey $apiKey -TunnelPid $share.Pid -LogFile $share.Log
}

Main
