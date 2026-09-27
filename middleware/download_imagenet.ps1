# =============================================================================
# ImageNet Download & Setup Script (Resilient Version)
# =============================================================================

$ErrorActionPreference = "Stop"

$url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
$outFile = "ILSVRC2012_img_val.tar"
$extractDir = "imagenet_val"

Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ImageNet Dataset Preparation             " -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan

# Use direct curl.exe for better control and resume support
if (Get-Command "curl.exe" -ErrorAction SilentlyContinue) {
    Write-Host "Downloading (Resumable Mode)..." -ForegroundColor Yellow
    
    # -L: Follow redirects
    # -k: Insecure (skip cert check)
    # -C -: Continue/Resume
    # --retry 10: Retry on transient errors
    # -o: Output file
    # -A: User agent
    curl.exe -L -k -C - --retry 10 -A "Mozilla/5.0" -o $outFile $url
    
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 33) { # 33 is often "already fully downloaded" for some curl versions
        Write-Host "Warning: curl exited with code $LASTEXITCODE" -ForegroundColor Red
    }
} else {
    Write-Host "curl.exe not found, falling back to BITS..." -ForegroundColor Magenta
    [System.Net.ServicePointManager]::ServerCertificateValidationCallback = {$true}
    if (!(Test-Path $outFile)) {
        Start-BitsTransfer -Source $url -Destination $outFile -Description "ImageNet Download"
    } else {
        Write-Host "File exists, BITS cannot easily resume. Please install curl for resume support." -ForegroundColor Yellow
    }
}

# Verify and Extract
if (Test-Path $outFile) {
    $fileSize = (Get-Item $outFile).Length
    if ($fileSize -gt 6000MB) {
        Write-Host "Download verified ($($fileSize / 1MB) MB). Extracting..." -ForegroundColor Green
        if (!(Test-Path $extractDir)) { New-Item -ItemType Directory -Path $extractDir }
        tar -xf $outFile -C $extractDir
        Write-Host "Extraction complete!" -ForegroundColor Green
    } else {
        Write-Host "Error: Downloaded file is too small ($($fileSize / 1MB) MB). It likely failed half-way." -ForegroundColor Red
        Write-Host "Rerun the script to resume the download." -ForegroundColor Yellow
    }
} else {
    Write-Host "Error: Could not find $outFile" -ForegroundColor Red
}
