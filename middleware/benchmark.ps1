# Magna Middleware — Benchmark Script
# Usage: .\benchmark.ps1 -EnginePath "path\to\model.engine" [-ImagePath "path\to\image.jpg"] [-Iterations 100]

param(
    [Parameter(Mandatory=$true)]
    [string]$EnginePath,
    
    [string]$ImagePath = "",
    
    [int]$Iterations = 100,
    
    [string]$Backend = "simulated",
    
    [string]$Precision = "fp32",
    
    [string]$LabelsPath = "",
    
    [int]$Warmup = 5
)

Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Magna Middleware — Benchmark Runner      " -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Engine     : $EnginePath"
Write-Host "  Backend    : $Backend"
Write-Host "  Precision  : $Precision"
Write-Host "  Warmup     : $Warmup"
Write-Host "  Iterations : $Iterations"
Write-Host ""

$args = @(
    "--engine", $EnginePath,
    "--backend", $Backend,
    "--precision", $Precision,
    "--warmup", $Warmup.ToString()
)

if ($ImagePath -ne "") {
    $args += @("--image", $ImagePath, "--benchmark", $Iterations.ToString())
}

if ($LabelsPath -ne "") {
    $args += @("--labels", $LabelsPath)
}

Write-Host "Running: cargo run --release -- $($args -join ' ')" -ForegroundColor Yellow
Write-Host ""

cargo run --release -- @args

Write-Host ""
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Benchmark Complete                       " -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
