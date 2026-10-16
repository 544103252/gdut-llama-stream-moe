param(
    [string] $Python = 'python',
    [string] $CMake = 'cmake',
    [string] $Model = 'D:\llama-cpp\models\Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf',
    [string] $Root = 'experiments/moe-markov',
    [switch] $SkipBuild
)

$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
    if (-not $SkipBuild) {
        & $CMake -S . -B build -G 'Visual Studio 18 2026' -A x64 -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON -DGGML_VULKAN_VALIDATE=OFF *> "$Root/configure.log"
        if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
        & $CMake --build build --config Release -j 8 *> "$Root/build.log"
        if ($LASTEXITCODE -ne 0) { throw 'Release build failed' }
    }
    & ./build/bin/Release/test-moe-markov.exe *> "$Root/unit.log"
    if ($LASTEXITCODE -ne 0) { throw 'Native cache tests failed' }
    & $Python scripts/test_moe_markov.py *> "$Root/python-tests.log"
    if ($LASTEXITCODE -ne 0) { throw 'Offline experiment tests failed' }
    foreach ($Stage in @('prepare', 'collect', 'calibrate', 'smoke', 'evaluate')) {
        if ($Stage -eq 'prepare' -and (Test-Path -LiteralPath "$Root/samples.json")) { continue }
        Write-Output "Markov experiment stage: $Stage"
        & $Python scripts/moe_markov.py $Stage --root $Root --model $Model *> "$Root/$Stage.log"
        if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Stage; inspect $Root/$Stage.log" }
    }
} finally {
    Pop-Location
}
