# One-command runner for the MEMOIR vs lasso-recipe editing benchmark (RTX 3060, Windows).
# Usage (PowerShell, from anywhere):
#   & "$HOME\gemma-lasso-router\continual_learning\run_edit_bench.ps1"
# Optional:  -NEdits 1000  -Model "Qwen/Qwen2.5-0.5B-Instruct"
param(
    [int]$NEdits = 1000,
    [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct"
)
$ErrorActionPreference = "Continue"     # Python writes warnings/progress bars to stderr; don't treat them as errors
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$root = "$HOME\gemma-lasso-router"
$here = "$root\continual_learning"
$py = "$root\.venv\Scripts\python.exe"
$data = "$here\zsre_mend_eval.json"
$out = "$here\edit_results"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# 1. benchmark data (7.7 MB, standard zsRE evaluation split used by MEND / WISE / MEMOIR)
if (-not (Test-Path $data)) {
    Write-Host "Downloading zsRE evaluation split..."
    Invoke-WebRequest -Uri "https://rome.baulab.info/data/dsets/zsre_mend_eval.json" -OutFile $data
}

# 2. GPU check
& $py -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | ForEach-Object { "$_" }

# 3. runs (each ~20-40 min on a 3060 at 1000 edits; estimates)
$runs = @(
    @{name="memoir";           args=@("--method","memoir")},
    @{name="slot_lasso_B500";  args=@("--method","slot_lasso","--budget","500")},
    @{name="slot_unbounded";   args=@("--method","slot_unbounded")},
    @{name="slot_evict_B500";  args=@("--method","slot_evict","--budget","500")},
    @{name="slot_lasso_B250";  args=@("--method","slot_lasso","--budget","250")},
    @{name="slot_evict_B250";  args=@("--method","slot_evict","--budget","250")}
)
foreach ($r in $runs) {
    $res = "$out\$($r.name)_N$NEdits.json"
    if (Test-Path $res) { Write-Host "skip $($r.name) (already done)"; continue }
    Write-Host "=== $($r.name) ==="
    # stderr (warnings, progress bars) is converted to plain text so it can never stop the script
    & $py "$here\edit_bench.py" --model $Model --data $data --n_edits $NEdits --out $res @($r.args) 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath "$out\$($r.name)_N$NEdits.log"
    if (-not (Test-Path $res)) { Write-Host "!!! $($r.name) did not produce results; see $out\$($r.name)_N$NEdits.log" }
}

# 4. summary table
& $py "$here\summarize_edit_bench.py" $out 2>&1 | ForEach-Object { "$_" }
Write-Host "Done. Results in $out"
