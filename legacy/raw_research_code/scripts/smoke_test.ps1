param(
    [string]$DataRoot = ".\\data",
    [string]$OutRoot = ".\\outputs"
)

Write-Host "Checking python availability..."
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python not found in PATH. Install Python 3.10+ and retry." -ForegroundColor Yellow
    exit 1
}

Write-Host "Running a small debugging sample..."
python modal_mask_generation.py `
  --data_root $DataRoot `
  --out_root $OutRoot `
  --dataset_type debugging `
  --position random `
  --multi_leaves 0 `
  --random_ratio true `
  --sample_limit 2
