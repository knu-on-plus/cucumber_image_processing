param(
    [string]$DataRoot = ".\\data",
    [string]$OutRoot = ".\\outputs",
    [ValidateSet("train","valid","debugging")] [string]$DatasetType = "debugging",
    [ValidateSet("top","middle","bottom","random")] [string]$Position = "random",
    [ValidateSet(0,1,2)] [int]$MultiLeaves = 0,
    [bool]$RandomRatio = $true,
    [int]$SampleLimit = 5
)

python modal_mask_generation.py `
  --data_root $DataRoot `
  --out_root $OutRoot `
  --dataset_type $DatasetType `
  --position $Position `
  --multi_leaves $MultiLeaves `
  --random_ratio $RandomRatio `
  --sample_limit $SampleLimit
