param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetPath
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigDir = Join-Path $Root "mmdetection_github\configs"
$ScriptDir = Join-Path $Root "mmdetection_github"

# Normalize: forward slashes, no trailing slash
$Path = $DatasetPath.Replace('\', '/').TrimEnd('/')

Write-Host "=== Setting dataset root to: $Path ==="

# 1. data_root in all configs
Write-Host "[1/3] Updating data_root in configs..."
Get-ChildItem $ConfigDir -Recurse -Filter "*.py" | ForEach-Object {
    $c = Get-Content $_.FullName -Raw
    # Replace any data_root path with the new one
    $c = $c -replace "data_root\s*=\s*'[^']*'", "data_root = '$Path'"
    Set-Content $_.FullName $c -NoNewline
}
Write-Host "       Done."

# 2. Fix _base_ absolute -> relative (idempotent)
Write-Host "[2/3] Fixing _base_ paths..."
$BasePath = '/home1/guojs/codes/Dark-ISP/mmdetection_github/configs/_base_/'
$LodRel = '../../_base_/'
$FlatRel = '../_base_/'

Get-ChildItem (Join-Path $ConfigDir "LOD") -Recurse -Filter "*.py" | ForEach-Object {
    $c = Get-Content $_.FullName -Raw
    $c = $c.Replace($BasePath, $LodRel)
    Set-Content $_.FullName $c -NoNewline
}
Get-ChildItem (Join-Path $ConfigDir "COCO") -Recurse -Filter "*.py" | ForEach-Object {
    $c = Get-Content $_.FullName -Raw
    $c = $c.Replace($BasePath, $FlatRel)
    Set-Content $_.FullName $c -NoNewline
}
Get-ChildItem (Join-Path $ConfigDir "NOD") -Recurse -Filter "*.py" | ForEach-Object {
    $c = Get-Content $_.FullName -Raw
    $c = $c.Replace($BasePath, $FlatRel)
    Set-Content $_.FullName $c -NoNewline
}
Write-Host "       Done."

# 3. Update scripts
Write-Host "[3/3] Updating scripts..."
$Scripts = @(
    "$ScriptDir\LOD_RAW_preprocess.py",
    "$ScriptDir\NOD_RAW_preprocess.py",
    "$ScriptDir\COCO_Syn_preprocess.py",
    "$ScriptDir\data_preparation\bayer_to_npy.py",
    "$ScriptDir\test_dataset.py"
)
foreach ($s in $Scripts) {
    if (Test-Path $s) {
        $c = Get-Content $s -Raw
        # Replace old server path or local path with new
        $c = $c -replace "'[^']*/(LOD|NOD|COCO|dataset)[^']*'", "'$Path'"
        $c = $c -replace "'/public/home/guojs/data/[^']*'", "'$Path'"
        Set-Content $s $c -NoNewline
        Write-Host "       $s"
    }
}
Write-Host "       Done."

Write-Host "=== Setup complete ==="
