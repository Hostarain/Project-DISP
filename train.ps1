# ============================================
#  训练脚本 — 修改下面几个参数即可
# ============================================

$Config = "configs/LOD/VOCmetric/smoke_test.py"
$TrainName = "20260513_test"
$ExtraArgs = ""

# ============================================
#  下面不用改
# ============================================
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $Root "mmdetection_github\work_dirs\$TrainName"

Write-Host "Config  : $Config"
Write-Host "Work Dir: $WorkDir"
Write-Host "Checkpoints will be saved to: checkpoints/$TrainName/"
Write-Host ""

Set-Location "$Root\mmdetection_github"
python tools/train.py $Config --work-dir $WorkDir $ExtraArgs
