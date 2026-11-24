$ErrorActionPreference = "Stop"
$Source = "D:\Thesis"
$Target = "D:\NeuroFusionNet Tumor App"

New-Item -ItemType Directory -Force "$Target\Models","$Target\src","$Target\configs","$Target\checkpoints","$Target\outputs" | Out-Null

if (!(Test-Path "$Source\Models\factory.py")) { throw "Missing D:\Thesis\Models\factory.py" }
if (!(Test-Path "$Source\src\xai.py")) { throw "Missing D:\Thesis\src\xai.py" }
if (!(Test-Path "$Source\Results\NeuroFusionNet\Checkpoints\best_model.pth")) { throw "Missing NeuroFusionNet best_model.pth" }

Copy-Item "$Source\Models\*" "$Target\Models" -Recurse -Force
Copy-Item "$Source\src\*" "$Target\src" -Recurse -Force
Copy-Item "$Source\configs\*" "$Target\configs" -Recurse -Force
Copy-Item "$Source\Results\NeuroFusionNet\Checkpoints\best_model.pth" "$Target\checkpoints\best_model.pth" -Force

Write-Host "Model code, XAI code, config and best checkpoint copied successfully."
Write-Host "Project: $Target"
