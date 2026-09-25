$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$VirtualEnvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VirtualEnvPython) {
    & $VirtualEnvPython -m bone_iqa
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 -m bone_iqa
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -m bone_iqa
} else {
    throw "未找到 Python 3。请安装 Python 3.11+，并先执行 pip install -e ."
}
