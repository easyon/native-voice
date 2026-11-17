# Create isolated venvs for the optional Qwen3 ASR/TTS engines.
#
#   scripts\setup_qwen_engines.ps1 -Kind tts -DownloadModels
#   scripts\setup_qwen_engines.ps1 -Kind asr -DownloadModels
#   scripts\setup_qwen_engines.ps1 -Kind both
#
# Qwen3-TTS pins transformers==4.57.3 and Qwen3-ASR pins transformers==4.57.6,
# so they intentionally do NOT share the root .venv (which uses 4.51.3 for
# CosyVoice3/FunASR). Only one engine per stage runs at a time.
param(
  [ValidateSet("tts", "asr", "both")][string]$Kind = "both",
  [switch]$DownloadModels
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
$uv = (Get-Command uv -ErrorAction Stop).Source
$pyVersion = "3.10"
$mirror = "https://mirrors.aliyun.com/pytorch-wheels/cu121"
$torch = "$mirror/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl"
$torchaudio = "$mirror/torchaudio-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl"

function New-QwenEnv([string]$name) {
  $dir = Join-Path $root "envs\$name"
  $py = Join-Path $dir "Scripts\python.exe"
  if (-not (Test-Path $py)) {
    Write-Host "creating $dir"
    & $uv venv $dir --python $pyVersion
  }
  Write-Host "installing base project deps into $name"
  & $uv pip install --python $py -e $root
  Write-Host "installing torch 2.5.1+cu121 into $name"
  & $uv pip install --python $py $torch $torchaudio
  return $py
}

if ($Kind -in @("tts", "both")) {
  $py = New-QwenEnv "qwen3tts"
  Write-Host "installing qwen-tts"
  & $uv pip install --python $py `
    "transformers==4.57.3" "accelerate==1.12.0" `
    "librosa>=0.10,<0.11" "soundfile>=0.12" "sox" "onnxruntime>=1.18,<1.20" "einops>=0.8,<1"
  & $uv pip install --python $py "qwen-tts==0.1.1" --no-deps
}

if ($Kind -in @("asr", "both")) {
  $py = New-QwenEnv "qwen3asr"
  Write-Host "installing qwen-asr"
  & $uv pip install --python $py `
    "transformers==4.57.6" "accelerate==1.12.0" `
    "nagisa==0.2.11" "soynlp==0.0.493" "librosa>=0.10,<0.11" "soundfile>=0.12"
  & $uv pip install --python $py "qwen-asr==0.0.6" --no-deps
}

if ($DownloadModels) {
  $ms = Join-Path $root ".venv\Scripts\modelscope.exe"
  if (-not (Test-Path $ms)) {
    throw "modelscope CLI not found in .venv; run 'uv sync --extra gpu' first"
  }
  if ($Kind -in @("tts", "both")) {
    Write-Host "downloading Qwen3-TTS-12Hz-0.6B-Base"
    & $ms download --model "Qwen/Qwen3-TTS-12Hz-0.6B-Base" `
      --local_dir (Join-Path $root "models\Qwen3-TTS-12Hz-0.6B-Base")
  }
  if ($Kind -in @("asr", "both")) {
    Write-Host "downloading Qwen3-ASR-0.6B"
    & $ms download --model "Qwen/Qwen3-ASR-0.6B" `
      --local_dir (Join-Path $root "models\Qwen3-ASR-0.6B")
  }
}

Write-Host "done. select with:"
Write-Host "  scripts\serve_dev.ps1 -Backend worker -Asr qwen3asr -Tts qwen3tts"
