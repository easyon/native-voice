# Dev gateway: fake backend by default, local GPU models with -Backend local.
# LLM placement: $env:LLM_DEVICE = "cpu" (llama.cpp -ngl 0 + CPU binary) or
# "gpu" (-ngl 99 + CUDA binary). $env:LLM_SERVER overrides the binary path.
param(
  [ValidateSet("local", "fake", "worker")][string]$Backend = "fake",
  [string]$Config = "configs/service.yaml",
  [int]$Port = 8090,
  [string]$Asr = "",       # paraformer | qwen3asr
  [string]$Tts = "",       # cosyvoice3 | qwen3tts | qwen3tts_vllm | cosyvoice3_vllm
  [switch]$NoDevKey
)
$root = Resolve-Path "$PSScriptRoot\.."
$env:PYTHONPATH = "$root\src"
$args = @("serve", "--config", $Config, "--backend", $Backend, "--port", "$Port")
if ($Asr) { $args += @("--asr-provider", $Asr) }
if ($Tts) { $args += @("--tts-provider", $Tts) }
switch ($env:LLM_DEVICE) {
  "cpu" {
    $args += @("--llm-gpu-layers", "0")
    if (Test-Path "$root\bin\llama\llama-server.exe") {
      $args += @("--llm-server", "bin/llama/llama-server.exe")
    }
  }
  "gpu" {
    $args += @("--llm-gpu-layers", "99")
    if (Test-Path "$root\bin\llama-cuda\llama-server.exe") {
      $args += @("--llm-server", "bin/llama-cuda/llama-server.exe")
    }
  }
  "" { }
  default {
    Write-Error "LLM_DEVICE must be 'cpu' or 'gpu' (got '$env:LLM_DEVICE')"
    exit 2
  }
}
if ($env:LLM_SERVER) { $args += @("--llm-server", $env:LLM_SERVER) }
if (-not $NoDevKey) { $args += "--dev-key" }
$exe = Join-Path $root ".venv\Scripts\native-voice.exe"
if (Test-Path $exe) { & $exe @args; exit $LASTEXITCODE }
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python -m native_voice.cli.main @args
