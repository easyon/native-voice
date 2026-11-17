@echo off
rem Launch the full-duplex voice call with the project's conda env.
rem Edit PY if your Python environment lives elsewhere.
setlocal
set "PY=C:\Users\xcc\AppData\Local\miniconda3\envs\cosyvoice\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0desktop\voice_call.py" %*
if errorlevel 1 pause
