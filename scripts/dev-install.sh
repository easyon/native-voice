#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e packages/common
python -m pip install -e "packages/runtime[dev]"
python -m pip install -e "packages/data[dev]"
python -m pip install -e "packages/training[dev]"
python -m pip install -e "packages/evaluation[dev]"
