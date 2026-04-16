#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_LIST="GPVAR" bash "$script_dir/run_ar_baseline.sh"
