#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root="$repo_root/outputs/runtime"
output_root="$repo_root/outputs/smoke_korean"
mkdir -p "$runtime_root" "$output_root"

stop_server() {
  local runtime_dir=$1
  if [[ -f "$runtime_dir/server.pid" ]]; then
    local pid
    pid=$(<"$runtime_dir/server.pid")
    if [[ $pid =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
      done
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
}

run_group() {
  local launch_config=$1 gpu=$2 port=$3
  shift 3
  local config_ids=("$launch_config" "$@")
  local runtime_dir="$runtime_root/${launch_config,,}-$port"
  if "$repo_root/.venv/bin/python" "$repo_root/scripts/smoke_status.py" \
      "$output_root" "${config_ids[@]}"; then
    echo "Skipping completed Korean smoke group: ${config_ids[*]}"
    return 0
  fi
  rm -f "$runtime_dir/ready.json"
  trap 'stop_server "$runtime_dir"' RETURN
  "$repo_root/scripts/serve_model.sh" "$launch_config" "$gpu" "$port" "$runtime_dir"
  local load_seconds
  load_seconds=$("$repo_root/.venv/bin/python" -c "import json; print(json.load(open('$runtime_dir/ready.json'))['load_seconds'])")
  for config_id in "${config_ids[@]}"; do
    "$repo_root/.venv/bin/python" "$repo_root/scripts/run_configuration.py" "$config_id" \
      --base-url "http://127.0.0.1:$port/v1" --output-root "$output_root" \
      --limit 1 --load-seconds "$load_seconds" --gpu-index "$gpu" \
      --prompt-version vqa-generation-ko-v1 --retry-invalid-once
  done
  stop_server "$runtime_dir"
  trap - RETURN
}

# GPU 0 is intentionally avoided because unrelated processes are present there.
run_group C01 1 8101 C10
run_group C02 2 8102 C11
run_group C03 3 8103 C09
run_group C04 1 8104 C06
run_group C05 2 8105
run_group C07 3 8107
run_group C08 1 8108

"$repo_root/.venv/bin/python" -m hsi_vqagen.evaluation.checks \
  --outputs "$output_root" \
  --require-configs C01,C02,C03,C04,C05,C06,C07,C08,C09,C10,C11 \
  --require-samples 1 --require-pairs 4
