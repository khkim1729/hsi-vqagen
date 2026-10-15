#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
study=${2:-}
plan_only=false
if [[ ${1:-} == --plan ]]; then
  plan_only=true
else
  study=${1:-}
fi

case "$study" in
  paper50)
    sample_manifest="$repo_root/configs/paper_samples_50.yaml"
    sample_hash=40d3f07e006cd255dee4245fec224b5da16ed958f8dc47b458d83048524a8ec2
    sample_count=50
    groups=("C01:8811:C01 C10" "C02:8812:C02 C11" "C03:8813:C03 C09" "C04:8814:C04 C06")
    registry_args=()
    ;;
  robustness)
    sample_manifest="$repo_root/configs/robustness_samples_20.yaml"
    sample_hash=636ce872445090612592511e4805eba08dbd4f42634e82eacb71532b28238d9b
    sample_count=20
    groups=("C01:8821:R01 R02 R03" "C03:8823:R04 R05 R06")
    registry_args=(--robustness-registry "$repo_root/configs/robustness_experiments.yaml")
    ;;
  *)
    echo "usage: $0 [--plan] <paper50|robustness> [idle-gpu-index]" >&2
    exit 2
    ;;
esac

if [[ $plan_only == true ]]; then
  for group in "${groups[@]}"; do
    IFS=: read -r launch port configs <<<"$group"
    echo "$study: serve $launch on port $port; run $configs"
  done
  exit 0
fi

gpu_index=${2:-}
if [[ ! $gpu_index =~ ^[0-3]$ ]]; then
  echo "an explicit GPU index 0-3 is required" >&2
  exit 2
fi
export HSI_VQAGEN_VENV=${HSI_VQAGEN_VENV:-"$repo_root/.worktrees/feasibility-implementation/.venv"}
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
local_paths=${HSI_VQAGEN_LOCAL_PATHS:-"$repo_root/.worktrees/feasibility-implementation/configs/local_paths.yaml"}
if [[ ! -x "$HSI_VQAGEN_VENV/bin/python" || ! -f "$local_paths" ]]; then
  echo "existing venv or local path config is unavailable" >&2
  exit 2
fi

# Validate frozen manifests and configurations before loading a checkpoint.
"$HSI_VQAGEN_VENV/bin/python" - "$study" <<'PY'
import sys
from scripts.evaluate_studies import resolve_study
resolve_study(sys.argv[1])
PY

gpu_uuid=$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader)
gpu_numbers=$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r gpu_memory gpu_utilization <<<"$gpu_numbers"
gpu_memory=${gpu_memory//[[:space:]]/}
gpu_utilization=${gpu_utilization//[[:space:]]/}
if (( gpu_memory > 512 || gpu_utilization > 0 )); then
  echo "GPU $gpu_index is not idle (memory=${gpu_memory} MiB, utilization=${gpu_utilization}%)" >&2
  exit 1
fi
while IFS=, read -r process_uuid process_pid; do
  process_uuid=${process_uuid//[[:space:]]/}
  if [[ $process_uuid == "$gpu_uuid" && -n $process_pid ]]; then
    echo "GPU $gpu_index has an active compute process; refusing to start" >&2
    exit 1
  fi
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)

server_pid=""
server_checkpoint=""
server_revision=""
server_start_ticks=""
server_handoff_ready=""
read_ready_metadata() {
  local fields
  fields=$("$HSI_VQAGEN_VENV/bin/python" - "$server_handoff_ready" <<'PY'
import json,sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
print("\t".join(str(data[key]) for key in
      ("pid", "model", "revision", "process_start_ticks", "load_seconds")))
PY
  ) || return 1
  IFS=$'\t' read -r server_pid server_checkpoint server_revision server_start_ticks load_seconds <<<"$fields"
}
cleanup_server() {
  if [[ -n $server_handoff_ready && ( -z $server_pid || -z $server_checkpoint || -z $server_revision || -z $server_start_ticks ) ]]; then
    if ! read_ready_metadata; then
      echo "server handoff metadata is unreadable: $server_handoff_ready" >&2
      return 1
    fi
  fi
  if [[ -z $server_pid ]] || ! kill -0 "$server_pid" 2>/dev/null; then
    server_pid=""
    server_handoff_ready=""
    return 0
  fi
  if ! "$HSI_VQAGEN_VENV/bin/python" "$repo_root/scripts/server_process.py" --same-launch \
      --pid "$server_pid" --expected-start-ticks "$server_start_ticks" \
      --model "$server_checkpoint" --revision "$server_revision" >/dev/null 2>&1; then
    echo "refusing to stop PID $server_pid: launch identity does not match" >&2
    return 1
  fi
  kill -TERM -- "-$server_pid"
  for _ in $(seq 1 30); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      server_pid=""
      server_handoff_ready=""
      return 0
    fi
    sleep 1
  done
  echo "server group $server_pid did not exit after SIGTERM; inspect it manually" >&2
  return 1
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for group in "${groups[@]}"; do
  IFS=: read -r launch port configs <<<"$group"
  runtime_dir="$repo_root/outputs/runtime/$study-$launch"
  "$repo_root/scripts/serve_model.sh" "$launch" "$gpu_index" "$port" "$runtime_dir"
  server_handoff_ready="$runtime_dir/ready.json"
  read_ready_metadata
  read -r -a config_ids <<<"$configs"
  for config_id in "${config_ids[@]}"; do
    "$HSI_VQAGEN_VENV/bin/python" "$repo_root/scripts/run_configuration.py" "$config_id" \
      --base-url "http://127.0.0.1:$port/v1" \
      --sample-manifest "$sample_manifest" \
      --sample-manifest-sha256 "$sample_hash" \
      --sample-count "$sample_count" \
      --output-root "$repo_root/outputs/$study" \
      --local-paths "$local_paths" \
      --gpu-index "$gpu_index" --load-seconds "$load_seconds" \
      --prompt-version vqa-generation-ko-v1 --retry-invalid-once \
      "${registry_args[@]}"
  done
  cleanup_server
done
trap - EXIT INT TERM
