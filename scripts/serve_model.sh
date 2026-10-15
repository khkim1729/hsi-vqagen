#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv_dir=${HSI_VQAGEN_VENV:-"$repo_root/.venv"}
if [[ ${1:-} == --show-runtime ]]; then
  echo "$venv_dir"
  exit 0
fi

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <config-id> <gpu-index> <port> <runtime-dir>" >&2
  exit 2
fi

config_id=$1
gpu_index=$2
port=$3
runtime_dir=$4
if [[ ! -x "$venv_dir/bin/python" || ! -x "$venv_dir/bin/vllm" ]]; then
  echo "runtime lacks python or vllm: $venv_dir" >&2
  exit 2
fi
"$venv_dir/bin/python" "$repo_root/scripts/server_process.py" --preflight --port "$port"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$runtime_dir"

mapfile -t model_fields < <(
  "$venv_dir/bin/python" - "$repo_root" "$config_id" <<'PY'
import sys
from pathlib import Path
from hsi_vqagen.config import load_experiment_registry
root, config_id = Path(sys.argv[1]), sys.argv[2]
config = load_experiment_registry(root / "configs" / "experiments.yaml").by_id[config_id]
print(config.checkpoint)
print(config.revision)
print("true" if config.template_controls.get("trust_remote_code") else "false")
print(config.template_controls.get("tokenizer_mode") or "")
print(config.template_controls.get("config_format") or "")
print(config.template_controls.get("load_format") or "")
PY
)
checkpoint=${model_fields[0]}
revision=${model_fields[1]}
trust_remote=${model_fields[2]}
tokenizer_mode=${model_fields[3]}
config_format=${model_fields[4]}
load_format=${model_fields[5]}

args=(
  "$venv_dir/bin/vllm" serve "$checkpoint"
  --revision "$revision"
  --served-model-name "$checkpoint"
  --dtype bfloat16
  --max-model-len 8192
  --gpu-memory-utilization 0.85
  --port "$port"
)
if [[ $trust_remote == true ]]; then args+=(--trust-remote-code); fi
if [[ -n $tokenizer_mode ]]; then args+=(--tokenizer-mode "$tokenizer_mode"); fi
if [[ -n $config_format ]]; then args+=(--config-format "$config_format"); fi
if [[ -n $load_format ]]; then args+=(--load-format "$load_format"); fi

export HF_HOME=/data/hsi-vqagen-cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/hsi-vqagen-cache/huggingface/hub
server_pid=""
server_start_ticks=""
handed_off=false
cleanup_unhanded_server() {
  status=$?
  trap - EXIT
  if [[ $handed_off != true && -n $server_pid && -n $server_start_ticks ]]; then
    if "$venv_dir/bin/python" "$repo_root/scripts/server_process.py" --same-launch \
        --pid "$server_pid" --expected-start-ticks "$server_start_ticks" \
        --model "$checkpoint" --revision "$revision" >/dev/null 2>&1; then
      kill -TERM -- "-$server_pid" || true
    fi
  fi
  exit "$status"
}
trap cleanup_unhanded_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_epoch=$(date +%s)
setsid env PATH="$venv_dir/bin:$PATH" CUDA_VISIBLE_DEVICES="$gpu_index" \
  "$venv_dir/bin/python" "$repo_root/scripts/server_supervisor.py" -- "${args[@]}" \
  >"$runtime_dir/server.log" 2>&1 &
server_pid=$!
server_start_ticks=$("$venv_dir/bin/python" "$repo_root/scripts/server_process.py" --start-ticks --pid "$server_pid")
echo "$server_pid" >"$runtime_dir/server.pid"

for _ in $(seq 1 240); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited before readiness; inspect $runtime_dir/server.log" >&2
    exit 1
  fi
  if "$venv_dir/bin/python" "$repo_root/scripts/check_server.py" \
      --base-url "http://127.0.0.1:$port/v1" --model "$checkpoint" \
      --revision "$revision" --health-only >/dev/null 2>&1; then
    "$venv_dir/bin/python" "$repo_root/scripts/server_process.py" \
      --assert-owned --port "$port" --pgid "$server_pid"
    break
  fi
  sleep 5
done

verification=$("$venv_dir/bin/python" "$repo_root/scripts/check_server.py" \
  --base-url "http://127.0.0.1:$port/v1" --model "$checkpoint" --revision "$revision")
"$venv_dir/bin/python" "$repo_root/scripts/server_process.py" \
  --assert-owned --port "$port" --pgid "$server_pid"
ready_epoch=$(date +%s)
load_seconds=$((ready_epoch - start_epoch))
"$venv_dir/bin/python" - "$runtime_dir" "$config_id" "$checkpoint" "$revision" "$gpu_index" "$port" "$server_pid" "$server_start_ticks" "$load_seconds" "$verification" <<'PY'
import json,sys
from pathlib import Path
runtime, config_id, model, revision, gpu, port, pid, start_ticks, load_seconds, verification = sys.argv[1:]
payload = {"config_id": config_id, "model": model, "revision": revision,
           "gpu_index": int(gpu), "port": int(port), "pid": int(pid),
           "process_start_ticks": start_ticks, "load_seconds": float(load_seconds),
           **json.loads(verification)}
Path(runtime, "ready.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
handed_off=true
trap - EXIT INT TERM
echo "$runtime_dir/ready.json"
