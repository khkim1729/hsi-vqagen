# ICCE 2027 paper experiments

These studies extend the completed five-scene Korean feasibility run. They do not
replace its manifest or write into its saved outputs. Both manifests below were
frozen before any new inference.

## Fifty-scene paired study

[`paper_samples_50.yaml`](../configs/paper_samples_50.yaml) selects 50 scenes from
the audited 10,000 fully paired records. Selection is deterministic with seed
`icce-2027-v1`: records are ranked by SHA-256 of seed and sample ID within
NEON-site × global description-length-tertile strata; selection rotates across
sites and length bands while preferring an unused shard. The boundaries in the
source collection are 705 and 798 characters. The selected cohort contains 17
short, 15 middle, and 18 long descriptions across all 23 available NEON sites
and 39 shards. It excludes the original five scenes. Equal site coverage is a
deliberate stress on transfer across sites, not a population-proportional sample.

The ordered manifest hash is
`40d3f07e006cd255dee4245fec224b5da16ed958f8dc47b458d83048524a8ec2`.
Each entry also pins SHA-256 digests of the UTF-8 description and the exact RGB
file bytes. The ordered content commitment is
`fb2764269685ac84347f43c15823bcf3565351034ea8ea3a2b0aa0d41922791f`;
the loader checks both the manifest and this independently pinned value before
inference. Its `experiment_id` is `hsi-vqagen-icce2027-paper50-v1`, separate
from the five-scene feasibility ID.
Run the four weight-controlled pairs C01/C10, C02/C11, C03/C09, and C04/C06.
This is 400 requested cases and 1,600 requested QA pairs. The original
checkpoint revisions, Korean prompt, temperature 0, seed, and 4-QA schema are
held fixed. Each pair differs only by the presence of the scene's RGB rendering.

## Twenty-scene RGB robustness study

[`robustness_samples_20.yaml`](../configs/robustness_samples_20.yaml) is a frozen
subset of the 50 scenes. Within each NEON site, candidate pairs were ranked by
Euclidean difference in mean RGB color, normalized by the maximum possible RGB
distance. The ten highest-contrast sites supply two scenes each; their RGBs
are swapped bidirectionally within site. This intentionally high-contrast
cohort measures sensitivity to an inconsistent RGB input and is not a random
estimate of error prevalence. Human inspection must determine whether an
output actually contradicts the provided image or description.

The ordered sample hash is
`636ce872445090612592511e4805eba08dbd4f42634e82eacb71532b28238d9b`;
the ordered image-mapping hash is
`3420119df2d0eed7b80b08907d88b26ed7d932a598828ac840df93138ff6fc10`.
The ordered description/RGB content commitment is
`5c2ef2dc06c0f5cc3c929ac9a448c42f1a8a9b633e4814b9ae19a80d95442ce0`.
The robustness `experiment_id` is `hsi-vqagen-icce2027-rgb-robustness-v1`.
R01/R02/R03 use the pinned Qwen3-VL-8B checkpoint with correct RGB, shuffled
RGB, and description only. R04/R05/R06 repeat those three conditions with the
pinned Gemma-4-12B checkpoint. These IDs, output directories, provenance
fields, and manifest identity are distinct from C01–C11. The 20 scenes × six
conditions request 120 cases and 480 QA pairs.

## Execution when a GPU is genuinely idle

As of 2026-10-15 UTC, GPUs 0–3 were occupied by other users. No paper or
robustness inference was started. Check both utilization and compute processes
immediately before a run. The orchestrator refuses a GPU reporting more than
512 MiB used, nonzero utilization, or a compute process; it never chooses a GPU
or stops another user's process. It starts each model on the explicit GPU and
stops only the server process group it launched after verifying its PID and
command and process start time. The server launcher refuses a bound port
before launch and requires the ready listener's socket to belong to its new
process group. A failed or interrupted launch cleans up only its verified
child; a successful launch hands its PID and start-time token to the study
orchestrator. A group-leader supervisor also stops leftover vLLM workers if
the serving parent exits first, escalating only its own still-live session
after a 30-second grace period. It never accepts an older compatible service
as its own.

The registry pins the requested Hugging Face revision and the launch command
passes that revision to vLLM. `check_server.py` validates the served model ID
and compares a served revision if the `/models` endpoint exposes one. Most
OpenAI-compatible model cards do not expose a weight commit, so `ready.json`
records `configured_revision` separately from `served_revision_verified`
(normally false) and never claims the runtime weights were independently
verified. Run/case `revision` is configured provenance, not a cryptographic
attestation of loaded weights.

```bash
cd /home/khkim/gnew_projects/01_hsi_vqa_gen_github/hsi-vqagen
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv
bash scripts/run_paper_studies.sh --plan paper50
bash scripts/run_paper_studies.sh --plan robustness
```

Once an index is genuinely idle, replace `<idle-gpu-index>` with that index:

```bash
HSI_VQAGEN_VENV="$PWD/.worktrees/feasibility-implementation/.venv" \
HSI_VQAGEN_LOCAL_PATHS="$PWD/.worktrees/feasibility-implementation/configs/local_paths.yaml" \
bash scripts/run_paper_studies.sh paper50 <idle-gpu-index>

HSI_VQAGEN_VENV="$PWD/.worktrees/feasibility-implementation/.venv" \
HSI_VQAGEN_LOCAL_PATHS="$PWD/.worktrees/feasibility-implementation/configs/local_paths.yaml" \
bash scripts/run_paper_studies.sh robustness <idle-gpu-index>
```

The paper group runs C01/C10 on one Qwen server, C02/C11 on one InternVL
server, C03/C09 on one Gemma server, then C04/C06 on one Mistral server.
Robustness runs R01/R02/R03 on the Qwen server and R04/R05/R06 on the Gemma
server. Raw responses, normalized rows, and manifests go to ignored
`outputs/paper50/` and `outputs/robustness/` in this worktree. The launcher
uses the existing pinned virtual environment and local dataset path config
without copying either into the tracked repository. Existing five-scene
outputs in the older worktree remain untouched.

Using the observed five-scene mean case latencies, 50-scene inference is about
30 minutes of generation plus about 10 minutes of four checkpoint loads;
allow 45–60 minutes including scans and retries on one idle H200. The
robustness run is about 9 minutes of generation plus roughly 7 minutes of
two checkpoint loads; allow 20–30 minutes. These are planning estimates,
not measured times for the new cohorts.

## Analysis after complete inference

Only after every condition has completed, run:

```bash
PYTHONPATH=src .worktrees/feasibility-implementation/.venv/bin/python \
  scripts/evaluate_studies.py --study paper50 --outputs outputs/paper50 \
  --artifact-prefix artifacts/paper50

PYTHONPATH=src .worktrees/feasibility-implementation/.venv/bin/python \
  scripts/evaluate_studies.py --study robustness --outputs outputs/robustness \
  --artifact-prefix artifacts/robustness
```

The analyzer checks the ordered scene manifest, Korean prompt, completed and
failed case sets, registry checkpoint/revision/generation settings, image
variant and mapping hash, frozen input digests, per-row provenance, and four
normalized pairs per completed case. It exports
JSON; case, configuration, and comparison CSVs; and an overview PNG/PDF.
By default it refuses incomplete cohorts, so `final_complete` reports require
all registered conditions and all frozen scenes. For a deliberately partial
diagnostic, pass `--allow-partial`; the report is labeled
`exploratory_partial`, lists missing scenes, and must not be presented as a
final paper result. Bootstrap CIs are withheld for comparisons with fewer
than 10 paired scenes; eligible comparison bars remain visible in mixed
exploratory figures. Rows retain the registry generation settings as the
requested base settings and separately record the effective successful-attempt
settings and whether a validation retry used `seed+1`. The analyzer verifies
those fields against the run telemetry and retry policy. Per-case
`raw/*.attempts.json` retains complete first/retry backend records without
changing the legacy first-attempt `raw/*.json` format.
Validation rate uses all frozen scenes as its denominator. Question-type
distribution uses the model's declared labels. Paired deltas and seeded 95%
bootstrap intervals resample scenes, with each scene contributing one case per
condition; independently generated QA pairs are not treated as matched
observations. Source-description cosine is lexical overlap, not correctness.
No automated statistic here establishes visual grounding or answer truth.
Those claims require the planned blind human review against the available
description and RGB, with explicit tracking of the shuffled image source.
