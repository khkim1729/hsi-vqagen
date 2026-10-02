# Synthetic shard fixture

Dataset tests build tiny shard trees under pytest temporary directories so malformed and missing files can be exercised without committing binary PNG files. The builder mirrors `descriptions.jsonl`, `sft.jsonl`, `patches/<sample_id>/meta.json`, and `artifacts/rgb/<sample_id>.png`.
