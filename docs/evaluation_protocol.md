# Feasibility Evaluation Protocol

Evaluation has two deliberately separate layers. Automated diagnostics test observable structural properties; human reviewers assign the seven subjective ratings. The automated code never manufactures correctness, grounding, hallucination, diversity, visual-dependence, usefulness, or specificity scores.

## Deterministic gate

`python -m hsi_vqagen.evaluation.checks` reads each configuration's `normalized.jsonl`. It verifies the required configuration × sample matrix, four indexed pairs per cell, non-empty fields, and reports duplicate/high-overlap questions. It also warns when a spectral term is absent from the paired description and when a description-only output claims direct image access. These warnings identify review targets; they are not proof that an answer is wrong.

Smoke runs require one sample in all requested configurations and four valid rows per cell. The full gate requires the immutable five sample IDs and all available C01–C11 configurations. C10/C11 compatibility failures remain explicit missing cells rather than fabricated outputs.

## Human review

The worksheet uses deterministic blind IDs and omits model/configuration names from the reviewer-facing table. Reviewers fill blank fields for correctness, grounding, hallucination, diversity, visual dependence, usefulness, and specificity, plus optional notes. Ratings must not be populated until a person has inspected the relevant evidence.

## Controlled ablations

Rows are aligned exactly by `(sample_id, pair_index)` for C01/C10, C02/C11, C03/C09, and C04/C06. C03/C09 is the primary preregistered Gemma-4-12B comparison. The analysis code reports only observable overlap and length deltas automatically. The following interpretations remain blank for human review: description-absent visual detail, paraphrase tendency, hallucination direction, diversity change, and specificity change.

Pair index alignment supports presentation but does not imply that two independently generated questions express the same intent. Reviewers should compare each sample as a set as well as inspect the aligned rows.
