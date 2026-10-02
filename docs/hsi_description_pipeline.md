# Verified HSI Description-Generation Pipeline

This document records the realized pipeline behind the audited 10,000 paired descriptions. It is based on the production batch implementation in `/data/jypark/CLI_Descriptive_System`, the generated per-patch metadata, and the legacy projects used only as read-only algorithm references. The production path takes precedence where comments or legacy capabilities differ from executable behavior.

The recorded prompt set is `core-config-prompts@2026-07-30` with hash `0e88f131955c0f41`. The five locked feasibility samples all report the same enabled-agent flags, prompt metadata, model roles, and call-wave structure.

## Input and preprocessing

Each audited patch contains a georeferenced hyperspectral cube with shape 64 × 64 × 426. For the locked samples, wavelength metadata spans approximately 381.54–2509.93 nm and the recorded valid-pixel ratio is 1.0. The production runner loads a patch, evaluates its valid-pixel ratio, and applies a configurable minimum threshold of 0.5 before invoking the analysis pipeline.

The description system receives the HSI cube, wavelength metadata, no-data information, and geospatial metadata. It then requests automatic spectral-index selection, RGB attachment, saved maps, external context, and batch-agent execution. These raw inputs belong to the upstream description-generation process: raw HSI cubes are not inputs to the VQA-generation models in this study.

## RGB and spectral-index evidence

The production imaging code renders an HSI-derived RGB view and upscales it to 384 × 384 pixels. This paired RGB image is the visual input used only in the multimodal VQA condition.

The analysis pipeline automatically selects spectral indices and renders their spatial maps. The index-map renderer supports category-specific colour maps and a monochrome convention in which low values are white and high values are black; no-data pixels are rendered magenta. Category agents receive the RGB view plus the maps assigned to their category. Thus, an agent's statement can be grounded in spatial evidence derived from bands beyond the three-band RGB composite even though the downstream VQA models do not receive those maps directly.

The active cluster path uses spectral-angle distance with K-medoids. It selects the number of clusters with an elbow procedure, computes cluster proportions and mean spectra, and prepares a cluster map and spectral summary for the cluster agent. A stale pipeline comment refers to PCA/K-means, but the invoked implementation is the spectral-angle K-medoids path described here.

## Agent workflow

The realized orchestration comprises a real-time web-context side channel followed by three dependency-ordered batch waves:

1. `web_research` gathers coarse external context in real time with `gpt-5.4-mini`.
2. Wave 1 runs category-specific catalog agents and the cluster agent with `gpt-5.4-mini`. Catalog agents inspect the RGB image and their assigned index maps. The cluster agent receives the RGB image, cluster map, mean spectra, area proportions, and any candidate library matches exposed by the pipeline.
3. Wave 2 runs catalog synthesis with `gpt-5.4`, consolidating category-agent evidence.
4. Wave 3 runs the final reduce stage with `gpt-5.4`, combining the RGB image, catalog synthesis, cluster analysis, and available web context.

The source metadata labels catalog and cluster calls as wave 1, catalog synthesis as wave 2, and reduce as wave 3. Web research is kept separate because it is issued as the real-time side channel rather than as a batch wave. This distinction is preserved in `artifacts/hsi_description_pipeline_evidence.json`.

## Description synthesis and output

The reduce prompt requests a single coherent Korean prose description. Its direct visual observation is the rendered RGB image; the richer hyperspectral evidence reaches it through textual outputs from the index-map and cluster analyses. The finalizer writes per-run metadata and aggregates results into `descriptions.jsonl` and `sft.jsonl`, together with a summary artifact.

For the present VQA study, the paired description is used unchanged as the common semantic input. Multimodal conditions receive that description plus the HSI-derived RGB image; text-only conditions receive the description and no image. VQA outputs therefore do not constitute new HSI measurements.

## Disabled components

The production flags and audited metadata show that pixel-level supervised SAM observation was disabled for the 10,000-description run. This optional observation agent is not part of the realized evidence path.

Likewise, Tetracorder material identification was disabled. Candidate material names that may appear in cluster context must not be described as confirmed Tetracorder identifications.

## Limitations

- A three-channel RGB rendering cannot preserve all information in a 426-band HSI cube. Visual details available to a VQA model are therefore a lossy projection of the hyperspectral measurement.
- Descriptions are model-synthesized interpretations, not direct ground-truth labels. The realized upstream workflow used `gpt-5.4-mini` for catalog, cluster, and web roles and `gpt-5.4` for catalog synthesis and reduction.
- External web context is coarse contextual evidence and does not verify pixel-level material identity.
- Spectral-library candidates and cluster summaries are analytical cues, not definitive material confirmation.
- The feasibility comparison measures VQA generation conditioned on an RGB representation and/or the paired description; it does not measure a model's ability to ingest a raw HSI cube.

## Traceability

Every methodological statement category above is mapped to a production source symbol or generated artifact field in `artifacts/hsi_description_pipeline_evidence.json`. Absolute legacy and dataset locations are intentionally excluded from that portable evidence map; the source-root labels `production`, `legacy`, and `dataset` identify the corresponding read-only trees.
