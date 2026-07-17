<div align="center">

# IMUG-Bench: Benchmarking Unified Multimodal Models on Interleaved Understanding and Generation

<p align="center">
  <strong>
    Lingyi Meng<sup>&ast;,1</sup>,
    Zecong Tang<sup>&ast;,&dagger;,1</sup>,
    Haoran Li<sup>&ast;,1</sup>,
    Tengju Ru<sup>&ast;,1</sup>,
    Zhejun Cui<sup>1</sup>,
    Weitong Lian<sup>1</sup>,
    Qi Kang<sup>1</sup>,
    Hangshuo Cao<sup>1</sup>,
    Yichen Zhu<sup>1</sup>,
    Yechi Liu<sup>3</sup>,
    Kaixuan Wang<sup>2</sup>,
    Yu-Jie Yuan<sup>4</sup>,
    Chunwei Wang<sup>4</sup>,
    Yu Zhang<sup>&Dagger;,1</sup>,
    Bo Dai<sup>&Dagger;,2</sup>
  </strong>
</p>


<sup>1</sup>Zhejiang University &nbsp; <sup>2</sup>The University of Hong Kong &nbsp; <sup>3</sup>Institute of Automation, Chinese Academy of Sciences &nbsp; <sup>4</sup>Huawei

<sup>*</sup>Equal contribution &nbsp; <sup>†</sup>Project leader &nbsp; <sup>‡</sup>Corresponding author

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2606.09169-B31B1B.svg)](https://arxiv.org/abs/2606.09169)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-IMUG--Bench-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ccccEsion/IMUG-Bench)

[Paper](https://arxiv.org/abs/2606.09169) | [Dataset](#dataset) | [Evaluation](#model-evaluation) | [Scoring](#quickstart) | [Results](#results--analysis)

</div>

<p align="center">
  <img src="assets/overview.png" alt="Overview of IMUG-Bench" width="100%">
</p>

## Overview

IMUG-Bench is a benchmark for **interleaved multimodal understanding and generation** in multi-turn image-text dialogue. It evaluates whether unified multimodal models can answer questions, generate or edit images, preserve relevant visual context, and reason about model-specific outcomes as a conversation evolves.

Unlike benchmarks that evaluate understanding and generation separately or rely on static, single-turn questions, IMUG-Bench evaluates both capabilities in a shared multi-turn protocol. It includes dynamic questions whose ground-truth answers depend on the evaluated model's own earlier outputs.

### Key Facts

- **Scale:** 3,113 samples and 12,034 interaction turns.
- **Coverage:** 19 domains and 97 fine-grained tasks.
- **Dialogue length:** 2 to 6 interleaved text and image output turns per sample.
- **Task types:** static MCQ, dynamic MCQ, and image generation or editing.
- **Score analysis:** all modalities, text-only, image-only, domain, class, and turn-wise results.

### Contributions

- We introduce a large-scale benchmark for jointly evaluating understanding and generation in multi-turn interleaved dialogue.
- We include dynamic MCQs that resolve their reference answers from a model's earlier generated content, rather than treating every answer as fixed.
- We provide point-based image evaluation with historical visual references for instruction following and cross-turn consistency.
- We characterize generation-side exposure bias in longer interactions and study test-time scaling strategies for mitigating it.

## Benchmark Structure

IMUG-Bench organizes multi-turn image-text dialogue into three complementary task classes.

### Task Classes

- **Static Spatial:** evaluates target-element attributes, ownership, and spatial relations.
- **Temporal Causal:** evaluates implicit natural laws and common-sense causal reasoning in context.
- **Hybrid:** combines spatial and causal requirements in diverse everyday scenarios.

### Interleaved Task Protocol

Each sample contains an ordered sequence of turns. At every turn, the model must return either text or an image; subsequent turns may depend on preceding inputs and model outputs.

| Task Type | Model Output | Evaluation |
| --- | --- | --- |
| Static MCQ | Answer letters | Direct answer and format scoring |
| Dynamic MCQ | Answer letters | A judge first resolves the context-dependent reference answer from referenced prior turns, then applies MCQ scoring |
| Image Generation | Generated or edited image | A judge scores each evaluation point from 0 to 5, using historical image references when needed |

For an image turn with $N$ evaluation points, the normalized turn score is the mean point score divided by 5. Text scores and image scores are both converted to percentages in the final summary.

## Dataset

The released data contains benchmark metadata, input images, reference links for image and dynamic evaluation, and grid ground truth for deterministic geometric-coloring scoring.

### Repository Layout

```text
data/
  benchmark.jsonl             # Benchmark samples and turn metadata
  images/                     # Input images
  image_references.jsonl      # Reference-image links for image scoring
  dynamic_references.jsonl    # Referenced turns for dynamic MCQ resolution
  grid_gt/                    # Ground truth for geometric coloring
config/
  domain_classes.json         # Paper-defined class-to-domain mapping
scripts/
  run_evaluation.py          # Unified client for model evaluation
  run_scoring.py              # Interactive end-to-end scoring runner
  model_servers/              # Reference model server implementations
    BAGEL_demo.py
    BLIP3o_demo.py
  auxiliary/
    common.py                 # Shared path, validation, and I/O helpers
    resolve_dynamic_answers.py # Dynamic MCQ answer resolution
    score_image.py            # Judge-based image scoring
    score_grid.py             # Deterministic geometric-coloring scoring
    score_mcq.py              # Static and dynamic MCQ scoring
    summarize_results.py      # Percentage-based result summaries
    dynamic_prompt.txt        # Dynamic-answer judge prompt
    eval_prompt.txt           # Image-scoring judge prompt
```

### Data Format

`data/benchmark.jsonl` is JSONL. Every record contains a domain, subdomain, sample identifier, and ordered `tasks`. Each task contains its `turn`, expected output `modality`, inputs, and either an MCQ answer or image evaluation points.

Model outputs must follow this layout:

```text
outputs/model_outputs/<model>/<domain>/<subdomain>/question_<sample_id>/
  result.json
  turn_<turn>_output.png
```

`result.json` stores per-turn text responses in its `tasks` list. Image turns are read from `turn_<turn>_output.png`; image filenames referenced from `result.json` are also supported by the image scorer.

Scoring artifacts are separated by evaluated model:

```text
outputs/<model>/
  dynamic_answer/<model>_dynamic_answer.jsonl
  score_img/<model>_img_score.jsonl
  score_mcq/<model>_mcq_score.jsonl
  summary/
```

The supporting scorers, prompts, and shared helpers are kept under `scripts/auxiliary/`; they are normally invoked through `run_scoring.py`.

## Model Evaluation

The unified evaluation client sends each sample to a model server one turn at a time. It preserves the ordered multimodal history and expects the server to expose the `/infer` endpoint with the following response format:

```json
{"response": {"text": "..."}}
```

For image turns, the response uses a base64-encoded PNG instead:

```json
{"response": {"image": "<BASE64_PNG>"}}
```

Start one of the server demonstrations under `scripts/model_servers/`, or provide your own server implementing the same protocol. Model-specific dependencies and checkpoints are not included in this repository.

Example server commands:

```bash
IMUG_MODEL_PATH=/path/to/model \
python scripts/model_servers/BAGEL_demo.py --port 8000
```

```bash
IMUG_MODEL_PATH=/path/to/model \
IMUG_PROCESSOR_PATH=/path/to/processor \
python scripts/model_servers/BLIP3o_demo.py --port 8000
```

Run evaluation with the model name used in the output directory:

```bash
python scripts/run_evaluation.py \
  --model <model> \
  --api-url http://127.0.0.1:8000/infer
```

Repeat `--api-url` to distribute requests across multiple compatible server workers. Evaluation results are written to `outputs/model_outputs/<model>/` and can then be processed by `run_scoring.py`.

## Results & Analysis

The benchmark supports separate analysis of understanding and generation, together with domain-level and turn-wise reporting. The paper finds that image-generation scores can decline as dialogue context grows, exposing the difficulty of maintaining requirements and visual consistency across turns.

<p align="center">
  <img src="assets/exposure_bias.svg" alt="Image-modality score trends and a multi-turn example" width="100%">
</p>

<p align="center"><em>Image-modality trends across dialogue turns and a representative multi-turn interaction.</em></p>

IMUG-Bench spans three classes and 19 domains. The benchmark composition is summarized below; detailed per-domain, per-class, and per-turn reports are produced by `summarize_results.py`.

<p align="center">
  <img src="assets/distribution.svg" alt="IMUG-Bench class and domain distribution" width="82%">
</p>

<p align="center"><em>Class and domain composition of IMUG-Bench.</em></p>

## Quickstart

### 1. Start a Model Server

`run_evaluation.py` is a model-independent client. Before running it, start a model-side server that implements the IMUG `/infer` protocol described in [Model Evaluation](#model-evaluation).

Two reference server demos are included:

```text
scripts/model_servers/
  BAGEL_demo.py
  BLIP3o_demo.py
```

They show how to load a specific model, convert the ordered IMUG conversation history into that model's input format, and return either text or a base64-encoded image. They are reference implementations rather than required model backends.

Start the BAGEL demo:

```bash
IMUG_MODEL_PATH=/path/to/BAGEL \
python scripts/model_servers/BAGEL_demo.py --port 8000
```

Start the BLIP3-o demo:

```bash
IMUG_MODEL_PATH=/path/to/BLIP3o \
IMUG_PROCESSOR_PATH=/path/to/processor \
python scripts/model_servers/BLIP3o_demo.py --port 8000
```

### 2. Run Model Evaluation

The evaluation client reads the benchmark turn by turn, preserves the ordered `system → user → assistant` multimodal history, sends the complete history to the model server, and saves text and image outputs in the standard directory layout.

A typical command is:

```bash
python scripts/run_evaluation.py \
  --model MODEL_NAME \
  --api-url http://127.0.0.1:8000/infer \
  --benchmark data/benchmark.jsonl \
  --images data/images \
  --model-output-dir outputs/model_outputs
```

Evaluation outputs are written as:

```text
outputs/model_outputs/<model>/<domain>/<subdomain>/question_<sample_id>/
  result.json
  turn_<turn>_output.png
```

### 3. Run Scoring

Place the evaluated model outputs under `outputs/model_outputs/`, then run:

```bash
python scripts/run_scoring.py --models <model>
```

Use `--model <model>` for a single model. If the selected path directly contains one model's domain folders, specify it with `--model-output-dir`:

```bash
python scripts/run_scoring.py \
  --model MODEL_NAME \
  --model-output-dir /path/to/MODEL_NAME_outputs
```

On first use, the runner asks for the judge API base URL, API key, judge model, and model-output directory. The local settings are stored in `config/local_config.json`, which is excluded from version control. Use the following command to enter them again:

```bash
python scripts/run_scoring.py --reconfigure
```

The scoring workflow applies the following steps:

1. **Dynamic MCQ:** a judge resolves the context-dependent reference answer from the referenced earlier model outputs.
2. **Image generation:** a judge assigns a score from 0 to 5 to each evaluation point; the turn score is the mean point score divided by 5.
3. **Grid domain:** `score_grid.py` performs separate rule-based scoring against the grid ground truth. This deterministic scorer is used because it is more reliable for geometric-coloring outputs than a general judge model.
4. **MCQ scoring:** static answers and resolved dynamic answers are scored by matching option letters, with an additional output-format weight.

The original model outputs remain unchanged. Intermediate and scoring artifacts are stored by model:

```text
outputs/<model>/
  dynamic_answer/<model>_dynamic_answer.jsonl
  score_img/<model>_img_score.jsonl
  score_mcq/<model>_mcq_score.jsonl
  summary/
```

#### Scoring Smoke Test

The smoke test checks whether the output paths, result files, referenced images, and scoring pipeline can be read correctly **without connecting to a real judge model**. It starts a temporary local random judge, so its numerical scores are not meaningful; use it only to inspect pipeline and path correctness.

```bash
python scripts/run_scoring.py \
  --smoke-test \
  --models <model> \
  --seed 2026
```

Smoke-test artifacts are written separately to:

```text
outputs/smoke_test/<model>/
```

Use `--smoke-port <port>` to choose the local mock-judge port. The default value `0` automatically selects an available port.

For a model directory outside the default output root:

```bash
python scripts/run_scoring.py \
  --smoke-test \
  --model MODEL_NAME \
  --model-output-dir /path/to/MODEL_NAME_outputs
```

### 4. Summarize Results

```bash
python scripts/auxiliary/summarize_results.py --models <model>
```

For one model, the summary is written to:

```text
outputs/<model>/summary/summary.md
outputs/<model>/summary/summary.json
```

When multiple models are summarized together, the combined files are written to `outputs/summary/`. The default class mapping is loaded from `config/domain_classes.json`; use `--class-config` only when providing a custom mapping:

```bash
python scripts/auxiliary/summarize_results.py \
  --models <model> \
  --class-config config/domain_classes.json
```
## Citation

```bibtex
@article{meng2026imugbench,
  title={IMUG-Bench: Benchmarking Unified Multimodal Models on Interleaved Understanding and Generation},
  author={Meng, Lingyi and Tang, Zecong and Li, Haoran and Ru, Tengju and Cui, Zhejun and others},
  journal={arXiv preprint arXiv:2606.09169},
  year={2026}
}
```

## License

The benchmark text, annotations, metadata, photographs, and AI-generated images **created by the IMUG-Bench authors** are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Third-party images were collected from publicly available sources under licenses permitting their use in derivative research materials at the time of collection. These images are included solely as integral components of the annotated IMUG-Bench evaluation tasks and are not relicensed by the IMUG-Bench authors.

Users should not extract or redistribute individual third-party images separately from the benchmark.

Because the repository contains materials subject to different licensing terms, its Hugging Face metadata is marked as `license: other`.

## Contact

Please open a GitHub issue for questions about the benchmark or evaluation code.
