<div align="center">

[English](README.md) | [简体中文](README_zh.md)

# IMUG-Bench：统一多模态模型交错式理解与生成评测基准

**Lingyi Meng<sup>*1</sup>, Zecong Tang<sup>*†1</sup>, Haoran Li<sup>*1</sup>, Tengju Ru<sup>*1</sup>, Zhejun Cui<sup>1</sup>, Weitong Lian<sup>1</sup>, Qi Kang<sup>1</sup>, Hangshuo Cao<sup>1</sup>, Yichen Zhu<sup>1</sup>, Yechi Liu<sup>3</sup>, Kaixuan Wang<sup>2</sup>, Yu-Jie Yuan<sup>4</sup>, Chunwei Wang<sup>4</sup>, Yu Zhang<sup>‡1</sup>, Bo Dai<sup>‡2</sup>**

<sup>1</sup>浙江大学 &nbsp; <sup>2</sup>香港大学 &nbsp; <sup>3</sup>中国科学院自动化研究所 &nbsp; <sup>4</sup>华为

<sup>*</sup>共同一作 &nbsp; <sup>†</sup>项目负责人 &nbsp; <sup>‡</sup>通讯作者

[![论文](https://img.shields.io/badge/Paper-arXiv%3A2606.09169-B31B1B.svg)](https://arxiv.org/abs/2606.09169)
[![状态](https://img.shields.io/badge/status-public%20release%20in%20progress-blue)](https://github.com/ccccEsion/IMUG-Bench)
[![许可证](https://img.shields.io/badge/license-pending-lightgrey)](#许可证)

[论文](https://arxiv.org/abs/2606.09169) | [数据集](#数据集) | [评估](#快速开始) | [结果与分析](#结果与分析)

</div>

<p align="center">
  <img src="assets/overview.png" alt="IMUG-Bench 总览" width="100%">
</p>

## 概览

IMUG-Bench 面向多轮图文交错对话，系统评估统一多模态模型的理解与生成能力。它关注模型能否在连续交互中回答问题、生成或编辑图像、保留有效的视觉上下文，并基于自身先前的输出完成后续推理。

不同于将理解与生成分开评测、或仅采用静态单轮问题的既有基准，IMUG-Bench 在同一套多轮交互协议中考察两种能力。基准中的动态问题没有预设的固定答案，其标准答案由被测模型在此前轮次实际生成的内容决定。

### 核心事实

- **规模：** 共 3,113 个样本、12,034 个交互轮次。
- **覆盖范围：** 19 个领域和 97 个细粒度任务。
- **对话长度：** 每个样本包含 2 至 6 轮交错的文本或图像输出。
- **任务类型：** 静态选择题、动态选择题，以及图像生成或编辑。
- **结果统计：** 支持全模态、纯文本、纯图像，以及领域、任务类别和轮次维度的分析。

### 主要贡献

- 提出大规模多轮交错式评测基准，在同一对话中联合衡量模型的理解与生成能力。
- 设计动态选择题：参考答案依据模型此前生成的实际内容解析，而非对所有样本预先固定。
- 提供基于评测点的图像评估，并利用历史图像参考检查指令遵循情况和跨轮视觉一致性。
- 分析长链交互中的生成端暴露偏差（exposure bias），并研究相应的测试时扩展策略。

## 基准结构

IMUG-Bench 按任务内容将多轮图文对话划分为三个互补的类别。

### 任务类别

- **静态空间（Static Spatial）：** 考察模型对目标元素属性、归属关系和空间关系的理解与生成能力。
- **时序因果（Temporal Causal）：** 考察模型在给定情境下对隐含自然规律和常识性因果关系的推理能力。
- **混合（Hybrid）：** 在更丰富的日常场景中，同时考察空间与因果相关能力。

### 交错式任务协议

每个样本都是一个按顺序展开的多轮交互。在每一轮中，模型需要输出文本或图像；后续轮次可能会引用此前的输入或模型输出。

| 任务类型 | 模型输出 | 评估方式 |
| --- | --- | --- |
| 静态选择题 | 选项字母 | 直接核对答案，并检查输出格式 |
| 动态选择题 | 选项字母 | 裁判模型先依据被引用的历史轮次解析上下文相关的参考答案，再进行选择题评分 |
| 图像生成 | 生成或编辑后的图像 | 裁判模型在需要时结合历史图像参考，对每个评测点打 0 至 5 分 |

对于包含 $N$ 个评测点的图像轮次，先计算各评测点的平均分，再除以 5 得到标准化得分。最终报告会将文本和图像得分统一换算为百分制。

## 数据集

本次发布包含基准元数据、输入图像、图像与动态评测所需的参考信息，以及几何着色任务进行确定性评分所需的网格标注。

### 仓库结构

```text
data/
  benchmark.jsonl             # 基准样本及轮次元数据
  images/                     # 输入图像
  image_references.jsonl      # 图像评分所需的参考图像信息
  dynamic_references.jsonl    # 动态选择题解析所需的历史轮次
  grid_gt/                    # 几何着色任务的网格标注
config/
  domain_classes.json         # 论文定义的任务类别与领域映射
scripts/
  run_evaluation.py           # 交互式端到端评测运行器
  auxiliary/
    common.py                 # 共享的路径、校验和 I/O 工具
    resolve_dynamic_answers.py # 动态选择题答案解析
    score_image.py            # 基于裁判模型的图像评分
    score_grid.py             # 确定性的几何着色评分
    score_mcq.py              # 静态和动态选择题评分
    summarize_results.py      # 百分制结果汇总
    dynamic_prompt.txt        # 动态答案解析提示词
    eval_prompt.txt           # 图像评分提示词
```

### 数据格式

`data/benchmark.jsonl` 采用 JSONL 格式。每条记录包含领域、子领域、样本标识符和有序的 `tasks` 列表。每个任务包含轮次 `turn`、预期输出模态 `modality`、输入信息，以及选择题答案或图像评测点。

被测模型的输出应遵循以下目录结构：

```text
outputs/model_outputs/<model>/<domain>/<subdomain>/question_<sample_id>/
  result.json
  turn_<turn>_output.png
```

`result.json` 的 `tasks` 列表保存各轮文本响应。图像轮次默认读取 `turn_<turn>_output.png`；图像评分脚本也支持使用 `result.json` 中记录的图像文件名。

各被测模型的评测产物会分目录保存，避免多模型实验相互混淆：

```text
outputs/<model>/
  dynamic_answer/<model>_dynamic_answer.jsonl
  score_img/<model>_img_score.jsonl
  score_mcq/<model>_mcq_score.jsonl
  summary/
```

评分脚本、提示词文件和公共工具统一放在 `scripts/auxiliary/` 中，通常由 `run_evaluation.py` 自动调度。

## 结果与分析

该基准可以分别统计模型的理解与生成表现，并支持领域、任务类别和轮次级别的分析。论文结果显示，随着对话上下文变长，图像生成得分可能下降，反映出模型在多轮交互中持续满足要求、保持视觉一致性仍然存在困难。

<p align="center">
  <img src="assets/exposure_bias.svg" alt="图像模态得分趋势和多轮交互示例" width="100%">
</p>

<p align="center"><em>图像模态得分随交互轮次变化的趋势，以及一个多轮交互示例。</em></p>

IMUG-Bench 包含 3 个任务类别和 19 个领域。下图展示了基准构成；`summarize_results.py` 可生成详细的领域、任务类别和轮次报告。

<p align="center">
  <img src="assets/distribution.svg" alt="IMUG-Bench 的任务类别与领域分布" width="82%">
</p>

<p align="center"><em>IMUG-Bench 的任务类别与领域构成。</em></p>

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整评分流程

将一个或多个模型的输出目录放入 `outputs/model_outputs/` 后，运行：

```bash
python scripts/run_evaluation.py
```

首次运行时，运行器会询问裁判服务的 API 地址、API Key、裁判模型名称和模型输出目录。这些信息仅保存在 `config/local_config.json`，该文件不会被纳入版本控制。整个流程会依次完成动态答案解析、图像评分、确定性网格评分和选择题评分。

如果 `--model-output-dir` 本身就是某个模型的根目录，且其下直接包含各领域文件夹，可通过 `--model` 指定模型名称。运行器会创建临时别名，原始模型输出不会被改动。

更新本地配置：

```bash
python scripts/run_evaluation.py --reconfigure
```

### 3. 运行离线流程检查

离线流程冒烟测试（smoke test）不需要 API 密钥。运行器会先检查指定的模型输出目录，再启动一个临时的本地 OpenAI 兼容随机裁判服务，并通过标准的动态答案解析和图像评分客户端完成流程验证。结果会写入 `outputs/smoke_test/<model>/`。

```bash
python scripts/run_evaluation.py --smoke-test --models <model> --seed 2026
```

可使用 `--smoke-port <port>` 指定本地服务端口；默认值 `0` 会自动选择可用端口。

当模型输出目录本身就是模型根目录时，可按以下方式运行：

```bash
python scripts/run_evaluation.py \
  --smoke-test \
  --model bageltest \
  --model-output-dir ../BAGEL/
```

### 4. 汇总结果

```bash
python scripts/auxiliary/summarize_results.py --models <model>
```

单个模型的汇总报告默认写入 `outputs/<model>/summary/summary.md` 和 `outputs/<model>/summary/summary.json`；同时汇总多个模型时，结果写入 `outputs/summary/`。仓库中的 `config/domain_classes.json` 已包含论文定义的任务类别映射，会被自动加载。仅在使用自定义映射时，才需要通过 `--class-config` 指定文件：

```bash
python scripts/auxiliary/summarize_results.py \
  --models <model> \
  --class-config config/domain_classes.json
```

## 引用

```bibtex
@article{meng2026imugbench,
  title={IMUG-Bench: Benchmarking Unified Multimodal Models on Interleaved Understanding and Generation},
  author={Meng, Lingyi and Tang, Zecong and Li, Haoran and Ru, Tengju and Cui, Zhejun and others},
  journal={arXiv preprint arXiv:2606.09169},
  year={2026}
}
```

## 许可证

代码和数据的许可证将在公开发布前最终确定。

## 联系方式

如有关于基准或评测代码的问题，欢迎通过 GitHub Issue 交流。
