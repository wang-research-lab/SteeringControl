# 🎯 SteeringControl: Holistic Evaluation of Alignment Steering in LLMs

[![arXiv](https://img.shields.io/badge/📖_arXiv-2509.13450-b31b1b.svg)](https://arxiv.org/abs/2509.13450)
[![GitHub](https://img.shields.io/badge/GitHub-SteeringControl-blue.svg)](https://github.com/wang-research-lab/SteeringControl)
[![🤗 Hugging Face](https://img.shields.io/badge/🤗_HuggingFace-SteeringControl-yellow.svg)](https://huggingface.co/datasets/WangResearchLab/SteeringControl)

## Overview

While representation-based interventions have proven effective, currently there is a fundamental challenge: **specificity in intervention is underexplored**. 

To remedy this, we introduce **SteeringControl**: a comprehensive benchmark for evaluating representation steering methods across safety-relevant behaviors. 

Our focus is on **Behavioral Entanglement**, i.e., understanding how steering one behavior affects others, which is critical for safe deployment of steering interventions.

Without measuring entanglement, we risk introducing harmful side effects when attempting to improve model alignment.

We hope this benchmark will foster development of more precise steering methods and serve as a platform for introducing new approaches that achieve better effectiveness-entanglement tradeoffs.

### Key Contributions

- 📈 **Effectiveness vs. Entanglement** evaluation revealing critical tradeoffs and highlighting open challenges in steering specificity
- 📊 **17 datasets** collected and standardized covering 3 primary behaviors (harmful generation, demographic bias, hallucination) and 10 secondary behaviors for comprehensive entanglement evaluation
- 🔧 **Modular framework** decomposing training-free steering methods into standardized, interchangeable components

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/wang-research-lab/SteeringControl.git
cd SteeringControl

# Install dependencies
pip install -e .

# Run complete steering pipeline
python scripts/run/run_full_pipeline.py -m qwen25-7b -c refusal_base -M dim

# Run with different entanglement controls
python scripts/run/run_full_pipeline.py -m qwen25-7b -c refusal_base -M dim_nokl      # No KL constraint
python scripts/run/run_full_pipeline.py -m qwen25-7b -c refusal_base -M dim_conditional # With CAST
```

## 🧪 Running Experiments

### Parallel Execution
For comprehensive evaluation across multiple models, behaviors, and methods:

```bash
python scripts/run/run_parallel_experiments.py \
    --skip-metrics \
    --concepts explicit_bias implicit_bias hallucination_extrinsic refusal_base hallucination_intrinsic \
    --model llama3-1-8b qwen25-7b \
    --methods ace_conditional \
    --gpu 0 1 2 3
```

This runs experiments across specified concepts and models in parallel using multiple GPUs. Adjust the concept, model, and method lists as needed, as well as the GPU IDs based on your setup with CUDA_VISIBLE_DEVICES.

### Viewing Results

For analysis, run the following scripts to aggregate results and create visualizations as in the paper:

1. **Run full pipeline analysis**:
   ```bash
   python scripts/run/run_full_pipeline.py --metrics-only --experiments-dir experiments
   ```

   For specific directories:
   ```bash
   python scripts/run/run_concept_metrics.py \
       "experiments/lat_hallucination_intrinsic_llama3-1-8b/" \
       "experiments/baseline_llama3-1-8b/"
   ```
> **Note**: When looking at DarkBench results from experimental directories, the higher the metric the more unaligned the model is. To make it consistent with other datasets, you must do `1 - metric` to get the higher-is-better score.


2. **Generate analysis and plots**:
   ```bash
   python scripts/analysis/analyze_concept_metrics.py \
       --experiments-dir experiments \
       --output-prefix steering_analysis

   python scripts/analysis/generate_all_plots.py experiments/
   ```

## 🤖 Supported Models

Currently supported models:
- **Qwen2.5-7B**
- **Llama-3.1-8B**

Both models support chat templates and are compatible with all steering methods in our framework.

## 📚 Datasets

We evaluate on **17 datasets** across primary and secondary behaviors. See our [Hugging Face dataset](https://huggingface.co/datasets/WangResearchLab/SteeringControl) or [paper](https://arxiv.org/abs/2509.13450) for detailed descriptions.

### Primary Behaviors (Steering Targets)
- **Harmful Generation**: SALADBench (refusal behavior)
- **Demographic Bias**: BBQ (implicit), ToxiGen (explicit)
- **Hallucination**: FaithEval (intrinsic), PreciseWikiQA (extrinsic)

### Secondary Behaviors (Entanglement Measurement)
- **Social**: Sycophancy, Anthropomorphism, Brand Bias, User Retention
- **Reasoning**: Expert-level (GPQA), Commonsense (ARC-C)
- **Epistemic**: Factual Misconceptions (TruthfulQA), Sneaking
- **Normative**: Commonsense Morality, Political Views

## 🔧 Modular Framework

We decompose training-free steering methods into three phases:

### 1. Direction Generation
Extract steering vectors from training data:
- **Methods**: `DiffInMeans`, `PCA`, `LAT`
- **Formats**: `SteeringFormat.DEFAULT`, `SteeringFormat.REPE`, `SteeringFormat.CAA`

### 2. Direction Selection
Choose optimal layer and hyperparameters:
- **Grid Search**: Exhaustive search across the desired layers based on val score
- **COSMIC**: Efficient cosine similarity-based selection without full generation

### 3. Direction Application
Apply interventions during inference:
- **Activation Addition**: Add scaled direction to activations
- **Directional Ablation**: Remove projection along direction (with optional affine transformation)
- **Locations**: Where in the model to apply interventions (same layer as generation, all layers, cumulative across layers, etc.)
- **Positions**: `ALL_TOKENS`, `POST_INSTRUCTION`, `OUTPUT_ONLY`
- **Conditional (CAST)**: Apply only when activation similarity exceeds threshold

## 📋 Pre-configured Methods

We implement 5 methods from the literature, each with 3 variants for different effectiveness/entanglement tradeoffs:

All configurations can be found in the `configs/` directory with variants: `{method}.yaml`, `{method}_nokl.yaml`, `{method}_conditional.yaml`

| Method | Components | Paper | Implementation Notes |
|--------|------------|-------|----------|
| **DIM** | DiffInMeans + Directional Ablation | [Arditi et al.](https://arxiv.org/abs/2406.11717) + [COSMIC](https://arxiv.org/abs/2506.00085) | Original refusal steering method |
| **ACE** | DiffInMeans + Directional Ablation (affine) | [Marshall et al.](https://arxiv.org/abs/2411.09003) + [COSMIC](https://arxiv.org/abs/2506.00085) | Adds reference projection |
| **CAA** | DiffInMeans + Activation Addition (MC format) | [Panickssery et al.](https://arxiv.org/abs/2312.06681) | Uses multiple-choice format |
| **PCA** | PCA + Activation Addition | [Zou et al.](https://arxiv.org/abs/2310.01405) (RepE) + [CAST](https://arxiv.org/abs/2409.05907) + [AxBench](https://arxiv.org/abs/2501.17148) | Principal component analysis |
| **LAT** | LAT + Activation Addition (cumulative) | [Zou et al.](https://arxiv.org/abs/2310.01405) (RepE) + [AxBench](https://arxiv.org/abs/2501.17148) | Linear artificial tomography |

## ⚙️ Custom Configurations

Importantly, the above 5 methods are not exhaustive. Our modular framework allows easy creation of new methods by combining different components!

For example, to create a new method using LAT with CAA format, COSMIC selection, Directional Ablation application, and Conditional steering (CAST), with different layer and component choices than is used in the paper, simply create a new YAML config:

```yaml
# configs/custom.yaml - LAT + CAA format + COSMIC + Directional Ablation + Conditional

# Override dataset formatting to use CAA templates with LAT:
train_data:
  pos:
    params:
      format: SteeringFormat.CAA  # LAT with CAA format
  neg:
    params:
      format: SteeringFormat.CAA
  neutral: null

# Phase 1: Direction Generation
direction_generation:
  generator:
    class: direction_generation.linear.LAT
    params: {}
  param_grid:
    # Change for every middle layer and attn output component
    layer_pct_start: [0.3]
    layer_pct_end: [0.7]
    layer_step: [1]
    component: ['attn']
    attr: ['output']
    pos: [-1]
    ...

# Phase 2: Direction Selection
direction_selection:
  class: direction_selection.cosmic.COSMIC
  params:
    application_locations: []
    include_generation_loc: true
    generation_pos: POST_INSTRUCTION  # Targeted application
    use_kl_divergence_check: false
    ...

# Phase 3: Direction Application
direction_application:
  class: direction_application.unconditional.DirectionalAblation
  params:
    use_affine: false  # Pure directional ablation
    ...

# Enable conditional steering
conditional:
  enabled: true
  condition_selection:
    class: direction_selection.grid_search.ConditionalGridSearchSelector
    params:
      condition_thresholds: "auto"
      condition_comparators: ["greater"]
      ...
```

We also welcome contributions of new datasets, models, and components to further expand what can be evaluated.

## 📁 Repository Structure

```
SteeringControl/
├── configs/                      # Experiment configurations
│   ├── {method}.yaml            # Base configurations
│   ├── {method}_nokl.yaml       # No KL divergence check
│   └── {method}_conditional.yaml # With CAST
├── data/                        # Dataset loaders
│   ├── steering_data.py         # Main data interface
│   ├── refusal.py              # Harmful generation datasets
│   ├── bias.py                 # Demographic bias datasets
│   ├── hallucination.py        # Hallucination datasets
│   └── secondary_datasets.py   # Entanglement evaluation
├── direction_generation/        # Phase 1 components
│   ├── base.py
│   └── linear.py               # DiffInMeans, PCA, LAT
├── direction_selection/         # Phase 2 components
│   ├── base.py
│   ├── grid_search.py
│   └── cosmic.py
├── direction_application/      # Phase 3 components
│   ├── base.py
│   ├── unconditional.py       # Standard interventions
│   └── conditional.py         # CAST implementation
├── utils/                      # Utilities
│   ├── intervention_llm.py    # Model intervention code
│   ├── steering_utils.py      # Helper functions
│   └── enums.py              # Configuration enums
└── scripts/
    ├── run/                   # Experiment scripts
    └── analysis/              # Evaluation tools
```

## 🔬 Key Findings

Our evaluation reveals several critical insights about current steering methods:

1. **Pareto Frontiers**: Clear effectiveness-entanglement tradeoffs exist across all methods on both models evaluated, highlighting the fundamental challenge of precise intervention.

2. **Behavior-Specific Challenges**: Refusal is easiest to steer while extrinsic hallucination remains highly challenging. Significant room for improvement exists across all behaviors at improving effectiveness alone.

3. **Entanglement is Pervasive**: All methods exhibit behavioral entanglement, though it can be partially mitigated with conditional steering (CAST) and changes in KL hyperparameters. However, the fundamental challenge of achieving high effectiveness without entanglement remains an open problem.

4. **Safety Implications**: The presence of entanglement across social, reasoning, and normative behaviors underscores the critical need for comprehensive evaluation before deploying steering interventions in safety-critical applications.

**This represents a major open challenge in AI safety**: developing steering methods that can precisely target specific behaviors without unintended side effects. We hope this benchmark will accelerate progress toward more controllable and safer steering methods, and in the future, more generally towards safer and more aligned AI systems.

## 📝 Citation

```bibtex
@misc{siu2025steeringcontrol,
      title={SteeringControl: Holistic Evaluation of Alignment Steering in LLMs},
      author={Vincent Siu and Nicholas Crispino and David Park and Nathan W. Henry and Zhun Wang and Yang Liu and Dawn Song and Chenguang Wang},
      year={2025},
      eprint={2509.13450},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.13450},
}
```

## 📄 License

### Framework Code
The SteeringControl framework code is released under the **MIT License**.

### Datasets
This benchmark incorporates multiple existing datasets, each with their own licensing terms. For some datasets (e.g., HalluLens), we also utilize their evaluation code and metrics. Users must respect the individual licenses of constituent datasets:

| Dataset | License | Source |
|---------|---------|--------|
| **ARC-C** | CC-BY-SA-4.0 | [AllenAI](https://huggingface.co/datasets/allenai/ai2_arc) |
| **Alpaca** | CC-BY-NC-4.0 | [Stanford](https://huggingface.co/datasets/tatsu-lab/alpaca) |
| **BBQ** | CC-BY-4.0 | [NYU-MLL](https://github.com/nyu-mll/BBQ) |
| **CMTest** | CC-BY-SA-4.0 | [AI-Secure](https://huggingface.co/datasets/AI-Secure/DecodingTrust) |
| **DarkBench** | MIT | [Apart Research](https://huggingface.co/datasets/apart/darkbench) |
| **FaithEval** | See source* | [Salesforce](https://github.com/SalesforceAIResearch/FaithEval) |
| **GPQA** | CC-BY-4.0 | [Rein et al.](https://huggingface.co/datasets/Idavidrein/gpqa) |
| **HalluLens** | CC-BY-NC** | [Meta](https://github.com/facebookresearch/HalluLens) |
| **SALADBench** | Apache-2.0 | [OpenSafetyLab](https://github.com/OpenSafetyLab/SALAD-BENCH) |
| **ToxiGen** | See source* | [Microsoft](https://github.com/microsoft/TOXIGEN) |
| **TruthfulQA** | See source* | [Lin et al.](https://github.com/sylinrl/TruthfulQA) |
| **TwinViews** | CC-BY-4.0 | [Fulay et al.](https://huggingface.co/datasets/wwbrannon/twinviews-13k) |

*Datasets marked with asterisk seem to have no explicit dataset license but their associated codebases are licensed (Apache-2.0, MIT, etc.). Please refer to original sources for usage terms. HalluLens is mostly CC-BY-NC but contains some components with other licenses.*

## 🙏 Acknowledgments

We gratefully acknowledge the following for helpful resources and foundational work:
- [AxBench](https://arxiv.org/abs/2501.17148)
- [CAST](https://arxiv.org/abs/2409.05907)
- [DIM](https://arxiv.org/abs/2406.11717)
- [COSMIC](https://arxiv.org/abs/2506.00085)
- [RepE](https://arxiv.org/abs/2310.01405)
- [CAA](https://arxiv.org/abs/2312.06681)
- *and many others!*
