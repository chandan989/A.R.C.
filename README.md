<div align="center">

# 🏛️ A.R.C.

### **Adaptive Reasoning Chain**

*An Autonomous Multi-Agent System for Solving IMO-Level Mathematics*

[![Competition](https://img.shields.io/badge/Kaggle-AIMO_Progress_Prize_3-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
[![Deadline](https://img.shields.io/badge/Deadline-April_15,_2026-FF4444?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)]()

---

**A.R.C.** is our competition entry for the [AI Mathematical Olympiad – Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3).
The system chains frontier open-weight reasoning models with a sandboxed Python executor, generative solution selection, and adaptive test-time compute allocation to solve **110 original olympiad problems** across Algebra, Combinatorics, Geometry, and Number Theory — all within a **5-hour H100 GPU window with no internet access**.

</div>

---

## Table of Contents

1.  [Competition Overview](#1-competition-overview)
2.  [Competitive Landscape](#2-competitive-landscape--what-others-are-doing)
3.  [System Architecture](#3-system-architecture--how-arc-works)
4.  [Model Selection & Rationale](#4-model-selection--rationale)
5.  [Core Strategy: Tool-Integrated Reasoning (TIR)](#5-core-strategy-tool-integrated-reasoning-tir)
6.  [Test-Time Compute: Inference Scaling](#6-test-time-compute-inference-scaling)
7.  [Solution Selection: Beyond Majority Voting](#7-solution-selection-beyond-majority-voting)
8.  [Fine-Tuning Pipeline](#8-fine-tuning-pipeline)
9.  [Inference Optimization for Kaggle](#9-inference-optimization-for-kaggle)
10. [Execution Plan & Timeline](#10-execution-plan--timeline)
11. [Repository Structure](#11-repository-structure)
12. [References](#12-references)

### 📂 Additional Documentation

| Document | Description |
|---|---|
| [`docs/competitor-analysis.md`](docs/competitor-analysis.md) | Deep-dive into existing competitor submissions and their approaches |
| [`docs/suggested-models-and-approaches.md`](docs/suggested-models-and-approaches.md) | Comprehensive guide to recommended models, techniques, and our optimal strategy |

---

## 1. Competition Overview

The **AIMO Progress Prize 3** is the third iteration of XTX Markets' $10M grand challenge to build an AI that can achieve gold-medal performance at the International Mathematical Olympiad.

| Parameter | Detail |
|---|---|
| **Problems** | 110 original, never-before-published problems (Algebra, Combinatorics, Geometry, Number Theory) |
| **Difficulty** | National Olympiad → International Mathematical Olympiad (IMO) |
| **Answer Format** | Non-negative integer in range `[0, 99999]` (guessing is infeasible) |
| **Evaluation** | 50 public problems (live leaderboard) + 50 private problems (post-competition rerun) |
| **Progress Prize Threshold** | ≥ 47/50 on **both** public and private sets |
| **Compute** | Kaggle H100 GPU • 5-hour runtime • **No Internet** |
| **Submission** | Code competition — solutions generated via Kaggle's Python evaluation API |
| **Model Cutoff** | **Only models released before March 15, 2026** may be used at runtime |
| **Deadline** | **April 15, 2026** |

### Prize Categories

| Prize | Criteria |
|---|---|
| 🥇 **Overall Progress Prize** | Highest score ≥ 47/50 on both test sets |
| ⏱️ **Longest Leader Prize** | Longest cumulative time at #1 on the leaderboard |
| 📊 **Math Corpus Prize** | Most valuable public dataset contributed |
| 📝 **Write-up Prizes** | Best technical explanations of approach |
| 🧩 **Hardest Problem Prize** | Best performance on the least-solved problem |

---

## 2. Competitive Landscape — What Others Are Doing

> **Full analysis:** See [`docs/competitor-analysis.md`](docs/competitor-analysis.md) for detailed breakdowns of every submission.

As of April 6, 2026, the competition is **wide open** — the prize threshold of 47/50 has not yet been hit. Here's the real picture:

### Live Leaderboard (Actual Scores — Not Just Public Notebooks)

| # | Team | Score | Entries | Notes |
|---|---|---|---|---|
| 🥇 | ippeiogawa | **46/50** | 119 | **The leader.** Private code, heavy experimentation. |
| 🥈 | just public 44, all is luck | **45/50** | 2 | Name says it all — public 44 + secret sauce |
| 🥉 | Batman's Butler | **45/50** | 67 | Active optimization |
| 4 | Riku Suzuki | **45/50** | 29 | — |
| 5 | Seungjun Lee | **45/50** | 4 | Very few entries for 45 — strong approach |
| 6–25 | *20+ teams* | **44/50** | 3–131 | **The "wall"** — most plateau here |

### Best Public Notebooks (Code Tab)

| Notebook | Public Score | Votes | Model |
|---|---|---|---|
| [44/50] LET ME (over)COOK!!! | **44** | 2,747 | GPT-OSS-120B |
| ans_verifys | **44** | 611 | GPT-OSS-120B |
| [43/50] gpt-oss-120b weighted entropy | **43** | 520 | GPT-OSS-120B |
| [15/15] AIME 2026 I \| 120b in 20mins | **43** | 363 | GPT-OSS-120B |
| Chasing 47/50 (experiment journal) | **42** | 45 | GPT-OSS-120B |
| No vllm Deepseek R1 distill | **37** | — | DeepSeek-R1-32B |
| gpt-oss-puzzle-88B | **34** | — | GPT-OSS-88B |

### The Critical Gap

```
Prize Threshold:  47/50  ──────── NOT YET REACHED
Leaderboard #1:   46/50  ████████████████████████████████████████ (ippeiogawa, private code)
Public Code Best: 44/50  ████████████████████████████████ (LET ME COOK, public notebook)
```

**The 2-point gap** between public code (44) and the #1 private entry (46) represents hidden advantages: likely fine-tuned models, GenSelect, or multi-model ensembles. **The 1-point gap** from 46 to the 47 prize threshold is where we need to compete.

### Key Patterns from Top Submissions

1. **GPT-OSS-120B dominates:** 100% of 44+ public scores use this model (117B MoE, 5.1B active, MXFP4 native).
2. **Weighted Entropy > Majority Voting:** All 44+ scorers use entropy-based answer weighting (+3-5 points vs. simple majority).
3. **TIR is table stakes:** Every competitive entry uses sandboxed Python execution via `jupyter_client` persistent kernels.
4. **`min_p=0.02, temp=1.0`:** The standard sampling config across all top entries.
5. **8 attempts, early stop at 4:** Dominant rollout configuration.
6. **No public notebook uses fine-tuning** — this is the biggest untapped opportunity.

### Our Edge — How A.R.C. Will Reach 47/50

| Strategy | Expected Impact | Status |
|---|---|---|
| **GenSelect** over entropy voting | +2-3 points | Planned |
| **Fine-tuned model** (SFT + GRPO) | +2-3 points | Planned |
| **Two-tier model cascade** | +1-2 points on hard problems | Planned |
| **Z3/SMT formal verification** | +1-2 points on constraint problems | Planned |
| **Problem-type-specific prompts** | +1 point | Planned |

---

## 3. System Architecture — How A.R.C. Works

A.R.C. is an **end-to-end autonomous reasoning pipeline** that transforms a raw LaTeX problem statement into a verified integer answer. The system operates in four cascading phases:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        A.R.C. PIPELINE                              │
│                                                                     │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────┐ │
│  │ PROBLEM  │───▶│  TIR AGENT  │───▶│   SOLUTION   │───▶│ FINAL  │ │
│  │  INTAKE  │    │  (N paths)  │    │   SELECTOR   │    │ ANSWER │ │
│  └──────────┘    └─────────────┘    └──────────────┘    └────────┘ │
│       │                │                    │                │      │
│   Parse LaTeX     Generate N          GenSelect or          Submit  │
│   Classify type   reasoning paths     Weighted Entropy      ∈[0,   │
│   Allot compute   w/ code exec        + confidence          99999] │
│                                       scoring                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 1 — Problem Intake & Classification

- Parse the LaTeX-formatted problem from the Kaggle evaluation API.
- Classify the problem domain (Algebra / Combinatorics / Geometry / Number Theory) using a lightweight heuristic or the model's first-pass classification.
- Allocate a **compute budget** (number of rollouts, max tokens, temperature) based on estimated difficulty. Easy problems get fewer rollouts; hard problems get maximum test-time compute.

### Phase 2 — Tool-Integrated Reasoning (TIR) Agent

- The core reasoning engine. The model reads the problem, forms a plan, and enters a **think → code → execute → reflect** loop.
- Each rollout generates an independent solution trajectory using `sympy`, `numpy`, `itertools`, and custom combinatorial solvers.
- The agent handles code execution errors gracefully — errors are appended to the context and the model self-corrects.
- We generate **N = 8–64 rollouts** per problem (adjusted by difficulty class and remaining time budget).

### Phase 3 — Solution Selection (GenSelect + Weighted Entropy)

- Primary: **GenSelect** — a generative selector model that reads the full reasoning traces of top candidates and selects the most mathematically rigorous answer.
- Fallback: **Weighted Entropy Voting** — answers from lower-entropy (higher-confidence) generations are weighted more heavily.
- Final fallback: simple majority vote.

### Phase 4 — Answer Submission

- Extract the integer from `\boxed{...}` or final code output.
- Clamp to `[0, 99999]` and submit via the Kaggle API.

---

## 4. Model Selection & Rationale

> **Full analysis:** See [`docs/suggested-models-and-approaches.md`](docs/suggested-models-and-approaches.md) for comprehensive model comparisons.

> [!WARNING]
> **Model Cutoff Rule:** The competition only allows models **released before March 15, 2026** in the final runtime submission. Models released after this date (e.g., Gemma 4, released April 2, 2026) **cannot** be used for inference on Kaggle. However, they **can** be used offline for synthetic data generation, evaluation pipelines, and training data curation.

### Eligible Runtime Models (Released Before March 15, 2026)

| Model | Params | Active Params | Architecture | AIME '24 | Why Consider |
|---|---|---|---|---|---|
| **GPT-OSS-120B** ⭐ | 117B | 5.1B | MoE (128 experts, top-4) | ~90% | **Current AIMO-3 champion.** MXFP4 native, single H100. Proven 44/50 public, 46/50 LB. |
| **DeepSeek-R1** (32B distill) | 32B | 32B | Dense | 79.8% | Gold standard for open-weight reasoning. Proven at 37/50 without vLLM. |
| **Qwen3-235B-A22B** | 235B | ~22B | MoE | 85.7% | Highest raw math scores but requires aggressive quantization to fit H100. |
| **GPT-OSS-88B** | ~88B | ~4B | MoE | — | Lighter variant, proven at 34/50. Faster inference but weaker reasoning. |
| **Kimi K2.5 (Thinking)** | MoE (large) | — | MoE | ~85% | Deep multi-step planning. |
| **NuminaMath 7B TIR** | 7B | 7B | Dense | ~60% | AIMO-1 winner. Fast but limited (~20-25/50). |

### Offline-Only Models (Post-Cutoff — For Data Generation)

| Model | Released | Offline Use Case |
|---|---|---|
| **Gemma 4 26B A4B** | April 2, 2026 | ⚠️ **NOT eligible for runtime.** Use to generate synthetic TIR training traces, evaluate solution quality, and curate fine-tuning datasets. |
| **Gemma 4 31B Dense** | April 2, 2026 | ⚠️ **NOT eligible for runtime.** AIME '26: 89.2%. Use as a "gold-standard" solution verifier for offline data pipelines. |

### Our Strategy: GPT-OSS-120B Primary + DeepSeek-R1 Fallback

1. **Primary (GPT-OSS-120B):** Handles all 100 problems. MoE architecture (5.1B active) gives fast inference → maximize rollouts within the 300-minute budget.
2. **Fallback (DeepSeek-R1-32B):** Reserved for problems where GPT-OSS-120B fails to reach consensus. Deeper reasoning, fewer rollouts.
3. **Offline: Gemma 4 family** used to generate high-quality synthetic training data for fine-tuning GPT-OSS-120B via SFT + GRPO.

This cascade maximizes the **expected score** across easy/medium/hard problems under a fixed 5-hour compute envelope.

---

## 5. Core Strategy: Tool-Integrated Reasoning (TIR)

Pure Chain-of-Thought (CoT) reasoning is **insufficient** for IMO-level problems. LLMs make arithmetic errors, lose track of variable bindings, and hallucinate algebraic steps. The proven winning approach — used in AIMO-1 (NuminaMath), AIMO-2 (NVIDIA), and every 40+ scorer in AIMO-3 — is **Tool-Integrated Reasoning**.

### How TIR Works in A.R.C.

```
┌──────────────────────────────────────────────────────────┐
│                    TIR EXECUTION LOOP                     │
│                                                          │
│  (1) Model reads LaTeX problem                           │
│            ↓                                             │
│  (2) Model writes natural language plan                  │
│            ↓                                             │
│  (3) Model generates Python code block                   │
│      ┌─────────────────────────────────┐                 │
│      │ <|code_start|>                  │                 │
│      │ from sympy import *             │                 │
│      │ x = symbols('x')               │                 │
│      │ result = solve(x**3 - 7, x)    │                 │
│      │ print(result)                   │                 │
│      │ <|code_end|>                    │                 │
│      └─────────────────────────────────┘                 │
│            ↓                                             │
│  (4) Sandboxed executor runs code (timeout: 30s)         │
│            ↓                                             │
│  (5) Output (or error traceback) appended to context     │
│            ↓                                             │
│  (6) Model continues reasoning with deterministic output │
│            ↓                                             │
│  (7) Repeat (3)-(6) up to K iterations                   │
│            ↓                                             │
│  (8) Model outputs final \boxed{answer}                  │
└──────────────────────────────────────────────────────────┘
```

### Key Libraries (Pre-packaged as Kaggle Datasets)

| Library | Purpose |
|---|---|
| `sympy` | Symbolic algebra, equation solving, number theory (`factorint`, `divisors`, `isprime`, `totient`, `mobius`) |
| `numpy` / `scipy` | Numerical computation, optimization, root-finding |
| `itertools` / `functools` | Combinatorial enumeration, memoization |
| `mpmath` | Arbitrary-precision arithmetic (64+ digits, critical for large modular arithmetic) |
| `z3-solver` | SMT constraint solving (for logic/constraint problems — inspired by Gemma+Z3 notebook) |
| `networkx` | Graph theory problems (Ramsey, coloring, etc.) |

### Executor Design: PersistentKernel Pool

Inspired by the top-scoring notebooks, we use a pool of **persistent Jupyter kernels** rather than one-shot `exec()` calls:

```python
class PersistentKernel:
    PREAMBLE = (
        "import sys, math, itertools, functools, collections, fractions\n"
        "import numpy as np\n"
        "from sympy import *\n"
        "from sympy.ntheory import factorint, divisors, isprime, primerange, totient, mobius\n"
        "from sympy.combinatorics import Permutation, PermutationGroup\n"
    )

    def __init__(self):
        from jupyter_client import KernelManager
        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)
        self._run(self.PREAMBLE, timeout=120)  # Pre-import everything

    def execute(self, code, timeout=30):
        # Run with timeout, capture stdout + stderr
        ...

    def reset(self):
        # Clear variables but keep imports
        ...
```

**Benefits:** State persists across code blocks within a single problem (variables, imported modules). Pre-importing saves ~2s per code execution.

---

## 6. Test-Time Compute: Inference Scaling

A single greedy decode will **not** win this competition. We must maximize the utilization of the H100's compute within the 5-hour window.

### Strategy 1: Self-Consistency with Weighted Entropy

Based on what's working at the top of the leaderboard:

- Generate **N = 8 rollouts** per problem at `temperature=1.0, min_p=0.02`.
- For each rollout, compute the **generation entropy** from token log-probabilities.
- Weight each answer by `1.0 / entropy` — confident answers count more.
- The answer with the highest cumulative weight wins.

### Strategy 2: Adaptive Compute Allocation

| Tier | Condition | Rollouts | Time Budget |
|---|---|---|---|
| **Quick** | First 2 rollouts agree | 2 | ~50s |
| **Standard** | Default | 8 | ~360s |
| **Deep** | No consensus after 8 | 16–32 + Power Tier model | ~600s |
| **Nuclear** | No consensus + IMO-level | 64 + MCTS/ToT | remaining budget |

### Strategy 3: Early Consensus Stopping

```python
EARLY_STOP = 4  # Stop if 4 out of 8 attempts agree

for i in range(MAX_ATTEMPTS):
    answer = run_tir_rollout(problem)
    answers.append(answer)

    # Check for early consensus
    counter = Counter(answers)
    if counter.most_common(1)[0][1] >= EARLY_STOP:
        break  # High confidence — save time for harder problems
```

### Compute Budget Model

```
Total Time Budget:  5 hours = 18,000 seconds
Problems:           50 (public set scored in real-time)
Avg per problem:    360 seconds (6 minutes)

With GPT-OSS-120B (5.1B active MoE):
  - Estimated: ~60-80 tok/s on H100 w/ vLLM + MXFP4
  - Per rollout (avg 2000 tokens): ~30 seconds
  - Rollouts per problem @ 6 min: ~12 rollouts

With early stopping on easy problems:
  - Easy (30%): 2-4 rollouts → ~90s → save 270s each
  - Medium (50%): 8 rollouts → 360s
  - Hard (20%): 16-32 rollouts → 600s

  Total: ~14,700s ≈ 4.1 hours ✅ (buffer for overhead)
```

---

## 7. Solution Selection: Beyond Majority Voting

### Weighted Entropy Voting (Current Meta)

The current leaderboard meta, used by the 44/50 leader:

```python
def weighted_entropy_vote(answers: list, entropies: list) -> int:
    """Weight answers by inverse generation entropy."""
    weighted_counts = defaultdict(float)
    for answer, entropy in zip(answers, entropies):
        weight = 1.0 / max(entropy, 1e-6)  # Avoid div by zero
        weighted_counts[answer] += weight
    return max(weighted_counts, key=weighted_counts.get)
```

### GenSelect (Our Advantage)

GenSelect — the technique from NVIDIA's AIMO-2 win — goes further:

1. **Generate** N candidate solutions with full reasoning traces.
2. **Filter** to the top K unique answers (by frequency + entropy weighting).
3. **Feed the full reasoning traces** of the top K candidates into a selector model.
4. The selector performs **n-ary comparison**: reads all traces, identifies logical errors, checks mathematical rigor, and selects the most convincing solution.
5. For large K, run a **knockout tournament** to efficiently narrow down.

This bridges the gap between high `Pass@N` (correct answer exists in the pool) and low `Majority@N` (correct answer isn't the most common).

---

## 8. Fine-Tuning Pipeline

> No public notebook on the leaderboard uses a fine-tuned model. This is our primary competitive advantage.

### Stage 1: Supervised Fine-Tuning (SFT) on TIR Trajectories

| Parameter | Value |
|---|---|
| **Base Model** | GPT-OSS-120B (only eligible runtime model at 44+ level) |
| **Technique** | LoRA (rank=64, alpha=128) on attention + MLP layers |
| **Data** | NuminaMath-TIR (~70K), OpenMathReasoning (filtered, ~100K), synthetic IMO Shortlist solutions |
| **Epochs** | 2–3 |
| **LR** | 2e-5 with cosine decay |

### Stage 2: Reinforcement Learning via GRPO

**GRPO (Group Relative Policy Optimization)** computes advantages relative to the group, eliminating the need for a separate critic model:

```
For each prompt p:
  1. Generate N responses {r₁, r₂, ..., rₙ}
  2. Score each: Rᵢ = R(rᵢ)
  3. Advantage: Aᵢ = (Rᵢ - μ_group) / σ_group
  4. Update policy using PPO-clip with Aᵢ
```

**Reward Function:**

| Component | Weight | Description |
|---|---|---|
| **Accuracy** | 0.6 | Does `\boxed{answer}` match ground truth? |
| **Execution** | 0.2 | Did code execute without errors? |
| **Format** | 0.1 | Correct TIR formatting? |
| **Efficiency** | 0.1 | Penalty for excessively long traces |

---

## 9. Inference Optimization for Kaggle

The Kaggle environment: **1× H100 (80GB HBM3), 5 hours, no internet**.

### vLLM Engine Configuration

```python
# Matching the proven config from the 44/50 leader
from vllm import LLM, SamplingParams

llm = LLM(
    model="./gpt-oss-120b-finetuned",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.96,      # Aggressive — proven stable
    max_model_len=65536,              # 64K context
    kv_cache_dtype="fp8_e4m3",        # FP8 KV cache for memory
    enable_prefix_caching=True,
    enforce_eager=False,              # CUDA graphs
)

sampling_params = SamplingParams(
    temperature=1.0,
    min_p=0.02,                       # min_p > top_p for reasoning
    max_tokens=4096,
    logprobs=5,                       # For entropy computation
)
```

### Offline Packaging

```
kaggle-datasets/
├── model-weights/           # GPT-OSS-120B (MXFP4 quantized)
├── vllm-wheels/             # Custom vLLM build w/ TurboQuant support
├── python-libs/             # sympy, numpy, scipy, mpmath, z3-solver
└── lora-adapters/           # Fine-tuned LoRA weights (SFT + GRPO)
```

---

## 10. Execution Plan & Timeline

### Phase 1: Environment & Baseline (Days 1–2) `🟢 IN PROGRESS`

- [x] Repository setup and project scaffolding
- [ ] Create offline Kaggle dataset pipeline
- [ ] Build PersistentKernel-based TIR executor
- [ ] Run zero-shot GPT-OSS-120B baseline → target: match public 44/50

### Phase 2: Benchmarking (Days 3–4)

- [ ] Benchmark GPT-OSS-120B vs. DeepSeek-R1-32B (only pre-cutoff eligible models)
- [ ] Profile inference speed and max rollouts per 5 hours
- [ ] Implement weighted entropy voting → match current meta

### Phase 3: Our Competitive Edge (Days 5–8)

- [ ] Implement GenSelect selector
- [ ] Fine-tune (SFT + GRPO) on curated math datasets
- [ ] Build two-tier model cascade
- [ ] Add Z3 solver integration for constraint problems

### Phase 4: Final Optimization (Days 9–10)

- [ ] Stress test full 50-problem run in 5 hours
- [ ] A/B test GenSelect vs. weighted entropy
- [ ] Final submission

---

## 11. Repository Structure

```
A.R.C./
├── README.md                               # This file
├── docs/
│   ├── competitor-analysis.md              # Detailed competitor breakdowns
│   └── suggested-models-and-approaches.md  # Model & technique recommendations
├── main.py                                 # Entry point
├── ai-mathematical-olympiad-progress-prize-3/
│   ├── AIMO3_Reference_Problems.pdf
│   ├── reference.csv                       # 10 reference problems with answers
│   ├── sample_submission.csv
│   ├── test.csv
│   └── kaggle_evaluation/
├── arc/                                    # (Planned) Core library
│   ├── agent.py                            # TIR agent
│   ├── executor.py                         # PersistentKernel pool
│   ├── selector.py                         # GenSelect + entropy voting
│   ├── budget.py                           # Adaptive compute manager
│   └── utils.py                            # LaTeX parsing, answer extraction
├── training/                               # (Planned) Fine-tuning
│   ├── sft.py
│   ├── grpo.py
│   └── data/
└── configs/
    ├── gpt_oss_120b.yaml
    └── inference.yaml
```

---

## 12. References

### Competition
- [AIMO Progress Prize 3 — Kaggle](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [AIMO Prize Official Website](https://aimoprize.com)
- [AIMO Progress Prize 2 — Winning Solutions](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion)

### Models
- [GPT-OSS-120B — OpenAI (2026)](https://openai.com/index/gpt-oss/)
- [Gemma 4 Technical Report — Google DeepMind (April 2026)](https://ai.google.dev/gemma) *(post-cutoff: offline use only)*
- [DeepSeek-R1 — DeepSeek (Jan 2025)](https://arxiv.org/abs/2501.12948)
- [Qwen3 — Alibaba (2025)](https://arxiv.org/abs/2505.09388)
- [NuminaMath — Project Numina](https://projectnumina.ai)
- [OpenMathReasoning — NVIDIA (2025)](https://arxiv.org/abs/2504.16891)

### Techniques
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2501.12948)
- [GenSelect: Generative Best-of-N — NVIDIA (2025)](https://arxiv.org/abs/2507.10225)
- [Tool-Integrated Reasoning for Math](https://arxiv.org/abs/2405.13150)
- [vLLM: Efficient LLM Inference](https://arxiv.org/abs/2309.06180)

### Datasets
- [NuminaMath-TIR (70K) — HuggingFace](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR)
- [NuminaMath-CoT (860K)](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [OpenMathReasoning (3M+) — NVIDIA](https://huggingface.co/datasets/nvidia/OpenMathReasoning)

---

<div align="center">

**Let the games begin.** 🏛️

*Built with obsessive rigor by Team A.R.C.*

</div>
