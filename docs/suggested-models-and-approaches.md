# Suggested Models & Approaches — AIMO Progress Prize 3

> **Last Updated:** April 6, 2026
> **Purpose:** A comprehensive reference for choosing the right models, techniques, and inference strategies to win AIMO-3.

> [!WARNING]
> **Model Eligibility Cutoff:** The competition rules mandate that only models **released before March 15, 2026** may be used in the final runtime submission on Kaggle. Models released after this date (e.g., the Gemma 4 family, released April 2, 2026) **cannot run during evaluation** but **can be used offline** for synthetic data generation, training data curation, and solution verification.

---

## Table of Contents

1.  [Model Landscape](#1-model-landscape)
2.  [Model Comparison Matrix](#2-model-comparison-matrix)
3.  [Recommended Model: GPT-OSS-120B](#3-recommended-primary-model-gpt-oss-120b)
4.  [Alternative Models Deep Dive](#4-alternative-models-deep-dive)
5.  [Inference Approaches](#5-inference-approaches)
6.  [Solution Selection Strategies](#6-solution-selection-strategies)
7.  [Fine-Tuning Strategies](#7-fine-tuning-strategies)
8.  [Tool Integration Approaches](#8-tool-integration-approaches)
9.  [Optimal Configuration for A.R.C.](#9-optimal-configuration-for-arc)
10. [Experimental Ideas to Explore](#10-experimental-ideas-to-explore)

---

## 1. Model Landscape

The mathematical reasoning model landscape has exploded since AIMO-2. Here's where each family fits:

```
                    Reasoning Capability
                    ──────────────────▶
                    
    ▲  GPT-OSS-120B ★        Qwen3-235B ★★
    │  (5.1B active)          (22B active)
    │                         
    │  Gemma 4 27B MoE        DeepSeek-R1-32B
    │  (4B active) ★          (32B dense) ★
S   │                         
p   │  Kimi K2.5              Gemma 4 31B Dense
e   │  (MoE)                  (31B dense)
e   │
d   │  MiMo-V2-Flash          
    │  (hybrid thinking)      
    │                         
    │  NuminaMath-7B-TIR      Qwen3.5-27B
    │  (7B dense)             (27B dense)
    │
    │  DeepSeek-R1-7B
    │  (7B dense)
    │
    └────────────────────────────────────▶
                    Inference Speed
                    (tokens/sec on H100)
```

### Key Insight: MoE Models Dominate

Mixture-of-Experts (MoE) models are the clear winners for this competition because:
- **Small active parameter count** → fast inference → more rollouts in 5 hours
- **Large total parameter count** → deep knowledge → better reasoning
- **Native quantization** (MXFP4) → fit on single H100

---

## 2. Model Comparison Matrix

> [!IMPORTANT]
> Models marked ❌ in the **Eligible?** column were released after the March 15, 2026 cutoff and **cannot be used at runtime**. They can only be used offline.

| Model | Total Params | Active Params | Architecture | Best Quant | AIME '24 Score | Est. tok/s (H100) | Fits H100? | Competition Score | Eligible? |
|---|---|---|---|---|---|---|---|---|---|
| **GPT-OSS-120B** | 116.8B | 5.1B | MoE (128E, top-4) | MXFP4 | ~90% | 60-80 | ✅ | **44/50** pub / **46/50** LB | ✅ |
| **Qwen3-235B-A22B** | 235B | 22B | MoE | AWQ 4-bit | 85.7% | 15-25 | ⚠️ Tight | Untested | ✅ |
| **DeepSeek-R1-32B** | 32B | 32B | Dense | AWQ 4-bit | 79.8% | 40-60 | ✅ | **37/50** | ✅ |
| **GPT-OSS-88B** | ~88B | ~4B? | MoE | MXFP4 | — | 70-90 | ✅ | **34/50** | ✅ |
| **NuminaMath 7B TIR** | 7B | 7B | Dense | FP16 | ~60% | 100+ | ✅ | ~20-25/50 | ✅ |
| **DeepSeek-R1-7B** | 7B | 7B | Dense | AWQ | ~55% | 100+ | ✅ | ~20-25/50 | ✅ |
| ~~Gemma 4 26B A4B~~ | 26B | ~4B | MoE (128E, 8+1) | FP8/INT8 | — | 80-120 | ✅ | Untested | ❌ *Apr 2* |
| ~~Gemma 4 31B Dense~~ | 31B | 31B | Dense | FP8 | 89.2% (AIME '26) | 35-50 | ✅ | Untested | ❌ *Apr 2* |
| **Kimi K2.5** | Large MoE | — | MoE | — | ~85% | — | ⚠️ | Untested | ⚠️ *Check* |
| **MiMo-V2-Flash** | — | — | Hybrid | — | — | — | ✅ | Untested | ⚠️ *Check* |

> ★ **Key Takeaway:** GPT-OSS-120B is the **only competition-proven model that is eligible**. The gap between 7B models (~22/50) and 120B MoE (44-46/50) is enormous. Gemma 4 would have been a strong contender but is disqualified by the March 15 cutoff.

---

## 3. Recommended Primary Model: GPT-OSS-120B

### Why GPT-OSS-120B?

This is the proven champion model in the competition. Here's the case for making it our primary:

#### Strengths

| Factor | Detail |
|---|---|
| **Competition Proven** | Scores 44/50 publicly, 46/50 on private LB |
| **MoE Efficiency** | 5.1B active params → fast inference (~60-80 tok/s) |
| **Native Quantization** | Ships with MXFP4 weights → no accuracy loss from post-training quantization |
| **Reasoning Effort Control** | Configurable `ReasoningEffort.HIGH` / `MEDIUM` / `LOW` → natural adaptive compute |
| **FP8 KV Cache** | Built-in support for `fp8_e4m3` KV cache → extended context |
| **Large Context** | 64K token context window |
| **vLLM Support** | First-class vLLM integration with CUDA graphs |

#### Weaknesses

| Factor | Detail |
|---|---|
| **Closed-ish License** | Not fully open-source; weight distribution may have restrictions |
| **Fine-tuning** | MXFP4 weights may complicate LoRA adaptation |
| **Single-vendor risk** | If model weights are corrupted or incompatible, no fallback |

#### Recommended Configuration

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/kaggle/input/gpt-oss-120b/",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.96,
    max_model_len=65536,
    kv_cache_dtype="fp8_e4m3",
    enable_prefix_caching=True,
    enforce_eager=False,
)

sampling = SamplingParams(
    temperature=1.0,
    min_p=0.02,
    max_tokens=4096,
    logprobs=5,
    stop=["</python>\n", "<|end|>"],
)
```

---

## 4. Alternative Models Deep Dive

### 4.1 Gemma 4 26B A4B (MoE) — ⚠️ OFFLINE USE ONLY

> [!CAUTION]
> **NOT ELIGIBLE for runtime.** Released April 2, 2026 — after the March 15 cutoff. Can only be used for offline data generation, training data curation, and solution verification.

**Architecture:** 26B total params, ~4B active, 128 experts with 8+1 active routing, 256K context.

**Offline use cases:**
- **Synthetic TIR trace generation** — Use Gemma 4 to generate high-quality TIR training data for SFT of GPT-OSS-120B
- **Solution verification** — Cross-check GPT-OSS-120B solutions using Gemma 4 as a "gold standard" verifier
- **Dataset curation** — Filter and rank training problems by difficulty using Gemma 4's reasoning
- **Evaluation pipeline** — Use as an offline judge for A/B testing different prompt strategies

**Why it would have been ideal for runtime:**
- Apache 2.0 license, 256K context, native thinking mode (`<|think|>` tags)
- ~80-120 tok/s on H100 → could have enabled 16-32 rollouts per question
- Function calling support for clean TIR integration

### 4.2 DeepSeek-R1-Distill-Qwen-32B — Proven Reasoning ✅ ELIGIBLE

**Architecture:** 32B dense, distilled from the full DeepSeek-R1.

**Why consider:**
- **Proven in AIMO-2** (part of winning solution)
- **Native TIR support** (`<python>...</python>` code blocks)
- **No vLLM required** — works with raw HF Transformers (proven at 37/50)
- **Well-understood** — extensive community knowledge

**Best for:** Fallback model. When GPT-OSS-120B fails to reach consensus, invoke R1-32B for a deeper, more focused reasoning trace.

**Trade-off:** 32B dense is slower than 5.1B active MoE. Fewer rollouts in the time budget.

### 4.3 Qwen3-235B-A22B — Highest Raw Capability ✅ ELIGIBLE

**Architecture:** 235B total, 22B active MoE.

**Why consider:**
- **Highest math benchmark scores** (AIME '24: 85.7%, AIME '25: 81.5%)
- **Hybrid thinking toggle** — can switch between deep CoT and fast non-thinking modes
- **Dedicated math training** — trained on mathematical reasoning data

**Risk:** At 235B total params, fitting on a single H100 even with 4-bit quantization is extremely tight (~30GB for weights + KV cache overhead). May require aggressive model sharding that kills throughput.

### 4.4 Gemma 4 31B Dense — ⚠️ OFFLINE USE ONLY

> [!CAUTION]
> **NOT ELIGIBLE for runtime.** Released April 2, 2026 — after the March 15 cutoff.

**Architecture:** 31B dense transformer.

**Offline value:**
- **AIME '26 score: 89.2%** — highest published score on the latest AIME
- Excellent as an **offline solution verifier** and **training data generator**
- Use to generate "gold standard" TIR traces for SFT of the runtime model

---

## 5. Inference Approaches

### 5.1 vLLM (Recommended)

The industry standard for high-throughput LLM inference.

**Key optimizations for AIMO-3:**

| Feature | Purpose |
|---|---|
| `gpu_memory_utilization=0.96` | Squeeze every byte for larger batch sizes |
| `kv_cache_dtype="fp8_e4m3"` | 2× more KV cache entries → longer contexts |
| `enable_prefix_caching=True` | Cache system prompt tokens → skip recomputation |
| `enforce_eager=False` | Enable CUDA graphs for ~10% speedup |
| `min_p` sampling | Better sampling diversity than `top_p` for reasoning |

### 5.2 SGLang (Alternative)

An alternative to vLLM with potentially better performance for multi-turn conversations.

**Pros:** RadixAttention for even better prefix caching; native multi-turn support.
**Cons:** Less battle-tested in Kaggle specifically.

### 5.3 Raw Transformers (Fallback)

Direct Hugging Face `transformers` inference without a serving framework.

**Pros:** Simpler setup, fewer dependencies, proven at 37/50.
**Cons:** Much slower (~2-3× fewer rollouts in the time budget).

### 5.4 Speculative Decoding (Experimental)

Use a small draft model (e.g., 7B) to propose tokens, verified by the large model.

**Pros:** 2-4× speedup for greedy decoding.
**Cons:** Less benefit when using high-temperature sampling (which we need for diversity).

---

## 6. Solution Selection Strategies

### 6.1 Simple Majority Voting

```python
from collections import Counter
final_answer = Counter(answers).most_common(1)[0][0]
```

**Score ceiling:** ~39/50 (proven in competition)
**Failure mode:** When the correct answer is a minority answer

### 6.2 Weighted Entropy Voting ⭐ (Current Meta)

```python
def weighted_entropy_vote(answers, entropies):
    weighted_counts = defaultdict(float)
    for ans, ent in zip(answers, entropies):
        weighted_counts[ans] += 1.0 / max(ent, 1e-6)
    return max(weighted_counts, key=weighted_counts.get)
```

**Score ceiling:** ~44/50 (proven in competition)
**Why it works:** Low-entropy generations are more likely to be correct. Weighting by inverse entropy recovers correct minority answers.

### 6.3 GenSelect ⭐⭐ (Our Advantage)

```
1. Generate N solutions with full reasoning traces
2. Filter to top K unique answers
3. Feed ALL traces for top K into a selector model
4. Selector performs n-ary comparison:
   - Reads each reasoning chain
   - Identifies logical errors
   - Checks mathematical rigor
   - Selects the most convincing solution
```

**Expected ceiling:** 47+/50 (proven in AIMO-2 by NVIDIA to outperform all other selection methods)
**Why it works:** Actually evaluates reasoning quality, not just confidence signals

### 6.4 Answer Evolution (Alternative)

```
1. Generate initial solution
2. Model critiques its own solution
3. Model generates revised solution
4. Repeat until convergence
```

**Score ceiling:** ~41/50 (proven in competition)
**Best for:** Problems where the model "almost" gets the right answer

### 6.5 Hybrid: Entropy + GenSelect

Our recommended approach:

```
1. Generate 8 rollouts
2. Check for early consensus (4/8 agree) → submit immediately
3. If no consensus: run GenSelect on top-3 unique answers
4. If GenSelect is low-confidence: fall back to entropy voting
```

This maximizes both speed (easy problems resolved quickly) and accuracy (hard problems get deep analysis).

---

## 7. Fine-Tuning Strategies

### 7.1 Supervised Fine-Tuning (SFT)

**Purpose:** Teach the model the TIR format and mathematical reasoning patterns.

| Parameter | Recommended |
|---|---|
| **Method** | LoRA (rank=64, alpha=128) |
| **Target** | Attention Q/K/V/O + MLP gate/up/down |
| **Data** | NuminaMath-TIR (70K) + OpenMathReasoning (filtered 100K) + synthetic IMO solutions |
| **Epochs** | 2-3 |
| **LR** | 2e-5 with cosine decay |
| **Sequence Length** | 8192 tokens |

**Key datasets:**

| Dataset | Size | Content |
|---|---|---|
| NuminaMath-TIR | 70K | TIR trajectories with code execution |
| NuminaMath-CoT | 860K | Chain-of-thought solutions |
| OpenMathReasoning | 3M+ solutions, 540K problems | NVIDIA-curated math reasoning |
| AIMO3 Val Bench | ~100 problems | Competition-format validation set |

### 7.2 GRPO (Group Relative Policy Optimization)

**Purpose:** Teach the model to self-improve through trial-and-error on math problems.

```
For each prompt p:
  1. Generate N=8 responses {r₁, ..., rₙ}
  2. Score each: Rᵢ = accuracy + execution_quality + format_compliance
  3. Advantage: Aᵢ = (Rᵢ - mean(R)) / std(R)
  4. Update policy using PPO-clip with Aᵢ
```

**Why GRPO over PPO/DPO:**
- **No critic model** needed → saves VRAM
- **Group normalization** provides stable gradients
- **Proven** in DeepSeek-R1 and multiple math reasoning papers

**Reward function:**

| Component | Weight | Description |
|---|---|---|
| Accuracy | 0.6 | `\boxed{answer}` matches ground truth |
| Code Execution | 0.2 | Code runs without errors |
| Format | 0.1 | Proper TIR formatting |
| Efficiency | 0.1 | Penalty for excessively long traces |

### 7.3 Fine-Tuning Decision Matrix

| Scenario | Recommendation |
|---|---|
| Limited compute + time | Skip fine-tuning, use GPT-OSS-120B zero-shot (proven at 44/50) |
| Moderate compute | SFT only on NuminaMath-TIR (can improve format compliance by 5-10%) |
| Full compute budget | SFT + GRPO on curated dataset (expected: +3-5 points vs. zero-shot) |

---

## 8. Tool Integration Approaches

### 8.1 Standard TIR (Baseline)

```
Model generates: <python> code </python>
System executes code, returns stdout
Model incorporates result and continues reasoning
```

**Used by:** All top notebooks

### 8.2 Z3 SMT Solver Integration (Novel)

```python
from z3 import *

# Model generates Z3 constraints
x, y, z = Ints('x y z')
s = Solver()
s.add(x + y + z == 100)
s.add(x > 0, y > 0, z > 0)
s.add(x * y * z == 1000000)

if s.check() == sat:
    model = s.model()
    print(f"x={model[x]}, y={model[y]}, z={model[z]}")
```

**Best for:** Number theory, constraint satisfaction, combinatorics with hard constraints.
**Inspired by:** Gemma+Z3 notebook on the competition page.

### 8.3 Google OR-Tools Integration (Novel)

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
# Constraint programming for optimization problems
```

**Best for:** Optimization and scheduling-type problems.

### 8.4 Tool Priority Matrix

| Problem Type | Primary Tool | Secondary Tool |
|---|---|---|
| Algebra | SymPy (solve, simplify) | NumPy (numerical verification) |
| Number Theory | SymPy (factorint, isprime) | Z3 (constraint proofs) |
| Combinatorics | itertools + functools | Z3 + OR-Tools |
| Geometry | SymPy (geometry module) | NumPy + scipy.spatial |
| General | mpmath (64-digit precision) | brute-force enumeration |

---

## 9. Optimal Configuration for A.R.C.

Based on all competitive intelligence and model analysis, here is our recommended configuration:

### Configuration A: "Proven Winner" (Low Risk)

```yaml
model: GPT-OSS-120B
quantization: MXFP4 (native)
inference: vLLM
selection: Weighted Entropy Voting
rollouts: 8 per problem
early_stop: 4 consensus
temperature: 1.0
min_p: 0.02
time_budget: 360s per problem
fine_tuning: None (zero-shot)
expected_score: 44/50 (matches public baseline)
```

### Configuration B: "Smart Upgrade" (Medium Risk) ⭐ RECOMMENDED

```yaml
primary_model: GPT-OSS-120B (MXFP4)
fallback_model: DeepSeek-R1-32B (AWQ 4-bit)
inference: vLLM
selection: GenSelect (primary) + Entropy Voting (fallback)
rollouts: 8-16 per problem (adaptive)
early_stop: 4 consensus
temperature: 1.0
min_p: 0.02
tools: sympy, numpy, mpmath, z3-solver
time_budget: 360-600s per problem (adaptive)
fine_tuning: SFT on NuminaMath-TIR
expected_score: 46-47/50 (compete with / beat LB #1)
```

### Configuration C: "Maximum Ambition" (High Risk)

```yaml
primary_model: GPT-OSS-120B (MXFP4, SFT+GRPO fine-tuned)
fallback_model: DeepSeek-R1-32B (AWQ 4-bit)
inference: vLLM + speculative decoding
selection: GenSelect tournament
rollouts: 16-64 per problem (adaptive)
early_stop: 4 consensus
temperature: 1.0
min_p: 0.02
tools: sympy, numpy, mpmath, z3-solver, OR-Tools
time_budget: fully adaptive
fine_tuning: SFT + GRPO on curated 200K dataset
offline_data_gen: Gemma 4 26B + 31B (for synthetic TIR traces)
expected_score: 47-49/50 (prize territory)
```

> [!NOTE]
> Gemma 4 is used **only in the offline fine-tuning pipeline** (generating training data). The runtime model remains GPT-OSS-120B, which is eligible under the March 15 cutoff.

---

## 10. Experimental Ideas to Explore

### 10.1 Reasoning Chain Distillation

- Use a frontier closed-source model (Claude 4, GPT-5) to generate "gold" reasoning chains for olympiad problems
- Distill these chains into our open-weight model via SFT
- This is how DeepSeek-R1 was originally trained

### 10.2 Multi-Model Ensemble (Eligible Models Only)

Run 2 eligible models and combine their answers:
```
GPT-OSS-120B (8 rollouts) + DeepSeek-R1-32B (4 rollouts)
→ Cross-model entropy-weighted voting
```

Risk: Time budget may not permit running multiple large models.

### 10.3 Formal Verification Layer

After generating a candidate answer, attempt to formally verify it:
1. Translate the problem + solution into Z3 constraints
2. Check if the solution satisfies all constraints
3. If verification fails, discard the solution and retry

### 10.4 Curriculum-Based GRPO

Instead of training on random problems, organize training problems by difficulty:
1. Start with easy problems (AMC level)
2. Progressively increase to IMO level
3. This mirrors how human mathematicians develop problem-solving skills

### 10.5 Tree-of-Thought (ToT) for Hard Problems

For the hardest problems where linear reasoning fails:
```
                    [Problem]
                   /    |    \
              [Approach1] [Approach2] [Approach3]
                 /  \       |           \
            [Step1a] [Step1b] [Step2a]  [Step3a]
              ...    ...      ...        ...
```

Explore multiple reasoning branches in parallel, evaluate each, and select the most promising path.

### 10.6 Offline Data Pipeline with Gemma 4

Since Gemma 4 cannot be used at runtime but **can** be used offline:

```python
# OFFLINE ONLY — run locally, NOT on Kaggle
# Generate gold-standard TIR traces for fine-tuning GPT-OSS-120B

from vllm import LLM

# Use Gemma 4 31B Dense for highest accuracy
gemma = LLM(model="google/gemma-4-31b-it", ...)

for problem in olympiad_problems:
    # Generate multiple solution traces
    traces = gemma.generate(problem, n=16, temperature=1.0)
    # Filter correct solutions
    correct = [t for t in traces if verify_answer(t, problem.answer)]
    # Save as SFT training data for GPT-OSS-120B
    save_tir_traces(correct, format="gpt_oss")
```

This lets us leverage Gemma 4's superior math reasoning (89.2% AIME) to generate training data that improves GPT-OSS-120B's runtime performance.

---

## Summary: Priority Actions

> **Competitive Reality (April 6 2026):** Prize threshold is 47/50. Best public code = 44. Best private LB = 46 (ippeiogawa, 119 entries). Gap to prize = 1 point.

| Priority | Action | Expected Impact | Target |
|---|---|---|---|
| 🔴 **P0** | Get GPT-OSS-120B running with proven config | Baseline competitive score | 44/50 |
| 🔴 **P0** | Implement PersistentKernel pool + TIR | Required for any competitive score | 44/50 |
| 🟠 **P1** | Implement weighted entropy voting | +3-5 points vs. majority voting | 44/50 |
| 🟠 **P1** | Add Z3 solver as additional tool | +1-2 points on constraint problems | 45/50 |
| 🟡 **P2** | Implement GenSelect | +2-3 points vs. entropy voting | 46-47/50 |
| 🟡 **P2** | SFT fine-tune on NuminaMath-TIR | +2-3 points on format + reasoning | 46-47/50 |
| 🟢 **P3** | GRPO training | +1-2 additional points | 47+/50 |
| 🟢 **P3** | Two-tier model cascade | +1-2 points on hard problems | 47+/50 |

---

*This document should be updated as new models are released and competition dynamics evolve.*
