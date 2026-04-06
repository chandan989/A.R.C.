# Competitor Analysis — AIMO Progress Prize 3

> **Last Updated:** April 6, 2026
> **Source:** [Kaggle Leaderboard](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/leaderboard) & [Public Notebooks](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/code)

This document provides a detailed analysis of the competition landscape — the actual leaderboard standings, every notable public submission, and a reverse-engineering of what the top private entries are likely doing differently.

> [!WARNING]
> **Model Cutoff Rule:** Only models released before **March 15, 2026** are eligible for runtime use. Gemma 4 (released April 2, 2026) is NOT eligible.

---

## Table of Contents

1.  [Live Leaderboard (As of April 6, 2026)](#1-live-leaderboard)
2.  [The Gap: Public vs. Private Code](#2-the-gap-public-vs-private-code)
3.  [Tier S: 45–46/50 — Private Entries](#3-tier-s-4546--private-entries-reverse-engineered)
4.  [Tier A: 44/50 — The Public Meta](#4-tier-a-4450--the-public-meta)
5.  [Tier B: 43/50 — Strong Public Notebooks](#5-tier-b-4350--strong-public-notebooks)
6.  [Tier C: 37–42/50 — Alternative Approaches](#6-tier-c-3742--alternative-approaches)
7.  [Tier D: Experimental & Novel (Unscored)](#7-tier-d-experimental--novel-unscored)
8.  [Cross-Cutting Patterns](#8-cross-cutting-patterns)
9.  [Key Takeaways for A.R.C.](#9-key-takeaways-for-arc)

---

## 1. Live Leaderboard

The **actual competition leaderboard** (not just public notebooks) shows significantly higher scores than what's available in public code:

### Top 25 (Live, April 6 2026)

| # | Team | Score | Entries | Last Active | Prize Contender? |
|---|---|---|---|---|---|
| **1** | **ippeiogawa** | **46** | 119 | 9h ago | ✅ Gold |
| **2** | just public 44, all is luck | **45** | 2 | 1 month | ✅ Gold |
| **3** | Batman's Butler | **45** | 67 | 13h ago | ✅ Gold |
| **4** | Riku Suzuki | **45** | 29 | 1d ago | ✅ Gold |
| **5** | Seungjun Lee | **45** | 4 | 3d ago | ✅ Gold |
| **6** | i.won.a.maths.debate | **44** | 130 | 1d ago | ✅ Gold |
| **7** | Neeraj | **44** | 51 | 9d ago | ✅ Gold |
| **8** | Hrithik Reddy | **44** | 27 | 4d ago | ✅ Gold |
| **9** | Varra | **44** | 114 | 7h ago | ✅ Gold |
| **10** | Pai | **44** | 33 | 21d ago | ✅ Gold |
| **11** | Aman | **44** | 91 | 18h ago | ✅ Gold |
| **12** | ttytaen | **44** | 3 | 1 month | ✅ Gold |
| **13** | Boris Liskov | **44** | 53 | 21d ago | ✅ Gold |
| **14** | Harrabi Raouf | **44** | 105 | 14h ago | ✅ Gold |
| **15** | shaochenjie | **44** | 65 | 2d ago | ✅ Gold |
| **16** | Kunal Aarse | **44** | 131 | 18h ago | ✅ Gold |
| **17** | Corentin Lespagnol | **44** | 19 | 4d ago | ✅ Gold |
| **18** | que | **44** | 55 | 1d ago | — |
| **19** | Leaderboard drama | **44** | 112 | 14h ago | — |
| **20** | amine | **44** | 15 | 3d ago | — |
| **21** | Dark Horse ▼ | **44** | 83 | 6h ago | — |
| **22** | Onur Fidaner | **44** | 37 | 9h ago | — |
| **23** | Sandeep Goyal | **44** | 29 | 1d ago | — |
| **24** | RandomTeam | **44** | 80 | 1d ago | — |
| **25** | Haraguchi-T | **44** | 38 | 1d ago | — |

### Score Distribution

```
46/50  ████  1 team      ← THE TARGET TO BEAT
45/50  ████████  4 teams
44/50  ████████████████████████████████████  20+ teams  ← The "wall"
43/50  ██████████████  Many teams
42/50  ██████████  Many teams
...
```

### Key Observations

- **The prize threshold is ≥47/50.** Nobody has hit it yet. The #1 team at 46 is tantalizingly close.
- **46 → 47 is the hardest 2 points.** The gap between 44 (easy to replicate with public code) and 47 (prize-winning) is where the real competition lies.
- **ippeiogawa has 119 entries** — this is someone doing systematic experimentation, not just running a public notebook.
- **"just public 44, all is luck"** — their team name literally says the 44/50 public notebook approach was their starting point, and they squeezed out +1 point through unknown modifications.
- **Multiple teams at 44 with 100+ entries** — brute-force hyperparameter search is happening at scale.

---

## 2. The Gap: Public vs. Private Code

This is the most important insight in this analysis:

| Source | Best Score | Notes |
|---|---|---|
| **Leaderboard** (private code) | **46/50** | Top scorer, 119 entries, all private |
| **Public notebooks** (Code tab) | **44/50** | "[44/50] LET ME (over)COOK!!!" |
| **Prize threshold** | **47/50** | Not yet achieved |

The **2-point gap** between public code (44) and the leaderboard leader (46) represents:

1. **Secret sauce in answer selection** — likely GenSelect or a proprietary aggregation method
2. **Fine-tuned models** — custom LoRA/GRPO-tuned variants not shared publicly
3. **Better prompts** — carefully optimized system prompts developed over 119 submissions
4. **Ensemble strategies** — multi-model approaches
5. **Custom tooling** — Z3, OR-Tools, or other specialized solvers

The additional **1-point gap** from 46 to the 47 prize threshold may require:
- A fundamentally better reasoning model
- Novel verification techniques (formal proofs)
- Problem-type-specific strategies
- Or simply luck (the test set is only 50 problems — variance is high)

---

## 3. Tier S: 45–46/50 — Private Entries (Reverse-Engineered)

These entries don't share their code publicly. We can infer their approaches from:
- Entry count patterns
- Team member profiles
- Competition discussion posts
- The "Chasing 47/50" experiment journal

### 📊 ippeiogawa (#1, Score: 46, 119 Entries)

**What we know:**
- 119 entries → systematic experimentation over weeks/months
- Active 9 hours ago → still iterating, likely squeezing for 47
- Gold prize contender medal

**Likely approach (inferred):**
- GPT-OSS-120B as base (it's the only model publicly shown to reach 44+)
- Likely has **custom answer verification** beyond simple entropy voting
- With 119 entries of A/B testing, they've likely optimized:
  - System prompt down to specific mathematical reasoning patterns
  - Sampling parameters (temperature, min_p, top_k combinations)
  - Per-problem-type strategies
  - Time allocation between easy/hard problems
- Possibly uses a **fine-tuned model** or custom LoRA adapter

### 📊 Teams at 45 (4 teams)

**"just public 44, all is luck" (#2, 45, 2 entries)**
- Their name implies they started with the public 44/50 notebook
- Only 2 entries → got lucky or made one targeted modification
- Could be as simple as a better random seed or a slightly different voting threshold

**Batman's Butler (#3, 45, 67 entries)**
- 67 entries → serious experimentation
- Likely found a specific improvement over the 44/50 baseline

**Riku Suzuki (#4, 45, 29 entries)**
- Moderate entry count → found an effective modification relatively quickly

**Seungjun Lee (#5, 45, 4 entries)**
- Very few entries for such a high score → likely has a fundamentally different approach or a very strong model

---

## 4. Tier A: 44/50 — The Public Meta

### 📓 [44/50] LET ME (over)COOK!!!

**Author:** nihilisticneuralnet | **Public Score:** 44 | **Votes:** 2,747 | **Forks:** 4,936+

This notebook defines the current **public** meta and is the starting point for ~80% of top leaderboard entries.

#### Model & Infrastructure

```
Model:              GPT-OSS-120B (116.8B total, 5.1B active MoE)
Architecture:       36 layers, 128 experts, top-4 routing
Quantization:       MXFP4 (Microscaling FP4) — native, no accuracy loss
KV Cache:           FP8 (fp8_e4m3)
Inference Engine:   vLLM (served as OpenAI-compatible API server)
GPU:                1× H100 (80GB)
```

#### vLLM Configuration (Exact)

```python
gpu_memory_utilization = 0.96    # Extremely aggressive — proven stable
temperature = 1.0                 # High diversity for self-consistency
min_p = 0.02                      # min_p sampling (NOT top_p)
context_tokens = 65536            # 64K context window
batch_size = 256
workers = 16                      # 16 parallel Jupyter kernels
attempts = 8                      # 8 rollouts per problem
early_stop = 4                    # Stop if 4/8 agree
turns = 128                       # Max code execution turns per rollout
seed = 42
```

#### System Prompt (Key Excerpts)

```
"You are a world-class mathematician..."

# Mathematical Reasoning Principles:
- Break complex problems into smaller, manageable sub-problems
- Look for patterns, symmetries, and special cases
- Use concrete examples to build intuition before generalizing
- Consider extreme cases and boundary conditions
- If stuck, try working backwards from the desired result
- Be willing to restart with a different approach if needed

# Verification Requirements:
- Cross-check arithmetic and algebraic manipulations
- Verify that your solution satisfies all problem constraints
- Test your answer with simple cases or special values
- Ensure dimensional consistency and reasonableness

# Output Format:
The final answer must be a non-negative integer between 0 and 99999.
Place your final numerical answer inside \boxed{}, e.g., \boxed{42}
```

#### Tool Prompt

```
Use this tool to execute Python code for:
- Complex calculations that would be error-prone by hand
- Numerical verification of analytical results
- Generating examples or testing conjectures
- Brute-force verification for small cases

The environment is a stateful Jupyter notebook. Code persists between executions.
Always use print() to display results. Write clear, well-commented code.
Code should support your mathematical reasoning, not replace it.
```

#### Key Innovation: Weighted Entropy Voting

```python
# Instead of simple majority vote:
# weight_i = 1.0 / entropy_i
# final_answer = argmax(Σ weight_i for each unique answer)
```

Entropy from token log-probabilities → lower entropy = higher confidence = more weight. This recovers correct minority answers that majority voting would miss.

#### Architecture Diagram

```
┌───────────────┐     ┌──────────────────┐     ┌───────────────┐
│  vLLM Server  │◄───▶│  Main Controller │◄───▶│ 16× Jupyter   │
│  (subprocess) │     │  (orchestrator)  │     │ Kernel Pool   │
│               │     │                  │     │               │
│  GPT-OSS-120B │     │  - Problem queue │     │ - sympy       │
│  MXFP4 + FP8  │     │  - 8 attempts    │     │ - numpy       │
│  64K context   │     │  - Entropy calc  │     │ - mpmath(64)  │
│               │     │  - Early stop@4  │     │ - itertools   │
└───────────────┘     └──────────────────┘     └───────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │ Weighted     │
                      │ Entropy Vote │
                      │ → \boxed{}   │
                      └──────────────┘
```

### 📓 ans_verifys (Score: 44, 611 votes)

A fork/variant of the 44/50 notebook with an **answer verification** step:
- After generating candidate answers, the model is asked to verify each answer independently
- Answers that survive verification get boosted weight
- This is a lighter-weight version of GenSelect

### Why 44 and Not 46+?

The 44 ceiling for public code is likely due to:
1. **No fine-tuning** — all public notebooks use zero-shot models
2. **Simple selection** — entropy voting is good but not optimal
3. **No problem-specific strategies** — same approach for all 50 problems
4. **No verification layer** — answers aren't formally verified

---

## 5. Tier B: 43/50 — Strong Public Notebooks

### 📓 [43/50] AIMO 3: gpt-oss-120b weighted entropy (Score: 43, 520 votes)

**Author:** Boning Cui (copied from Andreas Bisiadis)

Same model and approach as 44/50 but with slightly different hyperparameters. The 1-point gap is likely:
- Fewer parallel workers
- Different early stopping threshold
- Slightly different system prompt

**Key lesson:** Even small hyperparameter differences yield measurable score differences at this level.

### 📓 [15/15] AIME 2026 I | 120b in 20mins (Score: 43, 363 votes)

**Author:** (copied from JK-Piece)

Notable for achieving a **perfect 15/15 on AIME 2026** with GPT-OSS-120B. This demonstrates the model's raw capability when given enough compute per problem. The lower AIMO score (43 vs 44) may be due to less optimized time budgeting for the 50-problem set.

---

## 6. Tier C: 37–42/50 — Alternative Approaches

### 📓 Chasing 47/50: AIMO3 Journey of 50+ Experiments (Score: 42, 45 votes)

**Author:** Pawan Rama Mali | **Important: An experiment journal, not a submission notebook**

This is the single most valuable public resource in the competition. Despite only scoring 42/50, it documents **50+ experiments** conducted over 2 months, providing critical insights:

#### Key Details
```
Model:      GPT-OSS-120B (danielhanchen/gpt-oss-120b)
Hardware:   NVIDIA H100 (80GB)
Best Score: 42/50 (Version 40, Feb 6 2026)
```

#### Documented Findings (from Executive Summary)
- **Model size is the #1 variable** — no amount of prompt engineering overcomes a weak model
- **Code execution correctness > frequency** — it's better to write fewer, correct code blocks than many
- **Prompt engineering has diminishing returns** past a quality threshold
- **Entropy-based selection** outperforms simple majority voting by 3-5 points
- **Temperature=1.0 is optimal** for diverse reasoning paths

### 📓 No vllm Deepseek R1 distill (Score: 37)

**Author:** abkmystery

Achieves competitive results **without vLLM** using raw HF Transformers + DeepSeek-R1-Distill-Qwen-32B.

```python
MODEL = "deepseek-r1-distill-qwen-32b"
MAX_MODEL_LEN = 16384
TOTAL_BUDGET_S = 17000         # ~4.7 hours
MAX_PER_PROBLEM_S = 540        # 9 min per problem
TIR_SAMPLES = 5                # Tool-integrated attempts
FAST_SAMPLES = 3               # Pure CoT attempts
EARLY_CONSENSUS = 3            # Stop if 3 agree
```

**Key techniques:**
- **Dual-mode sampling:** Both TIR (with code) and fast CoT (pure reasoning), combining results
- **PersistentKernel:** Pre-imports sympy, numpy, itertools, fractions, collections, mpmath
- **Stop sequences on `</python>`:** Clean code block extraction

**Why 37 not 44?**
- DeepSeek-R1-32B (dense 32B) << GPT-OSS-120B (117B MoE) in reasoning capability
- Slower inference → fewer rollouts in 5 hours
- Simple consensus voting, not entropy-weighted

### 📓 gpt-oss-puzzle-88B (Score: 34)

Uses the smaller **88B variant** of GPT-OSS. The 10-point gap from 120B confirms model scale is paramount.

---

## 7. Tier D: Experimental & Novel (Unscored)

### 📓 gemma 4 with z3 and cuda 13 and tools

**Author:** Fedor Baart

> [!CAUTION]
> **This approach uses Gemma 4, which is NOT ELIGIBLE for runtime** (released April 2, post-March 15 cutoff). However, the **Z3 + OR-Tools integration pattern** is extremely valuable and can be adapted to work with GPT-OSS-120B.

```
Model:    google/gemma-4-31b-it  ⚠️ INELIGIBLE (post-cutoff)
Tools:    Z3 SMT Solver + Google OR-Tools + SymPy
Method:   Native Gemma 4 tool-calling (function calling API)
```

Instead of relying solely on LLM reasoning + SymPy, this integrates the **Z3 SMT solver** for formal constraint verification and **Google OR-Tools** for optimization problems.

**What to steal for A.R.C. (with eligible models):**
- Z3 solver integration works with **any model** — just teach GPT-OSS-120B to generate Z3 constraints via TIR
- Constraint satisfaction problems (Number Theory, Combinatorics) are better solved by SAT/SMT solvers
- Z3 can formally prove/disprove propositions — no hallucination possible
- Complementary to TIR: use LLM for exploration, Z3 for verification

### 📓 AIMO3 Qwen3.5-27B 4xSpeed FP8 MTP

**Author:** The Silverhead Engineer

Focuses on **inference speed**: FP8 quantization + Multi-Token Prediction + Speculative Decoding. More rollouts per hour, but the base model (Qwen3.5-27B) may lack sufficient reasoning depth.

### 📓 Qwen Math LLM Dual-Pass

**Author:** Avik Das

**Two-pass reasoning:**
1. Pass 1 (Planning): High-level solution strategy
2. Pass 2 (Execution): Detailed mathematical derivation with code

Mirrors human mathematical problem-solving.

---

## 8. Cross-Cutting Patterns

### What ALL 44+ Scorers Share

| Pattern | Details |
|---|---|
| **GPT-OSS-120B** | 100% of 44+ public scorers use this model |
| **Tool-Integrated Reasoning** | Sandboxed code execution via `jupyter_client` |
| **Multiple rollouts** | 8+ attempts per problem |
| **`min_p=0.02, temp=1.0`** | The exact sampling configuration |
| **FP8 KV cache** | To fit 64K context in memory |
| **`gpu_memory_utilization=0.96`** | Aggressive but proven stable |
| **Pre-imported libraries** | sympy, numpy, math, itertools, mpmath at kernel start |
| **`\boxed{}` extraction** | Regex-based answer extraction |
| **Offline dependency packaging** | All deps pre-packaged as Kaggle Datasets |

### Score by Selection Method

| Method | Best Public Score | Best Leaderboard Score |
|---|---|---|
| Weighted Entropy Voting | **44/50** | **46/50** (likely) |
| Answer Verification (ans_verifys) | **44/50** | Unknown |
| Simple Majority Voting | ~39-41/50 | Unknown |
| Iterative Answer Refinement | ~41/50 | Unknown |

### Score by Model

| Model | Best Public | Best Leaderboard | Eligible? |
|---|---|---|
| GPT-OSS-120B (5.1B active MoE) | **44/50** | **46/50** | ✅ |
| GPT-OSS-88B | 34/50 | Unknown | ✅ |
| DeepSeek-R1-Distill-Qwen-32B | 37/50 | Unknown | ✅ |
| DeepSeek-R1-Distill-Qwen-7B | ~22/50 | Unknown | ✅ |
| Gemma 4 31B | Unscored | Unknown | ❌ *Post-cutoff (Apr 2)* |
| Qwen3.5-27B | Unscored | Unknown | ✅ |

---

## 9. Key Takeaways for A.R.C.

### The Competitive Reality

1. **The prize threshold (47/50) hasn't been hit.** The race is still open.
2. **The best public code scores 44/50.** The best private entry scores 46/50.
3. **The 44→46 gap is where the real competition is.** Everyone can copy public notebooks.
4. **The 46→47 gap is the prize.** This likely requires a qualitative improvement, not just tuning.
5. **Model cutoff (March 15, 2026)** eliminates Gemma 4 from runtime use. GPT-OSS-120B is the only proven eligible model.
6. **Gemma 4 can be used offline** — for generating synthetic training data, verifying solutions, and curating fine-tuning datasets.

### What We Must Do (Table Stakes for 44)

- [x] Use GPT-OSS-120B with MXFP4
- [ ] Implement PersistentKernel pool with 16 workers
- [ ] Use vLLM with `gpu_memory_utilization=0.96, kv_cache_dtype=fp8_e4m3`
- [ ] 8 rollouts, early stop at 4 consensus
- [ ] Weighted entropy voting
- [ ] `temperature=1.0, min_p=0.02`

### What Will Push Us Past 44 (Targeting 46+)

1. **GenSelect** instead of entropy voting
2. **Fine-tuned model** (SFT + GRPO) — no public notebook does this
3. **Z3 solver integration** for constraint problems (+1-2 points)
4. **Answer verification layer** (like ans_verifys but more rigorous)
5. **Problem-type-specific strategies** (different prompts for algebra vs. number theory)

### What Could Push Us to 47+ (Prize Territory)

1. **Two-tier model cascade** (GPT-OSS-120B + DeepSeek-R1-32B, adaptive)
2. **Formal verification** on candidate answers
3. **Synthetic training data** generated by Gemma 4 offline (post-cutoff model, allowed for offline use) or frontier closed-source models
4. **Custom GRPO reward** optimized on olympiad-difficulty problems
5. **MCTS/Tree-of-Thought** for the hardest 5-10 problems

### What NOT to Waste Time On

- ❌ Small models (7B-27B) — model capability is the dominant factor
- ❌ Complex prompt engineering beyond the established template
- ❌ Pure CoT without code execution
- ❌ Speculative decoding (marginal benefit with high-temp sampling)
- ❌ Trying to hit 47 before establishing a solid 44-45 baseline

---

*This analysis is based on publicly available notebooks and leaderboard data as of April 6, 2026. Private solutions may use additional techniques not documented here. The competition closes April 15, 2026.*
