# Level 8 attention: read-only analysis of seat-95

## Scope and snapshot

Inspected October 10, 2026. The quantitative snapshot below was frozen at **4:20:41 p.m. America/New_York (20:20:41 UTC)**. An existing agent job continued appending logs during inspection. We did not launch, stop, restart, or change that job or the model server.

Remote project: `/workspace/projects/02-kernel-agent` on `seat-95`. Sources inspected: `agent.py`, `nkibench.py`, `lint.py`, `block_transpose.py`, `block_matmul.py`, `block_softmax.py`, recorded JSONL attempts, and existing text logs. We also read installed SDK source/stubs under `/opt/conda/lib/python3.13/site-packages/nki`. No candidate, simulator, test suite, model generation, compiler, or accelerator benchmark was executed for this audit. Remote Python commands only read files and inspect their text/AST or summarize existing records.

The latest experiment's combined source hash, `ce85a6338485`, matches the inspected `agent.py` + `nkibench.py` + `lint.py`. Its recorded Git commit is `8f1ca41`; that commit alone does not capture the substantial working-tree changes.

The local checkout was observed on `team20-kernel-agent`, with project code moved to `projects/20-kernel-agent` and the old local `projects/02-kernel-agent` containing only `__pycache__`. The combined hash of local `agent.py`, `nkibench.py`, and `lint.py` also equals `ce85a6338485`, so those three current local files match the audited remote source snapshot. Paths and saved artifacts still differ; do not apply findings from an earlier local checkout to this version. This report is saved locally; no remote documentation or code was edited.

## Finding

**The current attention agent is stuck on NKI layout/API composition before numerical correctness. The latest inspected experiment is not merely failing a few unusual numerical dataset cases.**

In the frozen snapshot, all 36 candidates in `attempts-l8-lint3.jsonl` were rejected by static checks. Every record has `parses=true`, `rules=true`, `runs=false`, `correct=false`, and `finish=stop`. The model emits complete, parseable code, but the code is structurally invalid.

Reward **0.30 does not mean 30% correct attention**. It is `0.10` for parsing plus `0.20` for passing the basic rule scan. The optional placement/shape linter runs after `parts['rules']=True`; it can reject the candidate before import or simulation while the reported reward remains 0.30.

## What the records establish

| Recorded experiment | Candidates at snapshot | Successful execution recorded | Correct candidates recorded |
|---|---:|---:|---:|
| `attempts-l8-blocks-a.jsonl` | 64 | 0 | 0 |
| `attempts-l8-lint.jsonl` | 16 | 0 | 0 |
| `attempts-l8-lint2.jsonl` | 8 | 0 | 0 |
| `attempts-l8-lint3.jsonl` | 36 | 0 | 0 |
| Total | 124 | 0 | 0 |

These are logged candidates across different configurations, not 124 independent full runs. Earlier candidates were sometimes executed and raised exceptions; `runs=false` means no simulator call returned successfully. It does not prove execution was never attempted. For the latest 36 specifically, the recorded feedback confirms static rejection before simulation.

The older 64-candidate blocks experiment contains 37 first-reported memory-placement failures, 16 contraction mismatches, 9 shape/layout failures, and 2 syntax failures. These categories classify the recorded first error, not every defect present in each kernel.

The first full run of the latest experiment ended with `reward 0.30 after 8 round(s)`, `solved 0/1`. Its selected candidates had **4, 2, 5, 1, 3, 2, 1, 1** reported static issues across rounds 0–7. That is oscillation, not consistent convergence. A second run was already underway in the frozen snapshot; do not report a final 0/2 solve rate from this partial data.

That first run generated 41,152 prompt tokens and 21,605 completion tokens across 32 candidates. These are aggregate request counts, not a single context window or a device-performance measurement.

## The exact attention contract

Remote NKI level 8 computes dense, single-head attention:

`S = Q @ K.T / sqrt(D)`

`P = exp(S - row_max(S)) / row_sum(exp(S - row_max(S)))`

`O = P @ V`

Inputs `Q,K,V` each have shape `(N,D)`, and output must be `(N,D)`. This is not NumPy level-9 windowed attention and is not a change to the attention mechanism inside Qwen.

The three NKI test specifications are `(N,D) = (128,64), (64,128), (96,32)`. `_args_attention` generates float32 standard-normal Q/K/V. `grade()` calls `make_inputs` without an alternate seed, so repeated grading reuses the deterministic default cases. All three are non-square: swapping sequence and feature dimensions cannot pass by accident.

These sizes support a whole-tile design for this challenge. There is no need to introduce a long-sequence tiled attention algorithm merely to resolve the present failure.

## The NKI layout rule the model keeps missing

The installed SDK documents `nisa.nc_matmul` as:

`dst = stationary.T @ moving`

It contracts the first/partition axis of both input tiles. Consequently:

| Operation | Stationary SBUF input | Moving SBUF input | PSUM result |
|---|---|---|---|
| Scores `Q @ K.T` | Actual transposed Q: `(D,N)` | Actual transposed K: `(D,N)` | `(N,N)` |
| Output `P @ V` | Actual transposed P: `(N,N)` | V: `(N,D)` | `(N,D)` |

Transposition here is a data/layout operation. Allocating a buffer with swapped dimensions or performing a normal copy does not transpose the matrix.

For the Tensor-engine transpose pattern used by this system, the flow is SBUF input -> `nc_transpose` -> PSUM -> `tensor_copy` -> SBUF. Both matmul inputs must be SBUF; matmul outputs go to PSUM. Final output uses PSUM -> `tensor_copy` -> SBUF -> `dma_copy` -> HBM.

P and P.T are both `(N,N)`. A shape checker cannot tell that the necessary transpose was omitted merely by comparing their dimensions. The model must preserve the actual mathematical operation and data layout.

## Exact defects in a recorded selected candidate

The following line numbers refer to **run 0, round 7, selected sample 2** in `attempts-l8-lint3.jsonl`, not to a live temporary file. This is the final selected candidate of the first completed run.

### 1. Line 30: the currently reported contraction error

```python
nisa.nc_matmul(dst=t_ps, stationary=q_sb, moving=t_ps_sb)
```

`q_sb` is `(N,D)`. `t_ps_sb` contains K.T and is `(D,N)`. NKI therefore sees first dimensions N and D, which do not match in any of the three supplied cases. For `(128,64)`, the mismatched contraction lengths are 128 and 64.

**Required correction in a later implementation:** transpose Q as well as K, copy both transposes from PSUM to distinct SBUF tiles, and allocate a separate `(N,N)` score accumulator. Its correct operands are `stationary=qT_sb`, `moving=kT_sb`.

The current `t_ps` buffer is only `(D,N)`, because it was allocated for K's transpose. Reusing it as the score result is another error: scores require `(N,N)`. Fixing the operand alone is insufficient.

### 2. Line 31: a non-callable dtype and reversed scaling

```python
nisa.tensor_scalar(dst=t_ps, data=t_ps, op0=nl.divide,
                   operand0=nl.float32(1.0 / (dim ** 0.5)))
```

The installed SDK defines `float32: DType = 'float32'` in `nki/language/_dtypes.py:39`, and exports it through `nki.language`. It is a dtype identifier, not a Python cast function. Calling `nl.float32(...)` is therefore a source-level defect that would raise a non-callable-string error if execution reached this expression. This was inferred by reading the installed source; we did not execute it.

Independently, dividing scores by `1/sqrt(D)` multiplies them by `sqrt(D)`. Attention requires multiplication by `1/sqrt(D)`. The resulting score scale would be too large by a factor of D if this division were performed. For D=64, that is a 64x error in logits, which changes softmax probabilities.

Use a compile-time Python scalar and the intended multiply operation in a future implementation; allocate the scaled score tile with `(N,N)`.

### 3. Lines 34–35: downstream shape inconsistency

The current `t_ps` is `(D,N)`, so reducing its free axis produces `(D,1)`, not the intended `(N,1)` per-query row statistic. It is then combined with a destination `y` allocated as `(N,N)`. Fix the upstream score layout; changing the denominator or output allocation to match this incorrect score tile would preserve the wrong attention operation.

### 4. Line 41: an invented reduction API

```python
nisa.sum(dst=r, view=y, axis=[1], keepdims=True)
```

The installed `nki.isa` exports no `sum`; its source has no dynamic module fallback that supplies one. The appropriate reduction shown by this project's API card and softmax block is `nl.sum(y, axis=[1], keepdims=True)`. Normalization then uses reciprocal and per-row multiplication. Merely fixing the matmul will leave this later API error in place.

### 5. Line 45: the final multiplication is reversed

```python
nisa.nc_matmul(dst=out_ps, stationary=v_sb, moving=e)
```

Assuming `e` is P, this computes `V.T @ P`, with shape `(D,N)`. The task requires `P @ V`, with shape `(N,D)`. This error survives dimension checks because the candidate also allocates `out_ps` as `(D,N)`.

Use the actual transpose P.T as stationary and V as moving. Preserve the final output contract `(N,D)`.

### 6. Lines 19–20: output buffers have the wrong shape

Both `out_ps` and `out` are allocated `(D,N)`. That disagrees with the reference `(N,D)` on every supplied test shape. The exact HBM region spelling is not the demonstrated issue: this SDK exports both `nl.hbm` and `nl.shared_hbm`; the examples use `shared_hbm`.

### 7. Line 46: unsupported final DMA source

```python
nisa.dma_copy(dst=out, src=out_ps)
```

The source is PSUM. The installed DMA documentation describes HBM/SBUF operands, and its implementation rejects this PSUM-to-HBM pairing. The required final movement is a tensor copy to an SBUF tile, then a DMA copy from SBUF to output HBM.

Only the contraction mismatch is currently reported by the linter for this candidate. **One reported static issue does not mean one remaining bug.** The other defects above are inferred from the recorded code and installed SDK source, not newly observed runtime traces.

## Why the present agent keeps reproducing bad code

### Prompt instructions conflict with attention's inputs

`agent.py:197–230` defines a shared API card containing, at line 214, “The left operand arrives already transposed.” That is true for the matmul ladder's `lhsT`, but false for attention's Q/K/V. The standard attention prompt includes this card. The shorter prompt repeats the claim at lines 753–757 for every level >=3, so shortening the prompt does not remove the misconception.

The attention API card provides useful calls, but the wrong general input-layout statement remains in the initial prompt. The input contract needs operation-specific wording.

### Static rejection bypasses the detailed attention hint

`grade()` returns immediately at lines 87–96 if `lint_kernel` finds issues. The detailed attention layout plan is added by `enrich()`/`level_hint()` only later when simulation raises an exception (lines 138–151 and 627–632).

Thus `--level-hints` is enabled, but the static rejection path does not receive the richer operation-specific plan. It receives the generic linter message instead. The repair prompt still includes the attention API card; this is a missing structural/semantic hint, not an absence of all documentation.

### Repair scope is too narrow for a broken composition

`repair_prompt()` at lines 793–798 says to change exactly what the checker names and keep everything else identical. When the checker identifies one contraction issue, this instruction preserves the reversed scaling, invented reduction, incorrect output product, and bad output layout.

The repair prompt includes current code and checker feedback, but does not reassert the reference formula, required output shape, or a stage-by-stage contract. As a result, a wrong function can become the next prompt's effective specification.

### Some linter advice fixes local consistency at the expense of the task

In the first round's feedback, the checker suggests changing an output allocation from `(N,D)` to `(D,N)` to match the copied source. That makes a copy locally consistent but violates attention's required output. The later selected candidate indeed uses `(D,N)` output. This is consistent with the repair following the wrong local advice, though the logs do not prove that this instruction alone caused the change.

The linter has no function-output contract. Prefer repairing the producer/layout toward the required result, rather than adapting the required result to an incorrect producer.

### Candidate selection cannot reliably distinguish structural progress

The primary reward is flat at 0.30 for static-rejected candidates. When `--echo-check` or `--portfolio` is enabled, the tie-breaker at lines 979–983 prefers changed code and different feedback; it does not prefer fewer verified violations or more correct attention stages.

There is concrete evidence: in the first round, sample 0 with four reported issues was selected while sample 3 had three. Fewer issues would not prove semantic correctness, but the current selection rule does not even use that signal. A changed error string is not sufficient evidence of improvement.

The latest flags enable `--prompt-portfolio` and `--echo-check`, but not `--portfolio`, so all samples use temperature 0.6. The four framings include both “rewrite cleanly” and “make the smallest possible change,” appended to the base instruction to keep everything else identical. This mixes incompatible repair scopes.

### Verified building blocks are available but not automatically composed

The remote contains transpose, matmul, and softmax block files. The latest inspected flags do not specify `--blocks`, so those files are not included simply because they exist. An earlier blocks experiment still had 64 unsuccessful candidates; showing examples alone did not solve composition.

The remote ladder now has helper levels 9 (transpose), 10 (row softmax), and 11 (scaled QK.T scores). Existing logs show the transpose helper solved 2/2 runs. The current local checkout additionally contains `solved/level10_row_softmax.check.txt`, recording 3/3 helper shapes passed with 1.00x the byte floor; this audit only read that saved result. It tests `(128,64)`, `(64,128)`, and `(96,32)`, rather than the square `(N,N)` score matrices in composed attention. This audit did not find a successful level-11 run or a local level-8/11 solved artifact. A helper's availability or a comment calling a block verified is not itself proof that it has passed the exact shapes used in composed attention.

## What the checker does and does not prove

Useful existing protections include syntax/rule checks, optional memory/layout linting, reference comparison, input-preservation checking, simulator hardware-hazard warnings, per-attempt telemetry, and provenance hashing. Current standalone NKI verification includes the same input/traffic/hazard gates as agent grading. The earlier local `KeyError: 'M'` issue is already guarded remotely by checking for `M` before applying the matmul roofline calculation.

Important limits:

- `lint.py` knows selected op names, memory locations, and simple symbolic shapes. It does not validate every API call, dtype-as-function call, mathematical scale, normalization, or required function return shape.
- It cannot infer a correct attention product from same-shaped square tiles. Actual P transposition matters.
- Shape inference handles simple positional shape tuples but not all equivalent expressions or keyword shape declarations. Memory/shape maps are populated by whole-tree passes rather than a complete control-flow analysis. Treat it as a partial checker, not a proof that every correct kernel is accepted.
- `nc_transpose` rules are specialized toward the Tensor-engine pattern. The installed SDK also supports Vector-engine transpose for appropriate tiles; a blanket “PSUM only” rule is not a universal description of every legal transpose.
- The default numerical criterion is worst absolute error divided by reference-output RMS <=0.02, not elementwise `rtol=0.02`.
- Level 8 has no `max_waste` threshold. Its description encourages keeping intermediate scores on-chip, but the current traffic gate alone does not enforce that attention-specific goal. Numerical acceptance would not prove fused attention or device speedup.
- Simulator traffic counting and CPU correctness are not accelerator latency, successful device compilation, or a complete on-chip resource-capacity check.

## Context compression is not the immediate remedy

The latest configuration uses Qwen/Qwen3-8B, context 8192, four samples, eight rounds, two requested repeats, level hints, lint, prompt portfolio, and echo checking. In the frozen 36-candidate snapshot, every finish reason is `stop`; the largest prompt is 1,582 tokens and the largest completion is 769. Even adding those independent maxima yields only 2,351 tokens before small request overhead.

There is no evidence here that context-window exhaustion or completion truncation causes the current failure. The inspected generation/repair path builds strings directly and calls the chat-completions endpoint; it contains no Jev selection step in that path.

Jev could later select optional history/API documentation. It should not remove the formula, shape/memory contracts, valid SDK call forms, current code, or failure evidence. Compressing contradictory guidance without correcting it would retain the contradiction.

## Recommended changes, not executed

1. **Pin the attention contract in every prompt.** Q/K/V arrive `(N,D)`; scores and probabilities are `(N,N)`; output is `(N,D)`; scale is multiplication by `1/sqrt(D)`. Remove the matmul-only pretransposed-input claim from attention prompts.
2. **Represent the stages explicitly.** Track Q.T, K.T, raw scores, scaled scores, stable probabilities, P.T, and output with distinct roles and appropriate buffers. Do not reuse K's transpose accumulator as the score accumulator.
3. **Route static failures through attention-specific feedback.** Explain the intended operation and its producer/output contract, not only a generic copy or matmul fix.
4. **Expand API lint selectively using the installed version.** Detect non-callable dtype identifiers, unsupported `nisa` names, illegal DMA memory pairs, and the required entry-point output shape. Keep numerical truth with the reference checker.
5. **Use a coherent repair strategy.** Preserve stages that are established correct; permit all linked fixes needed for a broken stage. Avoid asking the same sample both to rewrite and to keep everything identical.
6. **Improve selection without altering official correctness.** Use the furthest validated stage, hard-error categories/counts, and repeated-attempt avoidance as secondary selection signals. Novelty alone is insufficient; a lower lint count is also not proof of correctness.
7. **Use the helper curriculum to establish composition.** The missing scaled-score helper (level 11) is directly relevant. Softmax must be checked on `(N,N)` score matrices, and the final P.T/V matmul needs its own contract. Do this only when execution is authorized.
8. **Then validate independently.** Future cases should include Q or K zero (output equals mean V per query), constant V (output remains that constant), joint K/V row permutations (output unchanged), length one (output V), and large finite logits. Check supplied cases plus additional seeds and compare against the stated reference/tolerance. These tests were proposed, not run.
9. **Optimize after correctness.** Require evidence for on-chip intermediate handling and measure actual hardware latency separately. The present logs establish neither.

## Evidence pointers

All paths below are remote under `/workspace/projects/02-kernel-agent` unless marked SDK:

- `agent.py:55` grading; `87` static-lint early return; `153` successful-execution flag; `197` base API card; `214` wrong attention input-layout guidance; `238` attention call card.
- `agent.py:516` contraction-specific hint; `550` attention composition plan; `627` hint dispatch; `732` first prompt; `787` repair prompt; `830` request/token handling; `927` solve loop; `979` tie-breaking; `1054` configuration.
- `nkibench.py:164` attention reference; `212` inputs; `246` default seed; `355` level-8 contract; `369`, `380`, `391` helper levels; `483` numerical comparison; `632` traffic bar; `680` simulation/counting; `745` standalone checker.
- `lint.py:18` allowed memory rules; `65` linter; `140` matmul shape checks; `174` copy-shape checking; `187` allocation advice that can violate the task's output contract.
- `attempts-l8-lint3.jsonl`: provenance `ce85a6338485`, run 0 round 7 selected sample 2 for the detailed candidate analysis. `run-l8-lint3.log`: first-run unsolved summary.
- SDK `nki/language/_dtypes.py:39`: float32 dtype string. `nki/isa/__init__.py`: no sum export. `nki/isa/_copy.py`: DMA implementation. `nki/isa/__init__.pyi`: matmul, transpose, scalar-operation, and memory documentation. No SDK module was imported to execute these operations.

No credentials are stored in this report. No application code changes, remote edits, inference, or kernel execution were performed as part of this read-only analysis.
