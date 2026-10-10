# Adapting seat 95's Level 8 code on seat 97

Date: October 10, 2026 (America/New_York).

## Scope and result

We copied a fresh snapshot of `/workspace/projects/02-kernel-agent` from **seat 95**, preserved its sources, and adapted a copy under **seat 97**:

```text
/workspace/level8-seat95-adaptation-shubham-97
```

Seat 95 was read only throughout. Its project and running jobs were not edited or stopped. Seat 97's existing model server stayed running. The previously successful seat 97 experiment in `/workspace/level8-repair-shubham-97-planned` was preserved.

The selected original candidate fails all three official Level 8 cases. Its human-adapted version passes all three official cases on actual Trainium2 hardware, each executed twice, and passes 24 supplementary simulator cases. The adapted agent also solved **2/2 fresh runs on round 0**, with all four sampled candidates correct. Each selected generated kernel passed the same hardware and supplementary checks. This human repair is distinguished from the fresh model-generation experiment described below.

Start with [results/summary.json](projects/20-kernel-agent/experiments/seat95-adaptation/results/summary.json) for the final machine-readable results, provenance, and evidence-file hashes. The post-test inference probe returned HTTP 200 and `SEAT97_ADAPTATION_OK`; the original model-server PID 103 still existed.

Local experiment: [projects/20-kernel-agent/experiments/seat95-adaptation](projects/20-kernel-agent/experiments/seat95-adaptation).

## Exactly what was copied

The read snapshot at `2026-10-10T21:31:53Z` includes `agent.py`, `nkibench.py`, `lint.py`, four helper kernels, and `holdout_check.py`. Originals are in the local experiment's `original/` directory. [snapshot.json](projects/20-kernel-agent/experiments/seat95-adaptation/original/snapshot.json) records the source paths, SHA256 hashes, selected candidate metadata, and source-log hash.

We extracted the last selected candidate available in `attempts-l8-simple.jsonl`: **run 3, round 1, sample 0** (zero-based), reward `0.30`, `runs=false`, `correct=false`. The source log contained 92 complete candidate records and no correct candidate at that snapshot; this is a partial snapshot of a running experiment, not a final solve-rate result.

Original kernel: [original_level08_attention.py](projects/20-kernel-agent/experiments/seat95-adaptation/original/original_level08_attention.py).

Original candidate SHA256:

```text
432e82ce55c92dc9820100596467f1585c95b8212cb08865d32e5b059c419ec4
```

The copied `nkibench.py` and `lint.py` remain **byte-for-byte unchanged** in the working experiment. We also compared the attention reference, input builder, seed convention, and numerical checker with the previous seat 97 harness; their function ASTs match.

## Kernel repair

Repaired kernel: [level08_seat95_adapted.py](projects/20-kernel-agent/experiments/seat95-adaptation/solved/level08_seat95_adapted.py).

This is a human repair of the copied candidate, preserving its input-loading and output-storage structure. The exact [kernel.patch](projects/20-kernel-agent/experiments/seat95-adaptation/kernel.patch) records the changes:

1. Transpose Q as well as K, with separate PSUM and SBUF buffers. `nc_matmul` computes `stationary.T @ moving`, so Q.T and K.T are the score operands.
2. Allocate a `(seq,seq)` score accumulator instead of reusing K's `(dim,seq)` transpose buffer.
3. Add the missing multiplication by `1/sqrt(dim)` before softmax.
4. Keep scaled scores and shifted scores in distinct SBUF tiles; preserve stable row-max subtraction.
5. Normalize using `nl.sum`, `nisa.reciprocal`, and per-row multiplication.
6. Transpose probabilities rather than V for the final product. Use `stationary=P.T`, `moving=V`, and a `(seq,dim)` output accumulator.
7. Copy the final PSUM result through SBUF before storing the output in HBM.

Repaired candidate SHA256:

```text
bb6ecb551d1e9488686f3d35a8ed82fd76fc1db926bade2831bd6e8824d794d2
```

The contract remains `softmax(Q @ K.T / sqrt(dim), axis=1) @ V`, with float32 Q/K/V and output shape `(seq,dim)`.

## Agent adaptation

The latest seat 95 agent already contains several improvements absent from the earlier audit, including operation-aware API-card handling, diagnostic feedback, and selection using static-issue counts. Those changes were retained.

We made a focused change to the copied `agent.py`: add the previously verified attention contract and buffer/stage plan to **both** the initial Level 8 prompt and repair prompt. The repair instruction permits all linked stage corrections instead of requiring everything outside the first named error to remain identical. The first-prompt branch applies at every terseness setting.

The plan gives shapes, buffer roles, and SDK call patterns; it does not supply a complete hand-written kernel. No seed kernel or helper blocks are passed in the fresh-generation command.

Exact change: [agent.patch](projects/20-kernel-agent/experiments/seat95-adaptation/agent.patch). Combined adapted `agent.py` + unchanged `nkibench.py` + unchanged `lint.py` source hash: **`9d88e4062534`**. Other level-specific prompt paths and grading logic were retained.

## Validation of the human-adapted kernel

All kernel tests in this experiment ran on seat 97. No local laptop kernel execution was used.

| Check | Original copied candidate | Human-adapted candidate |
|---|---|---|
| Copied `nkibench.py --check`, official cases | 0/3; contraction-dimension exceptions | 3/3 passed |
| Copied agent grader with lint enabled | Reward 0.30; three static issues | Reward 1.00; all correctness stages passed |
| Supplementary CPU simulation | Not rerun | 24/24 passed |
| Actual Trainium2 execution, official cases | Not attempted | 3/3 passed, each called twice |

The original checker reproduced `stationary[0] != moving[0]` for all three shapes: `128 != 64`, `64 != 128`, and `96 != 32`.

The hardware runner imports the official shapes directly from the copied `nkibench.LEVELS[8]`, builds inputs through `make_inputs(..., seed=0)`, and uses that harness's reference and numerical acceptance function. It executes `kernel[2](*args)` through standalone NKI JIT and Neuron Runtime, without the CPU simulator.

| Official shape `(seq,dim)` | Device normalized maximum error |
|---|---:|
| `(128,64)` | `1.19955e-5` |
| `(64,128)` | `1.05806e-5` |
| `(96,32)` | `1.22213e-5` |

All results met the existing threshold `0.02`. Repeated outputs were bitwise equal and inputs stayed unchanged. Simulated HBM traffic on the three official cases was respectively `131072`, `131072`, and `49152` bytes, equal to reading Q/K/V once and storing output once. These traffic counts are simulator measurements, not hardware latency.

A first supplementary-runner invocation stopped before numerical execution because our copied runner passed `level=8` to seat 95's older `lint_kernel(source)` interface. We changed that runner call to `lint_kernel(source)` and preserved the failed invocation's log. The official checker and linter were not changed to resolve it.

## Fresh model-generation experiment

The adapted seat 95 agent runs against seat 97's existing `Qwen/Qwen3-8B` endpoint with two repeats, two samples per round, up to three rounds, context 8192, maximum completion 3000, lint, level hints, temperature portfolio, and echo checking. Successful selected kernels receive separate supplementary simulation and actual hardware checks.

**Completed result: 2/2 runs solved on round 0; all four candidates scored 1.00.** All completions ended with `stop`. Each request used 1515 prompt tokens; completions ranged from 1025 to 1034 tokens. Aggregate usage was 6060 prompt tokens and 4115 completion tokens across the four requests.

| Selected generated kernel | Official cases in agent simulator | Supplementary simulator cases | Official hardware cases |
|---|---|---|---|
| [Run 0](projects/20-kernel-agent/experiments/seat95-adaptation/solved/level08_generated_run0.py) | 3/3 | 24/24 | 3/3, each called twice |
| [Run 1](projects/20-kernel-agent/experiments/seat95-adaptation/solved/level08_generated_run1.py) | 3/3 | 24/24 | 3/3, each called twice |

Both selected kernels are byte-identical, SHA256 `4460dc8d1d0e6ef3e56a5909643c0e42d1699507d463deead43265539f7beda9`. The two runs establish repeatability of this smoke experiment, not diversity of solutions or a general solve-rate estimate. No paired fresh baseline run was performed. The baseline candidate failure and previous running seat 95 logs are separate observations.

For both generated kernels, maximum normalized device error was `1.22213e-5`; repeated outputs were bitwise equal and inputs unchanged. Their supplementary simulations had maximum normalized error `6.39689e-6` and minimum simulated HBM traffic in all 24 cases.

Evidence:

- [agent-fresh.jsonl](projects/20-kernel-agent/experiments/seat95-adaptation/results/agent-fresh.jsonl): all four complete model responses as extracted code, grades, token counts, generation settings, selection, and source hash.
- [generation-summary.json](projects/20-kernel-agent/experiments/seat95-adaptation/results/generation-summary.json): completed generation command and per-run follow-up exit codes.
- [adapted-device.json](projects/20-kernel-agent/experiments/seat95-adaptation/results/adapted-device.json): the human-repaired candidate's device results.
- [generated-run0-device.json](projects/20-kernel-agent/experiments/seat95-adaptation/results/generated-run0-device.json) and [generated-run1-device.json](projects/20-kernel-agent/experiments/seat95-adaptation/results/generated-run1-device.json): actual device execution of the generated kernels.
- `results/*-independent.json`: the supplementary simulator records for each saved kernel. `solved/*.provenance.json`: source and validation provenance.
- `results/original-official.log`, `results/adapted-official.log`, and `results/grade-comparison.json`: original failure, repaired official-check success, and grader comparison.
- `results/adapted-independent-v1.log`: the preserved supplementary-runner interface error before its correction.

These checks establish that the copied seat 95 code can be adapted successfully for Level 8 on seat 97. Do not treat the human repair as a model-generated solve, or the 24 simulator cases as hardware coverage.

## Reproduction

From the isolated remote directory:

```bash
cd /workspace/level8-seat95-adaptation-shubham-97

# Original copied checker, unchanged.
python3 -B nkibench.py --level 8 --check solved/level08_seat95_adapted.py

# Supplementary simulator cases.
python3 -B tests/check_attention.py solved/level08_seat95_adapted.py \
  --json results/adapted-independent-repeat.json

# Actual Trainium device execution; preserve existing model service.
env -u NEURON_RT_VISIBLE_CORES \
  NEURON_RT_NUM_CORES=1 NEURON_LOGICAL_NC_CONFIG=2 \
  NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  python3 -u -B tests/check_attention_device.py solved/level08_seat95_adapted.py \
  --lnc 2 --suite official --json results/adapted-device-repeat.json --stop-on-error

# Fresh model generation, without a seed or complete example kernel.
python3 -u -B agent.py --level 8 --rounds 3 --samples 2 --repeat 2 \
  --context 8192 --max-tokens 3000 --base http://localhost:8000/v1 \
  --model Qwen/Qwen3-8B --lint --level-hints --portfolio --echo-check \
  --log results/agent-fresh-repeat.jsonl
```

## Resume and limits

Read this document, the original snapshot, `agent.patch`, and the results summaries before analyzing the repo again. Keep the original copied sources and the previous seat 97 solution intact. Use new log filenames for new runs.

This establishes the workshop's small, unmasked, single-head float32 attention contract. It does not establish arbitrary shape/dtype support, long-context attention, production performance, or modification of Qwen's internal attention implementation. Three official hardware inputs and 24 supplementary simulator cases are distinct coverage claims. Recorded device-call wall times include host/runtime/compilation/transfers; no isolated kernel speedup has been measured.
