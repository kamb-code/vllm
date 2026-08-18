# vLLM issue #39146 — investigation data

Replication + discrimination experiments for
[vllm-project/vllm#39146](https://github.com/vllm-project/vllm/issues/39146)
("Non-deterministic output at temperature=0 without prefix caching").

**Scope stamp:** vLLM main @ `b01728b0880ca419bb41199523535457f4ab0010`
(v0.27.2rc1.dev199), built with `VLLM_USE_PRECOMPILED=1 uv pip install -e .`;
1× NVIDIA RTX A6000 48 GB (sm86, driver 580.159.03); Qwen/Qwen2.5-0.5B-Instruct
(bf16, FLASH_ATTN backend); 10 runs per finding per configuration; 2026-08-18.

**Conclusion:** the divergence replicates but is floating-point
batch-composition non-determinism, not KV corruption. See the issue comment for
the full argument.

## Files

- `summary_all_arms.json` — per-arm / per-finding / per-request statistics:
  number of distinct output token sequences across 10 runs, distinct common
  prefixes (separates real divergence from cancelled-stream truncation),
  first-divergence token index, top-2 logprob gap at that index in the
  reference run, and a sha256 digest of every run's full token-ID sequence.
- `raw/*.json.gz` — complete per-run captures: output token IDs, per-token
  logprobs, top-5 alternatives, HTTP status, finish reason, and
  `/metrics` scrapes (`vllm:num_preemptions_total` etc.) around every run.
- `raw/server_P1b_addendumA_zerolog.txt.gz` — zeroing-probe evidence log:
  1,913 zeroing calls / 30,341 blocks; per-call sampled-block canary
  (`pre_dirty` = block held nonzero stale data before zeroing,
  `post_zero_ok` = block read back all-zero after; 0 failures).
- `raw/server_P3_*_cobatch.txt.gz` — per-scheduler-step batch composition logs
  (request id → tokens scheduled) for the co-batching analysis.
- `repro_ext.py` — instrumented replayer (extends the reporter's repro.py,
  which it imports for bit-identical prompt reconstruction; the reporter's
  repro.py + finding JSONs are in the issue's gists).
- `p1b_zeroing.py`, `p1b_addendum_a.py` — the zeroing-probe patches applied to
  main (equivalent in effect to PRs #39283/#43741 on current main's pipeline),
  plus counters/canary instrumentation. Also on branch `probe/p1b-zeroing`.
- `p3_cobatch_log.py`, `p3_analyze.py` — scheduler co-batch logging patch and
  the join analysis. Also on branch `probe/p3-cobatch`.

## Exact reproduction commands

```bash
# server (reporter's flags + explicit no-APC arm):
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-memory-utilization 0.95 --max-model-len 32768 \
  --no-enable-prefix-caching

# replay (repro.py + finding JSONs from the issue's gists in the same dir):
python3 repro_ext.py --runs 10 --label stock_noapc --out-dir results/

# serialized control (bit-identical expected):
python3 repro_ext.py --runs 10 --label serialized --serialize --out-dir results/

# batch-invariant arm (bit-identical expected for non-cancelled requests):
VLLM_BATCH_INVARIANT=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-memory-utilization 0.95 --max-model-len 32768 \
  --no-enable-prefix-caching
python3 repro_ext.py --runs 10 --label batchinv_noapc --out-dir results/
```

Investigation performed with AI assistance (Claude Code); methodology and
results reviewed by the human submitter.
