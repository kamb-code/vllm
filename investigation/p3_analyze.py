#!/usr/bin/env python3
"""P3 analysis: join scheduler co-batch logs with divergence onsets.

For each run: reconstruct which requests shared each scheduler step; find, for
each diverging request, the step that produced its first-divergence token and
the co-batched set at that step. Also fingerprint each run's composition
sequence to show scheduling varied across runs.
"""
import hashlib
import json
import re
import sys

server_log, ext_json = sys.argv[1], sys.argv[2]

COBATCH_RE = re.compile(r"COBATCH (\{.*\})")
ID_RE = re.compile(r"^cmpl-(.+)-0-[0-9a-f]{8}$")

sys.path.insert(0, "/workspace/repro")
from repro import build_prompt  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

blob = json.load(open(ext_json))
finding = json.load(open("/workspace/repro/" + blob["finding"]))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
prompt_lens = {}
for e in finding["trace"]["events"]:
    if e["action"] == "send" and e["request_id"] not in prompt_lens:
        prompt_lens[e["request_id"]] = len(tok(build_prompt(e))["input_ids"])

# --- parse steps, split into runs at RUNMARK ---
runs_steps: list[list[dict[str, int]]] = []
cur: list[dict[str, int]] = []
for line in open(server_log):
    m = COBATCH_RE.search(line)
    if not m:
        continue
    d = eval(m.group(1))  # trusted local log
    step = {}
    is_mark = False
    for k, n in d.items():
        im = ID_RE.match(k)
        base = im.group(1) if im else k
        if base.startswith("RUNMARK"):
            is_mark = True
        else:
            step[base] = n
    if is_mark:
        runs_steps.append(cur)
        cur = []
        continue
    if step:
        cur.append(step)
runs_steps.append(cur)
runs_steps = [r for r in runs_steps if r][-len(blob["runs"]):]  # last N runs
print(f"parsed {len(runs_steps)} runs of steps; sizes: {[len(r) for r in runs_steps]}")

# --- per-run composition fingerprint ---
fps = []
for steps in runs_steps:
    fp = hashlib.md5(json.dumps(steps, sort_keys=True).encode()).hexdigest()[:10]
    fps.append(fp)
print("composition fingerprints per run:", fps)
print("distinct compositions:", len(set(fps)), "/", len(fps))

# --- divergence onset -> co-batch set ---
report = blob["report"]["requests"]
runs = blob["runs"]
for rid, e in sorted(report.items()):
    if e["distinct_token_seqs"] <= 1 or e["first_div_token_idx"] is None:
        continue
    fdi = e["first_div_token_idx"]
    plen = prompt_lens.get(rid)
    if plen is None:
        continue
    target = plen + fdi
    print(f"\n{rid}: fdi={fdi} (prompt {plen} tok, gap={e['top2_gap_at_div']})")
    for ri, steps in enumerate(runs_steps):
        cum = 0
        onset_set = None
        for step in steps:
            if rid in step:
                cum += step[rid]
                if cum >= target:
                    onset_set = sorted(k for k in step if k != rid)
                    break
        print(f"  run{ri}: co-batched at onset with {onset_set}")
