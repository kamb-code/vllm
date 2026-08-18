#!/usr/bin/env python3
"""
repro_ext.py — Instrumented replayer for vLLM issue #39146.

Extends Yunzez's repro.py (same directory; build_prompt is imported from it so
prompt reconstruction is bit-identical) with:

  * token-ID-level capture: `return_tokens_as_token_ids` + `logprobs=N`
    so every run stores token IDs, per-token logprobs, and top-N alternatives
  * raw JSON persistence of every run (for later first-divergence anatomy)
  * /metrics scrape (preemptions, request counts) before/after every run
  * HTTP error capture (status + body) per request  — a rejected request is
    recorded as status!=200, not silently dropped
  * control modes: --serialize (concurrency 1, no cancels, offsets ignored)
                   --no-cancel (trace timing kept, cancel events stripped)
  * analysis: per-request distinct outputs across runs at text AND token level,
    first-divergence token index, and top-2 logprob gap at that index

Usage:
    python3 repro_ext.py --base-url http://localhost:8000 --runs 10 \
        --label armA_default --out-dir /workspace/results/ext
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from repro import build_prompt  # noqa: E402  (bit-identical prompts)

FINDINGS = [
    "finding_00450_862114934.json",
    "finding_01410_1760617970.json",
    "finding_00030_999829240.json",
]

METRICS_RE = re.compile(
    r"^(vllm:num_preemptions_total|vllm:request_success_total)(\{[^}]*\})? (\S+)",
    re.M,
)


async def scrape_metrics(client: httpx.AsyncClient, base_url: str) -> dict[str, float]:
    try:
        text = (await client.get(f"{base_url}/metrics")).text
    except Exception:
        return {}
    out: dict[str, float] = {}
    for m in METRICS_RE.finditer(text):
        key = m.group(1) + (m.group(2) or "")
        out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


async def _send_stream(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    event: dict,
    cancel_ms: Optional[float],
    logprobs: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": build_prompt(event),
        "max_tokens": event.get("max_tokens") or 64,
        "temperature": 0.0,
        "stream": True,
    }
    if logprobs > 0:
        payload["logprobs"] = logprobs
        payload["return_tokens_as_token_ids"] = True

    headers = {"X-Request-Id": event["request_id"]}
    rec: dict = {
        "request_id": event["request_id"],
        "status": None,
        "error": None,
        "text": "",
        "token_ids": [],
        "token_logprobs": [],
        "top_logprobs": [],
        "finish_reason": None,
        "cancelled_at_ms": cancel_ms,
    }

    async def _collect() -> None:
        async with client.stream(
            "POST", f"{base_url}/v1/completions", json=payload, headers=headers
        ) as resp:
            rec["status"] = resp.status_code
            if resp.status_code != 200:
                body = await resp.aread()
                rec["error"] = body.decode(errors="replace")[:500]
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if raw.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(raw)
                    choice = obj["choices"][0]
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                text = choice.get("text", "")
                if text:
                    rec["text"] += text
                if choice.get("finish_reason"):
                    rec["finish_reason"] = choice["finish_reason"]
                lp = choice.get("logprobs")
                if lp:
                    for tok in lp.get("tokens") or []:
                        # "token_id:NNN" when return_tokens_as_token_ids
                        if isinstance(tok, str) and tok.startswith("token_id:"):
                            rec["token_ids"].append(int(tok.split(":", 1)[1]))
                        else:
                            rec["token_ids"].append(tok)
                    rec["token_logprobs"].extend(lp.get("token_logprobs") or [])
                    rec["top_logprobs"].extend(lp.get("top_logprobs") or [])

    try:
        if cancel_ms is not None:
            try:
                await asyncio.wait_for(_collect(), timeout=cancel_ms / 1000.0)
            except asyncio.TimeoutError:
                pass  # client-side cancel, same mechanism as repro.py
        else:
            await _collect()
    except Exception as exc:  # connection errors etc.
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def prepare_events(events: list[dict], no_cancel: bool) -> tuple[list[dict], dict]:
    sends = sorted(
        [e for e in events if e["action"] == "send"], key=lambda e: e["offset_ms"]
    )
    # dedupe identical request_ids, later offset wins (repro.py semantics)
    seen: dict[str, dict] = {}
    for s in sends:
        seen[s["request_id"]] = s
    sends = sorted(seen.values(), key=lambda e: e["offset_ms"])
    cancels = (
        {}
        if no_cancel
        else {e["request_id"]: e["offset_ms"] for e in events if e["action"] == "cancel"}
    )
    return sends, cancels


async def replay_once(
    base_url: str,
    model: str,
    events: list[dict],
    logprobs: int,
    serialize: bool,
    no_cancel: bool,
) -> dict[str, dict]:
    sends, cancels = prepare_events(events, no_cancel or serialize)
    outputs: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=300) as client:
        if serialize:
            for e in sends:
                outputs[e["request_id"]] = await _send_stream(
                    client, base_url, model, e, None, logprobs
                )
            return outputs

        t0 = asyncio.get_event_loop().time()

        async def run_send(event: dict) -> tuple[str, dict]:
            rid = event["request_id"]
            delay_s = event["offset_ms"] / 1000.0
            await asyncio.sleep(
                max(0.0, delay_s - (asyncio.get_event_loop().time() - t0))
            )
            cancel_offset = cancels.get(rid)
            cancel_ms = None
            if cancel_offset is not None and cancel_offset > event["offset_ms"]:
                cancel_ms = float(cancel_offset - event["offset_ms"])
            return rid, await _send_stream(
                client, base_url, model, event, cancel_ms, logprobs
            )

        results = await asyncio.gather(
            *[run_send(e) for e in sends], return_exceptions=True
        )
    for item in results:
        if isinstance(item, BaseException):
            continue
        rid, rec = item
        outputs[rid] = rec
    return outputs


def first_div_index(a: list, b: list) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1  # -1 = identical


def analyze(finding_label: str, runs: list[dict[str, dict]], expected: list[str]) -> dict:
    rids = sorted({rid for r in runs for rid in r})
    report: dict = {"finding": finding_label, "requests": {}, "diverged": [], "clean": []}
    for rid in rids:
        recs = [r[rid] for r in runs if rid in r]
        ok = [r for r in recs if r["status"] == 200 and not r["error"]]
        statuses = sorted({str(r["status"]) for r in recs})
        texts = [r["text"] for r in ok]
        tokens = [tuple(r["token_ids"]) for r in ok]
        distinct_texts = len(set(texts))
        distinct_tokens = len(set(tokens))
        # first divergence vs the first ok run, plus top-2 gap there
        fdi, gap = None, None
        if distinct_tokens > 1 and ok and ok[0]["token_ids"]:
            ref = ok[0]
            for r in ok[1:]:
                idx = first_div_index(ref["token_ids"], r["token_ids"])
                if idx >= 0 and (fdi is None or idx < fdi):
                    fdi = idx
            if fdi is not None and fdi < len(ref["top_logprobs"]) and ref["top_logprobs"][fdi]:
                tops = sorted(ref["top_logprobs"][fdi].values(), reverse=True)
                if len(tops) >= 2:
                    gap = tops[0] - tops[1]
        entry = {
            "n_recs": len(recs),
            "n_ok": len(ok),
            "statuses": statuses,
            "errors": sorted({r["error"][:120] for r in recs if r["error"]}),
            "distinct_texts": distinct_texts,
            "distinct_token_seqs": distinct_tokens,
            "gen_lens": sorted({len(t) for t in tokens}),
            "first_div_token_idx": fdi,
            "top2_gap_at_div": gap,
            "cancelled": any(r["cancelled_at_ms"] is not None for r in recs),
            "expected_diverge": rid in expected,
        }
        report["requests"][rid] = entry
        if distinct_tokens > 1 or distinct_texts > 1:
            report["diverged"].append(rid)
        elif len(ok) >= 2:
            report["clean"].append(rid)
    return report


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--logprobs", type=int, default=5)
    ap.add_argument("--serialize", action="store_true")
    ap.add_argument("--no-cancel", action="store_true")
    ap.add_argument("--findings", nargs="*", default=FINDINGS)
    ap.add_argument("--run-marker", action="store_true")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="/workspace/results/ext")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")

    async with httpx.AsyncClient(timeout=30) as client:
        model = (await client.get(f"{base_url}/v1/models")).json()["data"][0]["id"]
        try:
            version = (await client.get(f"{base_url}/version")).json()
        except Exception:
            version = None
        metrics_start = await scrape_metrics(client, base_url)

    stamp = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": model,
        "server_version": version,
        "label": args.label,
        "args": vars(args),
        "metrics_at_start": metrics_start,
    }
    print(json.dumps(stamp, indent=1))

    script_dir = Path(__file__).parent
    for name in args.findings:
        finding = json.loads((script_dir / name).read_text())
        events = finding["trace"]["events"]
        ev = finding.get("reproduction", {}).get("evidence", {})
        expected = (
            (ev.get("kv_corruption") or {}).get("diverged_requests")
            or (ev.get("inconsistency") or {}).get("diverged_requests")
            or []
        )
        runs: list[dict[str, dict]] = []
        per_run_metrics: list[dict] = []
        for i in range(args.runs):
            if args.run_marker:
                async with httpx.AsyncClient(timeout=60) as mc:
                    await mc.post(
                        f"{base_url}/v1/completions",
                        json={"model": model, "prompt": "x", "max_tokens": 1,
                              "temperature": 0.0},
                        headers={"X-Request-Id": f"RUNMARK-{args.label}-{name[8:13]}-{i}"},
                    )
            async with httpx.AsyncClient(timeout=30) as mc:
                m_before = await scrape_metrics(mc, base_url)
            outputs = await replay_once(
                base_url, model, events, args.logprobs, args.serialize, args.no_cancel
            )
            async with httpx.AsyncClient(timeout=30) as mc:
                m_after = await scrape_metrics(mc, base_url)
            runs.append(outputs)
            per_run_metrics.append({"before": m_before, "after": m_after})
            n_div_so_far = len(
                [
                    rid
                    for rid in outputs
                    if len(
                        {
                            tuple(r[rid]["token_ids"])
                            for r in runs
                            if rid in r and r[rid]["status"] == 200
                        }
                    )
                    > 1
                ]
            )
            print(f"  [{name}] run {i + 1}/{args.runs} done; diverging-so-far={n_div_so_far}", flush=True)

        report = analyze(name, runs, expected)
        blob = {
            "stamp": stamp,
            "finding": name,
            "expected_diverged": expected,
            "report": report,
            "per_run_metrics": per_run_metrics,
            "runs": runs,
        }
        out_path = out_dir / f"{args.label}__{name.replace('.json', '')}.json"
        out_path.write_text(json.dumps(blob))
        print(f"\n==== {name} [{args.label}] ====")
        print(f" diverged ({len(report['diverged'])}): {report['diverged']}")
        print(f" clean    ({len(report['clean'])}): {report['clean']}")
        print(f" expected ({len(expected)}): {expected}")
        for rid, e in sorted(report["requests"].items()):
            print(
                "  %-22s ok=%d/%d st=%s dtext=%d dtok=%d lens=%s fdi=%s gap=%s%s%s"
                % (
                    rid,
                    e["n_ok"],
                    e["n_recs"],
                    ",".join(e["statuses"]),
                    e["distinct_texts"],
                    e["distinct_token_seqs"],
                    e["gen_lens"][:4],
                    e["first_div_token_idx"],
                    None if e["top2_gap_at_div"] is None else round(e["top2_gap_at_div"], 4),
                    " CANCELLED" if e["cancelled"] else "",
                    " EXPECTED-DIV" if e["expected_diverge"] else "",
                )
            )
        preempt_keys = [k for k in metrics_start if "preempt" in k]
        for k in preempt_keys:
            start = metrics_start.get(k, 0)
            end = per_run_metrics[-1]["after"].get(k, 0)
            print(f" {k}: start={start} end={end}")
        print(f" raw saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
