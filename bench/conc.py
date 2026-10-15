#!/usr/bin/env python3
"""Decode table by prompt type and concurrency (README "Decode" table).

For each workload (RigMark's prose, code and structured prompts) and each concurrency c in 1, 2, 4, it sends c
identical streaming requests at once, `rounds` times, and reports the median over rounds of
  aggregate tok/s  = all completion tokens / wall time of the round (first request sent -> last stream done)
  per-stream tok/s = completion tokens / (stream end - first token), median over the streams of a round.
Thinking stays on at reasoning_effort low (the model's chat template has no thinking-off mode), temperature 0.

  python3 bench/conc.py --prompts <rigmark>/prompts.json [--url http://127.0.0.1:8100] [--rounds 3] [--max-tokens 256]

Prints one JSON line per round and a final {"table": ...} line.
"""
import argparse, json, statistics as st, threading, time, urllib.request


def stream(url, body, out, errs=None):
    try:
        _stream(url, body, out)
    except Exception as e:  # noqa: BLE001 - reported by the caller
        (errs if errs is not None else []).append(repr(e))
        if errs is None:
            raise


def _stream(url, body, out):
    t0 = time.monotonic(); first = None; n = 0; usage = None
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("error"):
                raise RuntimeError(f"stream error: {j['error']}")
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    n += 1
                    if first is None:
                        first = time.monotonic()
    end = time.monotonic()
    if not usage or not usage.get("completion_tokens"):
        raise RuntimeError("stream ended without usage (truncated or failed)")
    tok = usage["completion_tokens"]
    out.append({"t0": t0, "first": first or end, "end": end, "tokens": tok})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--conc", default="1,2,4")
    a = ap.parse_args()
    P = json.load(open(a.prompts))
    table = {}
    for kind in ("prose", "code", "structured"):
        body = {"model": a.model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
                "max_tokens": a.max_tokens, "chat_template_kwargs": {"reasoning_effort": "low"},
                "messages": [{"role": "system", "content": P["system"]}, {"role": "user", "content": P["workloads"][kind]}]}
        stream(a.url, body, [])                      # warm-up, discarded
        for c in map(int, a.conc.split(",")):
            aggs, pss = [], []
            for r in range(a.rounds):
                out, errs = [], []
                ts = [threading.Thread(target=stream, args=(a.url, body, out, errs)) for _ in range(c)]
                t0 = time.monotonic()
                for t in ts: t.start()
                for t in ts: t.join()
                if errs or len(out) != c:
                    raise SystemExit(f"{kind} c{c} round {r}: incomplete round ({len(out)}/{c} streams): {errs}")
                wall = max(o["end"] for o in out) - t0
                agg = sum(o["tokens"] for o in out) / wall
                ps = st.median(o["tokens"] / max(1e-6, o["end"] - o["first"]) for o in out)
                aggs.append(agg); pss.append(ps)
                print(json.dumps({"kind": kind, "c": c, "round": r, "aggregate": round(agg, 2), "per_stream": round(ps, 2),
                                  "tokens": [o["tokens"] for o in out]}), flush=True)
            table.setdefault(kind, {})[f"c{c}"] = [round(st.median(aggs), 1), round(st.median(pss), 1)]
    print(json.dumps({"table": table, "max_tokens": a.max_tokens, "rounds": a.rounds}), flush=True)


main()
