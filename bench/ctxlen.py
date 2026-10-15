#!/usr/bin/env python3
"""Decode speed vs context length (README "Decode vs context length" table).

For each added-context size it builds a prompt with a distinct random filler prefix (no prefix-cache reuse), asks
one question about it, and streams `max_tokens` output tokens at temperature 0, reasoning_effort low, c1. Reports
actual prompt tokens (from usage), TTFT and decode tok/s (completion tokens / (end - first token)), median of `reps`.

  python3 bench/ctxlen.py [--url http://127.0.0.1:8100] [--sizes 0,16384,32768,65536,98304,122880] [--reps 3]
"""
import argparse, json, random, statistics as st, time, urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
         "romeo sierra tango uniform victor whiskey xray yankee zulu").split()


def filler(n_tokens, seed):
    r = random.Random(seed)
    # ~1.3 tokens per word for this vocabulary; numbered lines keep every prefix distinct
    lines, words = [], 0
    while words < n_tokens / 1.3:
        line = f"{len(lines)}: " + " ".join(r.choice(WORDS) for _ in range(12))
        lines.append(line); words += 13
    return "\n".join(lines)


def run(url, model, prompt, max_tokens):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"reasoning_effort": "low"},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); first = None; usage = None; n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("error"):
                raise RuntimeError(f"stream error: {j['error']}")
            usage = j.get("usage") or usage
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    n += 1; first = first or time.monotonic()
    end = time.monotonic(); first = first or end
    if not usage or not usage.get("completion_tokens"):
        raise RuntimeError("stream ended without usage (truncated or failed)")
    tok = usage["completion_tokens"]
    return {"prompt_tokens": (usage or {}).get("prompt_tokens"), "ttft": first - t0, "tokens": tok,
            "decode_tps": tok / max(1e-6, end - first)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--sizes", default="0,16384,32768,65536,98304,122880")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    q = "Summarize in three sentences what kind of text appears above and how many numbered lines it has."
    run(a.url, a.model, "Say hi.", 16)                               # warm-up, discarded
    table = []
    for size in map(int, a.sizes.split(",")):
        res = []
        for r in range(a.reps):
            prompt = (filler(size, seed=size * 100 + r + int(time.time())) + "\n\n" if size else "") + q
            x = run(a.url, a.model, prompt, a.max_tokens); x.update(size=size, rep=r)
            print(json.dumps(x), flush=True); res.append(x)
        table.append({"size": size, "prompt_tokens": st.median(x["prompt_tokens"] or 0 for x in res),
                      "decode_tps": round(st.median(x["decode_tps"] for x in res), 1),
                      "ttft": round(st.median(x["ttft"] for x in res), 2)})
    print(json.dumps({"table": table, "max_tokens": a.max_tokens, "reps": a.reps}), flush=True)


main()
