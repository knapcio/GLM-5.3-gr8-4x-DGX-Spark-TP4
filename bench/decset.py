#!/usr/bin/env python3
"""decset.py: c1 decode over a fixed set of 24 prompts (12 prose, 6 code, 6 structured), T=0, effort low, 384 tokens.
Reports per kind: total completion tokens / total decode time (first token -> end), and the median per-prompt tok/s.
  python3 decset.py [--url http://127.0.0.1:8100] [--max-tokens 384]"""
import argparse, json, statistics as st, time, urllib.request
P = {
 "prose": [
  "Write a short story about a lighthouse keeper who finds a message in a bottle.",
  "Explain to a curious teenager why the sky is blue and sunsets are red.",
  "Write a persuasive essay arguing that cities should plant more trees.",
  "Describe a busy morning market in a coastal town, using vivid sensory detail.",
  "Write a letter from a retired teacher to a former student who just graduated.",
  "Summarize the causes and consequences of the printing press for European society.",
  "Write a reflective piece on what it means to learn a new language as an adult.",
  "Explain how vaccines train the immune system, in plain language.",
  "Write the opening chapter of a mystery novel set on a night train.",
  "Compare the advantages of living in a small village versus a large city.",
  "Write a product announcement for a quiet, energy-efficient home heat pump.",
  "Describe the life cycle of a star from nebula to remnant for a general audience."],
 "code": [
  "Write a Python function that parses an ISO-8601 duration string into seconds, with tests.",
  "Implement an LRU cache class in TypeScript with get/put in O(1) and explain the design.",
  "Write a Rust function that merges overlapping intervals and a few unit tests.",
  "Write a bash script that rotates log files older than 7 days and compresses them.",
  "Implement Dijkstra's shortest path in C++ using a priority queue, with a small example.",
  "Write a SQL schema for a library system (books, members, loans) and three useful queries."],
 "structured": [
  "Produce a JSON array of 12 fictional employees with fields id, name, department, salary, start_date.",
  "Create a markdown table of the 15 largest countries by area with columns rank, country, area_km2, capital.",
  "Output a YAML config for a web service with sections server, database, cache, logging, and feature_flags.",
  "List 20 common HTTP status codes as CSV with columns code, name, category, description.",
  "Generate a JSON schema for an order object with nested customer, items and shipping address.",
  "Write a weekly meal plan as a markdown table: days as rows, breakfast/lunch/dinner as columns."]}


def run(url, prompt, mt):
    body = {"model": "glm-5.3", "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": mt, "chat_template_kwargs": {"reasoning_effort": "low"},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    first = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            l = raw.decode().strip()
            if not l.startswith("data:") or l[5:].strip() == "[DONE]":
                continue
            j = json.loads(l[5:])
            if j.get("error"):
                raise RuntimeError(j["error"])
            usage = j.get("usage") or usage
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    first = first or time.monotonic()
    end = time.monotonic()
    if not usage or not usage.get("completion_tokens") or first is None:
        raise RuntimeError("stream ended without usage")
    return usage["completion_tokens"], end - first


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8100")
ap.add_argument("--max-tokens", type=int, default=384)
a = ap.parse_args()
run(a.url, "Say hi.", 16)
out = {}
for kind, ps in P.items():
    tt = dt = 0; per = []
    for p in ps:
        t, d = run(a.url, p, a.max_tokens); tt += t; dt += d; per.append(t / d)
    out[kind] = {"agg": round(tt / dt, 2), "median": round(st.median(per), 2), "per": [round(x, 1) for x in per]}
    print(json.dumps({kind: out[kind]}), flush=True)
print(json.dumps({"decset": {k: v["agg"] for k, v in out.items()}}), flush=True)
