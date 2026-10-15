#!/usr/bin/env python3
"""Multi-turn DFlash2-reuse fleet test (G6_DFL_REUSE, gr8 KNOWN-ISSUES #1). Python 3 stdlib only.

At TP=4 a request drafts with DFlash2 only when it is alone at its prompt tail and the prefix-reuse hit did not skip
into the last 2,048 prompt positions (the drafter's window). Under --s2 the reuse skip is a multiple of the 4,096-row
prefill grid (G6_REUSE_ALIGN), so turn N+1 lands in the window when (turn N's prompt mod 4096) + the new tokens < 2048.
This script builds conversations that hit that case on purpose, plus the cases around it:

  A1  fresh code prompt, 4096*k + ~300 tokens                  -> DFlash2 in every arm (writes the drafter ring)
  A2  A1 + reply + short ask: the hit skips into the window    -> off: MTP only; on: DFlash2 (ring rows of A1)
  A3  A2 + reply + short ask                                   -> printed case (in or out of the window)
  B1  another conversation, fresh                              -> DFlash2 (the ring is now B's)
  B2  B1 continued, hit into the window                        -> off: MTP; on: DFlash2 (ring of B1)
  A4  A continued after B: the ring is B's                     -> off: MTP if in window; on: taps-only context
                                                                  (MTP warm-up if < G6_DFL_REUSE_MIN rows)
  A4r the same prompt again (replay: no prefill at all)        -> off: MTP; on: DFlash2 (ring of A4)
  C1  long code request + a short one 1 s later (concurrent)   -> off/1: C1 MTP for life; 2: promoted after X ends
  C1x the short concurrent request

Per request it records prompt / completion tokens, TTFT, decode tok/s, the SHA-256 of reasoning + content (T=0
identity between arms), the /metrics deltas (spec rounds, draft tokens, accepted, accepted per position, prefix-cache
hits) and a drafter verdict: with --mtp 2 (the TP4 default) an accepted draft at position >= 2 can only be DFlash2's.
With --log (the rank-0 engine log, readable from where this runs) it also keeps the engine's lines for the request:
"# s2 reuse request", "# s2 dflash slot .. prompt context", "# s2 drafter choice", and with the flag on
"# s2 dflash reuse slot ..: N of M rounds on DFlash2".

Arms (one fresh boot each, nothing else on the endpoint; the first request after a boot is a discarded warm-up):
  off:  (no G6_DFL_REUSE in G6ENV_X)       python3 multiturn.py run --label off --out off.json [--log rank0.log]
  on1:  G6_DFL_REUSE=1 in G6ENV_X         python3 multiturn.py run --label on1 --out on1.json
  on2:  G6_DFL_REUSE=2 in G6ENV_X         python3 multiturn.py run --label on2 --out on2.json
  then: python3 multiturn.py compare off.json on1.json on2.json
For more samples use --repeat N (fresh conversation text per repeat, same text in every arm for a given seed).
"""
import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

WIN = 2048


# ----------------------------------------------------------------------------------------------- HTTP
def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_text(base, path, timeout=60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read().decode()


def metrics(base):
    """the counters this test reads from /metrics (vLLM names, gr8 frontend metrics.rs)"""
    m = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "per_pos": {}, "pc_hits": 0.0, "pc_queries": 0.0, "prompt": 0.0, "gen": 0.0}
    names = {
        "vllm:spec_decode_num_drafts_total": "drafts",
        "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted",
        "vllm:prefix_cache_hits_total": "pc_hits",
        "vllm:prefix_cache_queries_total": "pc_queries",
        "vllm:prompt_tokens_total": "prompt",
        "vllm:generation_tokens_total": "gen",
    }
    try:
        text = get_text(base, "/metrics")
    except Exception as e:  # metrics are a diagnostic: the run goes on without them
        print(f"# warning: /metrics: {e}", file=sys.stderr)
        return None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        mm = re.match(r"^([^{\s]+)(\{[^}]*\})?\s+(\S+)$", line)
        if not mm:
            continue
        name, labels, val = mm.group(1), mm.group(2) or "", float(mm.group(3))
        if name in names:
            m[names[name]] = val
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            p = re.search(r'position="(\d+)"', labels)
            if p:
                m["per_pos"][int(p.group(1))] = val
    return m


def mdelta(a, b):
    if a is None or b is None:
        return None
    d = {k: b[k] - a[k] for k in ("drafts", "draft_tokens", "accepted", "pc_hits", "pc_queries", "prompt", "gen")}
    pos = sorted(set(a["per_pos"]) | set(b["per_pos"]))
    d["per_pos"] = [b["per_pos"].get(i, 0) - a["per_pos"].get(i, 0) for i in range(max(pos) + 1)] if pos else []
    return d


def tokenize(base, model, messages, ctk):
    r = post(base, "/tokenize", {"model": model, "messages": messages, "chat_template_kwargs": ctk, "add_generation_prompt": True})
    return r["tokens"]


def chat_stream(base, model, messages, ctk, max_tokens, seed):
    """one streamed chat completion at temperature 0: text, reasoning, usage and timings"""
    body = {"model": model, "messages": messages, "temperature": 0, "seed": seed, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": ctk}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_last = None
    content, reasoning, usage, finish = [], [], None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = False
                if d.get("content"):
                    content.append(d["content"]); piece = True
                rc = d.get("reasoning_content") or d.get("reasoning")
                if rc:
                    reasoning.append(rc); piece = True
                if piece:
                    now = time.time()
                    t_first = t_first or now
                    t_last = now
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.time()
    return {"content": "".join(content), "reasoning": "".join(reasoning), "usage": usage or {}, "finish": finish,
            "t0": t0, "t_first": t_first, "t_last": t_last, "t_end": t_end}


# ----------------------------------------------------------------------------------------------- conversation text
VERBS = ["parse", "merge", "score", "render", "index", "split", "encode", "decode", "filter", "rank", "bucket", "align"]
NOUNS = ["record", "ledger", "invoice", "token", "segment", "packet", "frame", "route", "batch", "window", "shard", "cell"]


def code_lines(rng, tag):
    """an endless deterministic Python module: small typed functions and classes (code is where DFlash2 wins)"""
    i = 0
    while True:
        v, n = rng.choice(VERBS), rng.choice(NOUNS)
        f = f"{v}_{n}_{tag}{i}"
        a, b = rng.randint(2, 97), rng.randint(2, 97)
        if i % 5 == 4:
            yield from [
                f"class {n.capitalize()}{tag}{i}:",
                f'    """A {n} kept by the {v} stage ({i})."""',
                "",
                "    def __init__(self, items: list[int], limit: int = %d) -> None:" % a,
                "        self.items = list(items)",
                "        self.limit = limit",
                "",
                "    def total(self) -> int:",
                "        return sum(x for x in self.items if x < self.limit)",
                "",
                "    def top(self, k: int = %d) -> list[int]:" % (b % 7 + 1),
                "        return sorted(self.items, reverse=True)[:k]",
                "",
            ]
        else:
            yield from [
                f"def {f}(values: list[int], factor: int = {a}) -> dict[str, int]:",
                f'    """{v.capitalize()} each {n} and count the results modulo {b}."""',
                "    out: dict[str, int] = {}",
                "    for k, x in enumerate(values):",
                f"        key = f\"{n}-{{(x * factor + k) % {b}}}\"",
                "        out[key] = out.get(key, 0) + 1",
                "    return out",
                "",
            ]
        i += 1


def size_context(base, model, ctk, system, ask, rng_seed, tag, target, slack):
    """the module text whose turn-1 prompt has target .. target + slack tokens (binary search over its lines)"""
    lines = []
    gen = code_lines(random.Random(rng_seed), tag)

    def msgs(n):
        while len(lines) < n:
            lines.append(next(gen))
        body = "```python\n" + "\n".join(lines[:n]) + "\n```\n\n" + ask
        return [{"role": "system", "content": system}, {"role": "user", "content": body}]

    lo, hi = 1, 64
    while len(tokenize(base, model, msgs(hi), ctk)) < target:
        lo, hi = hi, hi * 2
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenize(base, model, msgs(mid), ctk)) < target:
            lo = mid + 1
        else:
            hi = mid
    m = msgs(lo)
    n = len(tokenize(base, model, m, ctk))
    if n > target + slack:
        print(f"# note: {tag}: prompt {n} tokens, wanted {target}..{target + slack}", file=sys.stderr)
    return m


# ----------------------------------------------------------------------------------------------- prediction
def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def predict(prev_prompts, ids, grid):
    """the --s2 reuse skip (prefix_reuse.h grid rule; retained prefill rows only) and whether it lands in the window"""
    n = len(ids)
    best = 0
    for p in prev_prompts:
        l = lcp(p, ids)
        pf_end = len(p) - 1                       # rows the earlier request prefilled (decode rows never count)
        skip = min(l - 1, n - 1)                  # MTP: the retained prefix stops a row before the first difference
        if skip <= 0:
            continue
        s = pf_end if (skip >= pf_end and n - 1 == pf_end) else min(skip, pf_end) // grid * grid
        best = max(best, s)
    wlo = max(0, n - 1 - WIN)
    case = "fresh" if best == 0 else ("replay" if best == n - 1 else ("hit-in-window" if best > wlo else "hit-outside-window"))
    return {"skip": best, "window_lo": wlo, "gap": n - 1 - best, "case": case}


def verdict(d, mtp):
    if not d or not d["drafts"]:
        return "unknown"
    hi = sum(d["per_pos"][mtp:]) if len(d["per_pos"]) > mtp else 0
    k = d["draft_tokens"] / d["drafts"]
    return "dflash" if hi > 0 or k > mtp + 0.01 else "mtp"


# ----------------------------------------------------------------------------------------------- engine log
class LogTail:
    KEEP = ("# s2 reuse request", "# s2 dflash slot", "# s2 drafter choice slot", "# s2 dflash reuse", "# s2 mode rank 0")

    def __init__(self, path):
        self.path, self.pos = path, 0
        if path:
            try:
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    self.pos = f.tell()
            except OSError as e:
                print(f"# warning: --log {path}: {e}", file=sys.stderr)
                self.path = None

    def new_lines(self):
        if not self.path:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        return [l for l in data.decode("utf-8", "replace").splitlines() if l.startswith(self.KEEP)]


# ----------------------------------------------------------------------------------------------- run
def run(a):
    base = a.base.rstrip("/")
    ctk = {"reasoning_effort": a.effort}
    log = LogTail(a.log)
    records = []
    prev_prompts = []                             # every request's prompt ids so far (the pool retains their runs)

    def one(name, messages, max_tokens, note=""):
        ids = tokenize(base, a.model, messages, ctk)
        pr = predict(prev_prompts, ids, a.grid)
        m0 = metrics(base)
        r = chat_stream(base, a.model, messages, ctk, max_tokens, a.seed)
        time.sleep(0.5)                           # the engine's terminal line and counters land after the last chunk
        m1 = metrics(base)
        prev_prompts.append(ids)
        rec = summarize(name, ids, pr, r, mdelta(m0, m1), log.new_lines(), note)
        records.append(rec)
        show(rec, a.mtp)
        return r

    for rep in range(a.repeat):
        seed = a.seed + 1000 * rep
        sys_code = "You are a senior Python engineer. Answer with code first, then at most three short sentences."
        print(f"# repeat {rep}: sizing prompts", file=sys.stderr)
        if rep == 0:
            one("warmup", [{"role": "user", "content": "Say hello in five words."}], 32, "discarded: first run after a boot")
        # A: 4096*2 + ~300 tokens, so turn 2's skip (8192) lands in its window
        A = size_context(base, a.model, ctk, sys_code, "Write pytest unit tests for the first three functions above.",
                         seed + 1, f"a{rep}", 2 * a.grid + a.offset, 150)
        r = one(f"r{rep}.A1", A, a.max_tokens)
        A = A + [{"role": "assistant", "content": r["content"]},
                 {"role": "user", "content": "Now refactor the first class above into a dataclass. Output the full new class only."}]
        r = one(f"r{rep}.A2", A, a.max_tokens)
        A = A + [{"role": "assistant", "content": r["content"]},
                 {"role": "user", "content": "Add a function that merges two of those dataclass instances, with type hints."}]
        r = one(f"r{rep}.A3", A, a.max_tokens)
        A = A + [{"role": "assistant", "content": r["content"]}]
        # B: another conversation, 4096 + ~300; B2 continues it (the ring is B1's)
        B = size_context(base, a.model, ctk, sys_code, "Write a docstring-complete version of the second function above.",
                         seed + 2, f"b{rep}", a.grid + a.offset, 150)
        r = one(f"r{rep}.B1", B, a.max_tokens)
        B = B + [{"role": "assistant", "content": r["content"]},
                 {"role": "user", "content": "Rewrite it to accept an iterator instead of a list. Code only."}]
        one(f"r{rep}.B2", B, a.max_tokens)
        # A again after B: the drafter ring holds B's tokens
        A = A + [{"role": "user", "content": "Write a short README section (markdown) describing the merge function."}]
        one(f"r{rep}.A4", A, a.max_tokens)
        # the same prompt again (retry / regenerate): every prompt row but the last is reused, no prefill at all
        one(f"r{rep}.A4r", A, a.max_tokens, "replay of A4")
        # C: a long request with a short one 1 s into its prefill (not alone at its prompt tail)
        C = size_context(base, a.model, ctk, sys_code, "Write a command-line interface (argparse) wrapping every function above.",
                         seed + 3, f"c{rep}", 2 * a.grid + a.offset, 150)
        X = [{"role": "user", "content": "Reply with exactly one word: ready."}]
        cid = tokenize(base, a.model, C, ctk)
        xid = tokenize(base, a.model, X, ctk)
        pc = predict(prev_prompts, cid, a.grid)
        px = predict(prev_prompts, xid, a.grid)
        m0 = metrics(base)
        out = {}

        def worker():
            out["x"] = chat_stream(base, a.model, X, ctk, 16, a.seed)

        t = threading.Thread(target=worker)
        timer = threading.Timer(a.concurrent_delay, t.start)
        timer.start()
        rc = chat_stream(base, a.model, C, ctk, a.max_tokens, a.seed)
        timer.join()
        t.join()
        time.sleep(0.5)
        m1 = metrics(base)
        prev_prompts.extend([cid, xid])
        lines = log.new_lines()
        d = mdelta(m0, m1)
        rec = summarize(f"r{rep}.C1", cid, pc, rc, d, lines, "concurrent with C1x; metrics are both requests'")
        records.append(rec)
        show(rec, a.mtp)
        rec = summarize(f"r{rep}.C1x", xid, px, out["x"], None, [], "the short concurrent request")
        records.append(rec)
        show(rec, a.mtp)

    res = {"label": a.label, "base": base, "model": a.model, "effort": a.effort, "grid": a.grid, "mtp": a.mtp,
           "max_tokens": a.max_tokens, "seed": a.seed, "repeat": a.repeat, "records": records}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"# wrote {a.out}", file=sys.stderr)


def summarize(name, ids, pr, r, d, lines, note):
    u = r["usage"]
    ct = u.get("completion_tokens") or 0
    dec = (r["t_last"] - r["t_first"]) if r["t_first"] and r["t_last"] else None
    h = hashlib.sha256((r["reasoning"] + "\x00" + r["content"]).encode()).hexdigest()[:16]
    return {"name": name, "prompt_tokens": u.get("prompt_tokens", len(ids)), "completion_tokens": ct,
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": r["finish"], "ttft_s": round(r["t_first"] - r["t0"], 3) if r["t_first"] else None,
            "decode_tps": round((ct - 1) / dec, 2) if dec and ct > 1 else None,
            "e2e_s": round(r["t_end"] - r["t0"], 3), "sha": h, "predicted": pr, "metrics": d,
            "drafter": None, "log": lines, "note": note}


def show(rec, mtp):
    d = rec["metrics"]
    spec = "-"
    if d and d["drafts"]:
        spec = (f"rounds {d['drafts']:.0f} tok/round {1 + d['accepted'] / d['drafts']:.2f} k {d['draft_tokens'] / d['drafts']:.2f} "
                f"pos {[int(x) for x in d['per_pos']]} hits {d['pc_hits']:.0f}")
    rec["drafter"] = None if d is None else verdict(d, mtp)
    p = rec["predicted"]
    print(f"{rec['name']:>9} prompt {rec['prompt_tokens']:>6} gen {rec['completion_tokens']:>5} ttft {rec['ttft_s']}s "
          f"decode {rec['decode_tps']} tok/s  {p['case']} (skip {p['skip']}, gap {p['gap']})  drafter {rec['drafter']}  "
          f"sha {rec['sha']}  {spec}")
    for l in rec["log"]:
        print(f"{'':>11}{l}")


# ----------------------------------------------------------------------------------------------- compare
def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    ref = runs[0]
    names = [r["name"] for r in ref["records"] if r["name"] != "warmup"]
    by = [{r["name"]: r for r in run["records"]} for run in runs]
    labels = [run["label"] for run in runs]
    print("request   case                " + "".join(f"{l:>22}" for l in labels) + "   identical")
    ratios = {l: {} for l in labels[1:]}
    same_all = True
    for n in names:
        row = [b.get(n) for b in by]
        case = row[0]["predicted"]["case"] if row[0] else "?"
        cells = []
        for r in row:
            cells.append("missing" if r is None else f"{r['decode_tps'] or 0:7.2f} {str(r['drafter'] or '-'):>6} {r['ttft_s'] or 0:5.1f}s")
        same = all(r is not None and r["sha"] == row[0]["sha"] for r in row)
        same_all &= same
        print(f"{n:<9} {case:<19} " + "".join(f"{c:>22}" for c in cells) + f"   {'yes' if same else 'NO'}")
        for l, r in zip(labels[1:], row[1:]):
            if r and row[0] and r["decode_tps"] and row[0]["decode_tps"]:
                ratios[l].setdefault(case, []).append(r["decode_tps"] / row[0]["decode_tps"])
    print("\ncells: decode tok/s, drafter verdict (from /metrics: an accepted draft at position >= --mtp is DFlash2's), TTFT")
    for l in labels[1:]:
        for case, v in sorted(ratios[l].items()):
            g = 1.0
            for x in v:
                g *= x
            g **= 1.0 / len(v)
            print(f"{l} vs {labels[0]}: {case:<19} decode x{g:.3f} over {len(v)} requests")
    print(f"\nT=0 outputs identical across arms: {'yes' if same_all else 'NO (see the rows marked NO)'}")
    return 0 if same_all else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--base", default="http://127.0.0.1:8100")
    r.add_argument("--model", default="glm-5.3")
    r.add_argument("--effort", default="low", help="chat_template_kwargs reasoning_effort")
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--grid", type=int, default=4096, help="the S2 prefill grid (--prefill rows; G6_REUSE_ALIGN)")
    r.add_argument("--offset", type=int, default=300, help="turn-1 prompt tokens past a grid multiple")
    r.add_argument("--mtp", type=int, default=2, help="the engine's --mtp (MTP drafts per round)")
    r.add_argument("--concurrent-delay", type=float, default=1.0, help="seconds after C1 starts that C1x is sent")
    r.add_argument("--log", help="the rank-0 engine log (stderr), to keep each request's drafter lines")
    c = sp.add_parser("compare")
    c.add_argument("files", nargs="+", help="the reference arm first (flag off)")
    a = ap.parse_args()
    if a.cmd == "compare":
        sys.exit(compare(a.files))
    if a.base.endswith("/v1/chat/completions"):
        a.base = a.base[: -len("/v1/chat/completions")]
    run(a)


if __name__ == "__main__":
    main()
