"""Interleaved ratio: blaze chunked-parallel vs Python tiktoken on one long doc.
Run from the repo root, PLUGGED INTO AC POWER (battery throttles all-core clocks):
    cargo build --release --example hundredx
    python3 scripts/ratio.py
"""
import os, subprocess, time, statistics

import tiktoken

BIG = "data/big.txt"
BIN = "target/release/examples/hundredx"

if not os.path.exists(BIG):
    d = open("data/shakespeare.txt").read()
    open(BIG, "w").write(d * 29)

text = open(BIG).read()
nbytes = len(text.encode())
enc = tiktoken.get_encoding("cl100k_base")
enc.encode_ordinary(text[:100000])  # warm

tk_times, blaze_mbps = [], []
for round_ in range(4):
    if round_:
        time.sleep(20)  # let the package cool between rounds
    t = time.perf_counter()
    n_tk = len(enc.encode_ordinary(text))
    tk_times.append(time.perf_counter() - t)
    out = subprocess.run([BIN, BIG], capture_output=True, text=True, env={**__import__("os").environ, "ITERS": "5"},
                         cwd=".").stdout
    for line in out.splitlines():
        if "chunked-parallel" in line:
            blaze_mbps.append(float(line.split(":")[1].strip().split()[0]))
        if line.startswith("doc:"):
            n_blaze = int(line.split("tokens:")[1])
    assert n_tk == n_blaze, (n_tk, n_blaze)
    print(f"round {round_}: tiktoken {nbytes/1e6/tk_times[-1]:.1f} MB/s, blaze {blaze_mbps[-1]:.0f} MB/s -> {blaze_mbps[-1]/(nbytes/1e6/tk_times[-1]):.1f}x")

tk = nbytes / 1e6 / statistics.median(tk_times)
bl = statistics.median(blaze_mbps)
print(f"\nmedian: tiktoken {tk:.1f} MB/s | blaze {bl:.0f} MB/s | ratio {bl/tk:.1f}x")
print(f"tokens: {n_tk} (identical)")
