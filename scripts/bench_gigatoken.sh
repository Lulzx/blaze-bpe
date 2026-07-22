#!/usr/bin/env bash
# Measure gigatoken (best-of-N) across the same scenarios the README charts use,
# on the same machine, cl100k via Xenova/gpt-4 and o200k via Xenova/gpt-4o.
# Also prints a blaze parallel sanity number to confirm full (plugged-in) clocks.
#
# Usage: bash scripts/bench_gigatoken.sh
set -euo pipefail
cd "$(dirname "$0")/.."

D=/tmp/blazebench
mkdir -p "$D"
python3 - <<'PY'
s = open('data/shakespeare.txt','rb').read()
open('/tmp/blazebench/shake_1mb.txt','wb').write(s)          # 1.12 MB
open('/tmp/blazebench/shake_8mb.txt','wb').write(s*8)        # 8.9 MB
open('/tmp/blazebench/shake_1gb.txt','wb').write(s*900)      # 1.0 GB
open('/tmp/blazebench/small_45.txt','wb').write(s[:45])
open('/tmp/blazebench/small_900.txt','wb').write(s[:900])
open('/tmp/blazebench/small_44k.txt','wb').write(s[:43900])
print("corpora ready")
PY

GT() { uvx --with tokenizers --from gigatoken gigatoken bench "$1" "$2" 2>&1 | grep -oE '[0-9.]+ MB/s|[0-9.]+ Mtok/s' | tr '\n' ' '; echo; }

# best-of over N runs: keep the max MB/s line
best_gt() {
  local model="$1" file="$2" n="${3:-5}" bestmb=0 bestln=""
  for _ in $(seq "$n"); do
    ln=$(uvx --with tokenizers --from gigatoken gigatoken bench "$model" "$file" 2>&1 | grep gigatoken || true)
    mb=$(echo "$ln" | grep -oE '[0-9.]+ MB/s' | grep -oE '[0-9.]+' || echo 0)
    if awk "BEGIN{exit !($mb>$bestmb)}"; then bestmb=$mb; bestln=$ln; fi
  done
  echo "$bestln"
}

echo "########## gigatoken best-of-5, cl100k (Xenova/gpt-4) ##########"
for f in small_45 small_900 small_44k shake_1mb shake_8mb; do
  printf '%-12s ' "$f"; best_gt 'Xenova/gpt-4' "$D/$f.txt" 5
done
printf '%-12s ' "big.txt(32M)"; best_gt 'Xenova/gpt-4' data/big.txt 5
printf '%-12s ' "shake_1gb"; best_gt 'Xenova/gpt-4' "$D/shake_1gb.txt" 5

echo "########## gigatoken best-of-5, o200k (Xenova/gpt-4o) ##########"
printf '%-12s ' "shake_1gb"; best_gt 'Xenova/gpt-4o' "$D/shake_1gb.txt" 5

echo "########## blaze parallel sanity (should approach ~1.14B tok/s at full clocks) ##########"
cargo run -q --release --example speedgraph "$D/shake_1gb.txt" 2>/dev/null
