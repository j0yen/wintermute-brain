#!/usr/bin/env bash
# detect-backend.sh — choose the best llama.cpp backend for the Ryzen 7 5700U APU.
#
# Strategy (PRD-constellation-brain-local §2):
#   1. Probe whether Vulkan RADV is available for the Vega 8 iGPU (gfx90c).
#   2. If Vulkan is available, run a short generation benchmark on BOTH Vulkan
#      and CPU-only. Pick whichever achieves higher tokens/s on generation
#      (NOT prefill — the APU's memory bandwidth ceiling means Vulkan rarely
#      beats CPU on generation, ~8-10 tok/s either way).
#   3. If Vulkan is absent or benchmark fails, fall back to CPU silently.
#   4. Explicitly NEVER use ROCm (measured slower on gfx90c, 6.84 tg/s vs
#      ~8-10 tok/s plain CPU; HSA_OVERRIDE_GFX_VERSION=9.0.0 is unreliable).
#   5. Print the chosen backend name (one of: "vulkan", "cpu") and the
#      measured tok/s to stdout. Exit 0 on success, 1 on failure.
#
# Usage:
#   ./detect-backend.sh [--model <path>]
#   Prints: backend=<vulkan|cpu> toks_per_s=<float>
#   On a non-Vulkan host (laptop, CI): exits 0 with backend=cpu toks_per_s=0
#
# Deps: llama-bench or llama-cli in PATH, vulkaninfo (optional).

set -uo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" && -n "${WM_LOCAL_LLM_MODEL_PATH:-}" ]]; then
    MODEL="$WM_LOCAL_LLM_MODEL_PATH"
fi

log() { echo "[detect-backend] $*" >&2; }

# ── 1. Probe Vulkan ──────────────────────────────────────────────────────────
VULKAN_AVAILABLE=0
if command -v vulkaninfo &>/dev/null; then
    # Look for RADV (AMD open-source Vulkan driver for gfx90c / Vega 8).
    if vulkaninfo --summary 2>/dev/null | grep -qi "radv\|amd"; then
        VULKAN_AVAILABLE=1
        log "Vulkan RADV detected (Vega 8 / gfx90c)."
    else
        log "vulkaninfo present but no RADV/AMD GPU found; falling back to CPU."
    fi
else
    log "vulkaninfo not found; falling back to CPU (install vulkan-radeon for Vulkan path)."
fi

# ── 2. Benchmark helper ──────────────────────────────────────────────────────
# Runs llama-bench with the given backend flag for a short generation run.
# Prints the tokens/s generation throughput (tg) as a float, or 0 on error.
bench_toks_per_s() {
    local extra_flags=("$@")
    if [[ -z "$MODEL" ]]; then
        echo "0"
        return
    fi
    if ! command -v llama-bench &>/dev/null; then
        echo "0"
        return
    fi
    # --n-gen 32: short enough to be fast, long enough to be a fair tg sample.
    local out
    out=$(llama-bench -m "$MODEL" --n-gen 32 --n-prompt 0 "${extra_flags[@]}" 2>/dev/null \
        | grep -oP '(?<=tg/s: )[\d.]+' | head -1)
    echo "${out:-0}"
}

# ── 3. Decision ─────────────────────────────────────────────────────────────
if [[ "$VULKAN_AVAILABLE" -eq 1 ]]; then
    log "Benchmarking Vulkan vs CPU generation throughput..."
    VK_TPS=$(bench_toks_per_s --n-gpu-layers 9999 --override-kv tokenizer.ggml.padding_eos_id=int:2)
    CPU_TPS=$(bench_toks_per_s)
    log "Vulkan tg/s: $VK_TPS  CPU tg/s: $CPU_TPS"
    # Pick Vulkan only if it is measurably faster (> 10% gain) — on the 5700U
    # APU they are usually within noise (~8-10 tok/s each), so CPU is the safe
    # default unless Vulkan clearly wins.
    if awk "BEGIN { exit ($VK_TPS > $CPU_TPS * 1.1) ? 0 : 1 }"; then
        log "Vulkan wins on generation; using Vulkan backend."
        echo "backend=vulkan toks_per_s=$VK_TPS"
    else
        log "Vulkan offers no meaningful gain over CPU; using CPU backend."
        echo "backend=cpu toks_per_s=$CPU_TPS"
    fi
else
    log "Using CPU backend."
    echo "backend=cpu toks_per_s=0"
fi
