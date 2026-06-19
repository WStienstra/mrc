#!/usr/bin/env python3
"""Generate all figures for the MRC paper from the real codec + real data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import mrc

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 170, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
ACC = "#1f4e79"; ACC2 = "#c0392b"; ACC3 = "#1e8449"; GREY = "#666666"

def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)

# ------------------------------------------------------------- fig 1: atlas
def fig_atlas():
    grids = mrc._render_grids()
    labels = []
    for spec in mrc.VIEWPORTS:
        if spec[0] == "m":
            labels.append(f"Mandelbrot\n({spec[1]:.6g}, {spec[2]:.6g})\nspan {spec[3]:g}")
        else:
            labels.append(f"Julia c={spec[1]}+{spec[2]}i\nspan {spec[5]:g}")
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.8))
    for ax, g, lab in zip(axes.ravel(), grids, labels):
        ax.imshow(g, cmap="magma", interpolation="nearest")
        ax.set_title(lab, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The eight atlas viewports — regenerated identically by encoder and decoder\n"
                 "from ~30 bytes of parameters; the codebook itself is never transmitted",
                 fontsize=10, y=1.02)
    save(fig, "fig_atlas.png")

# -------------------------------------------------- fig 2: atlas -> 1D bank
def fig_bank():
    atlas = mrc._ATLAS_CACHE.get("atlas")
    if atlas is None:
        atlas = mrc.render_atlas(); mrc._ATLAS_CACHE["atlas"] = atlas
    fig = plt.figure(figsize=(8, 5.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.3, 1, 1], hspace=0.85, wspace=0.3,
                          top=0.93, bottom=0.07)
    ax0 = fig.add_subplot(gs[0, :])
    # pick a structured 1,200-byte slice: wide value range, but a moderate
    # transition rate so it reads as structure rather than noise
    a = atlas.astype(np.float64)
    best, best_score = 120000, -1.0
    for s in range(0, len(a) - 1200, 600):
        w = a[s:s + 1200]
        trans = (np.abs(np.diff(w)) > 0).mean()
        if not (0.05 <= trans <= 0.30):
            continue
        score = w.std()
        if score > best_score:
            best_score, best = score, s
    seg = atlas[best:best + 1200]
    ax0.plot(seg, lw=0.6, color=ACC)
    ax0.set_title("A 1,200-byte slice of the 1 MB atlas signal (escape-time grid read as 1-D)", fontsize=9.5)
    ax0.set_xlabel("atlas position", fontsize=8); ax0.set_ylabel("byte value")
    # pick 8 windows as shape archetypes, scored explicitly so the panels
    # actually show the caption's "ramps, steps, spikes, oscillations, texture"
    C, _, _, _ = mrc.get_candidates(32, 0)
    Cf = C.astype(np.float64)
    n = Cf.shape[1]
    x = np.arange(n)
    rng = Cf.max(axis=1) - Cf.min(axis=1)
    span = np.maximum(rng, 1e-9)
    slope = np.polyfit(x, Cf.T, 1)[0]
    lin = np.outer(slope, x) + (Cf.mean(axis=1) - slope * x.mean())[:, None]
    lin_resid = np.sqrt(((Cf - lin) ** 2).mean(axis=1))
    d = np.diff(Cf, axis=1)
    max_jump = np.abs(d).max(axis=1)
    dev = Cf - Cf.mean(axis=1, keepdims=True)
    crossings = (np.diff(np.sign(dev + 1e-9), axis=1) != 0).sum(axis=1)
    spike_amp = np.abs(Cf - np.median(Cf, axis=1, keepdims=True)).max(axis=1)
    roughness = np.abs(d).mean(axis=1)
    big = rng > 40  # only windows with real dynamic range
    med = np.median(Cf, axis=1)
    ends_at_base = (np.abs(Cf[:, 0] - med) < 0.15 * span) & (np.abs(Cf[:, -1] - med) < 0.15 * span)
    narrow = (np.abs(dev) > 0.5 * span[:, None]).sum(axis=1)  # samples far from median
    gradual = max_jump / span < 0.25  # no single dominant cliff
    frac_inc = (d > 0).mean(axis=1); frac_dec = (d < 0).mean(axis=1)
    scores = {
        "ramp":        np.where(big & gradual & (lin_resid / span < 0.12),
                                np.abs(slope) * n / span, -1e9),
        "step":        np.where(big & (crossings <= 2),
                                max_jump / span - 2 * (roughness - max_jump / (n - 1)) / span, -1e9),
        "spike":       np.where(big & ends_at_base & (narrow <= 5) & (narrow >= 1),
                                spike_amp / span - roughness / span, -1e9),
        "oscillation": np.where(big & (crossings >= 8) & (crossings <= 16),
                                rng / 255.0 - 0 * crossings, -1e9),
        "rough texture": np.where(big & (crossings > 16), roughness, -1e9),
        "sigmoid rise": np.where(big & ((np.abs(d) > 0.15 * span[:, None]).sum(axis=1) >= 3)
                                & ((np.abs(d) > 0.15 * span[:, None]).sum(axis=1) <= 5)
                                & (max_jump / span < 0.5),
                                ((np.abs(d) < 2).mean(axis=1)) * rng / 255.0, -1e9),
        "smooth curve": np.where(big & gradual & (crossings <= 4) & (lin_resid / span > 0.12),
                                 -roughness / span, -1e9),
        "decay":       np.where(big & (frac_dec > 0.5) & (frac_inc < 0.2)
                                & (max_jump / span < 0.4),
                                rng / 255.0 - lin_resid / span, -1e9),
    }
    for label, sc in scores.items():
        assert sc.max() > -1e8, f"no candidate window qualifies for {label!r}"
    picks, used = [], set()
    for label, sc in scores.items():
        for p in np.argsort(sc)[::-1]:
            if int(p) not in used:
                used.add(int(p)); picks.append((label, int(p))); break
    for k, (label, p) in enumerate(picks):
        ax = fig.add_subplot(gs[1 + k // 4, k % 4])
        ax.plot(C[p], lw=1.1, color=plt.cm.viridis(k / 8))
        ax.set_title(f"{label} — #{p}", fontsize=7.5)
        ax.tick_params(labelsize=6)
    save(fig, "fig_bank.png")

# --------------------------------------------------- fig 3: multiscale reads
def fig_multiscale():
    atlas = mrc._ATLAS_CACHE["atlas"]
    o = 250000
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.4), sharey=True)
    for ax, (d, lab) in zip(axes, [(1, "1× (32 atlas bytes)"),
                                   (2, "2× dilation (64 atlas bytes)"),
                                   (4, "4× dilation (128 atlas bytes)")]):
        w = atlas[o:o + 32 * d:d]
        ax.plot(w, lw=1.4, color=ACC)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("window position")
    axes[0].set_ylabel("byte value")
    fig.suptitle("Multi-scale reads: the same atlas region sampled at three dilations gives three\n"
                 "different 32-byte shapes — fractal self-similarity as a free codebook multiplier", y=1.13)
    save(fig, "fig_multiscale.png")

# ------------------------------------------------ fig 4: real match example
def fig_match():
    audio = open("/home/patrick/clawd/projects/fractal-compress/test_audio_1m.bin", "rb").read()
    data = audio[40000:40000 + 256]
    block = 32
    mode, idxs, s_out, o_tx, zz = mrc._predict(data, block, 0)
    o_q = np.cumsum(o_tx.astype(np.int64)).astype(np.int16)
    C, _, _, _ = mrc.get_candidates(block, 0)
    # pick first fractal-mode block
    bi = int(np.nonzero(mode == 0)[0][0])
    X = np.frombuffer(data, dtype=np.uint8).reshape(-1, block)[bi].astype(float)
    cand = C[idxs[bi]]
    s = s_out[bi] / 16.0
    pred = np.round(s * cand + float(o_q[bi]))
    resid = X - pred
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7))
    axes[0].plot(X, "o-", ms=3, lw=1, color=ACC, label="input block")
    axes[0].plot(cand, "s--", ms=3, lw=1, color=ACC2, label=f"best atlas window (#{idxs[bi]})")
    axes[0].set_title("1. Search: best of 65,535 windows\n(normalized correlation)")
    axes[0].legend(fontsize=7)
    axes[1].plot(X, "o-", ms=3, lw=1, color=ACC, label="input block")
    axes[1].plot(pred, "s--", ms=3, lw=1, color=ACC3,
                 label=f"affine fit: {s:.2f}·window + {int(o_q[bi])}")
    axes[1].set_title("2. Fit: scale + offset (3 bytes)\nbend the window onto the data")
    axes[1].legend(fontsize=7)
    axes[2].bar(np.arange(block), resid, color=ACC2)
    axes[2].set_ylim(-30, 30)
    axes[2].set_title(f"3. Correction: residual (mean |r| = {np.abs(resid).mean():.1f})\nsmall numbers → cheap to entropy-code")
    for ax in axes:
        ax.set_xlabel("byte position in block")
    axes[0].set_ylabel("byte value")
    fig.suptitle("One real 32-byte audio block through the codec", y=1.06)
    fig.tight_layout()
    save(fig, "fig_match.png")

# ----------------------------------------------------- fig 5: pipeline schematic
def _box(ax, x, y, w, h, text, fc="#eaf1f8", ec=ACC, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def _arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=lw))

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
    # encoder row
    ax.text(0.2, 6.0, "ENCODER", fontsize=10, fontweight="bold", color=ACC)
    _box(ax, 0.2, 4.8, 1.5, 0.9, "input file\n(split into\n32-byte blocks)")
    _box(ax, 2.2, 4.8, 1.9, 0.9, "atlas search\nbest window of 65,535\n(normalized correlation)")
    _box(ax, 4.6, 4.8, 1.7, 0.9, "affine fit\nscale s, offset o\n(vs. linear-ramp mode)")
    _box(ax, 6.8, 4.8, 1.5, 0.9, "residual\nwrap + zigzag\n(small uints)")
    _box(ax, 8.7, 4.8, 1.1, 0.9, "entropy\ncoder", fc="#fdeeee", ec=ACC2)
    _arrow(ax, 1.7, 5.25, 2.2, 5.25); _arrow(ax, 4.1, 5.25, 4.6, 5.25)
    _arrow(ax, 6.3, 5.25, 6.8, 5.25); _arrow(ax, 8.3, 5.25, 8.7, 5.25)
    # shared atlas, left-center
    _box(ax, 0.2, 2.7, 4.4, 1.15,
         "DETERMINISTIC ATLAS\n8 Mandelbrot/Julia viewports → 1 MB pattern bank\nregenerated from ~30 bytes of parameters",
         fc="#fff7e6", ec="#b9770e", fs=8.5)
    _arrow(ax, 2.7, 3.85, 3.0, 4.8, color="#b9770e")
    _arrow(ax, 1.6, 2.7, 1.2, 1.6, color="#b9770e")
    ax.text(2.4, 2.42, "never transmitted — both sides rebuild it bit-identically",
            ha="center", fontsize=7.8, style="italic", color="#b9770e")
    # bitstream, right-center (green: the only bytes that travel)
    _box(ax, 5.0, 2.65, 4.9, 1.0,
         "bitstream — the only bytes that travel\nheader │ modes │ addresses │ scales │ offsets │ corrections",
         fc="#eef7ee", ec=ACC3, fs=8.0)
    _arrow(ax, 9.25, 4.8, 8.7, 3.65, color=GREY)
    _arrow(ax, 5.9, 2.65, 2.1, 1.62, color=GREY)
    # decoder row
    ax.text(0.2, 1.85, "DECODER", fontsize=10, fontweight="bold", color=ACC3)
    _box(ax, 0.2, 0.7, 2.3, 0.9, "look up addressed\nwindows in\nregenerated atlas", fc="#eef7ee", ec=ACC3, fs=8.2)
    _box(ax, 3.0, 0.7, 2.0, 0.9, "apply affine fit\n+ add corrections", fc="#eef7ee", ec=ACC3, fs=8.2)
    _box(ax, 5.5, 0.7, 1.6, 0.9, "exact original,\nbit for bit", fc="#eef7ee", ec=ACC3, fs=8.2)
    _arrow(ax, 2.5, 1.15, 3.0, 1.15); _arrow(ax, 5.0, 1.15, 5.5, 1.15)
    ax.set_title("MRC pipeline: the file is stored as addresses on the fractal plus a correction stream", fontsize=10)
    save(fig, "fig_pipeline.png")

# ----------------------------------------------------- fig 6: stream format
def fig_format():
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.2); ax.axis("off")
    segs = [("header\n10 B file / 1 B frame", 1.8, "#fff7e6", "#b9770e"),
            ("mode bits\n1 bit/block", 1.1, "#eaf1f8", ACC),
            ("addresses\n2 B/block", 1.3, "#eaf1f8", ACC),
            ("scales\n1 B/block", 1.1, "#eaf1f8", ACC),
            ("offset deltas\n~1 B/block", 1.3, "#eaf1f8", ACC),
            ("corrections (zigzag residuals)\nadaptive range coder, geometric prior", 3.0, "#fdeeee", ACC2)]
    x = 0.2
    for text, w, fc, ec in segs:
        _box(ax, x, 1.7, w, 0.95, text, fc=fc, ec=ec, fs=7.6)
        x += w + 0.07
    ax.text(0.2, 1.22, "Container selection: the encoder tries every format family and keeps the smallest —",
            fontsize=8.6)
    for i, (t, c) in enumerate([
            ("engine: raw LZMA or range-coded, plain or byte-plane (fmt 1/2/7/8)", ACC),
            ("delta transforms: byte delta (fmt 3) or int16 sample delta (fmt 9/10)", ACC2),
            ("whole-stream LZMA / plane-delta (fmt 5/6)", ACC3),
            ("stored (fmt 4)", GREY)]):
        ax.text(0.4, 0.86 - i * 0.24, "\u2022 " + t, fontsize=7.8, color=c)
    ax.text(10.2, 0.45, "one fmt byte makes every choice self-describing;\nthe codec can never lose to its own fallbacks",
            fontsize=7.8, style="italic", color=GREY, ha="right")
    ax.set_title("Bitstream layout (per file) and the container choice", fontsize=10)
    save(fig, "fig_format.png")

# ----------------------------------------------------- fig 7: benchmark bars
def fig_bench():
    # v2.3 sweep: avg savings (%) vs best of gzip-9/lzma-9e/delta+lzma per
    # source x size over 5 offsets (grind3-results.json, 2026-06-10)
    sources = ["audio", "pcm16", "terrain", "fbm", "perlin", "walk", "coastline"]
    s256 = [36.3, 32.3, 24.6, 31.8, 27.9, 18.5, -5.6]
    s512 = [27.4, 23.7, 25.4, 23.9, 18.5, 19.0, -0.4]
    s1k = [15.5, 10.9, 12.2, 12.0, 12.1, 9.8, 6.1]
    s4k = [4.2, 2.0, 2.2, 2.3, 4.1, 2.5, 7.3]
    x = np.arange(len(sources)); w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), width_ratios=[2.2, 1])
    ax = axes[0]
    for i, (vals, lab, c) in enumerate([(s256, "256 B", ACC), (s512, "512 B", "#2e86c1"),
                                        (s1k, "1 KB", "#85c1e9"), (s4k, "4 KB", "#d5dbdb")]):
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, sources)
    ax.set_ylabel("size reduction vs best classical (%)")
    ax.set_title("MRC v2.3 vs best of gzip-9 / LZMA-9e / delta+LZMA\n(140 slices; 132 wins; all 8 losses = sub-1 KB coastline)")
    ax.legend(fontsize=8)
    ax2 = axes[1]
    labels = ["raw\ndeflate", "zstd-19 +\ntrained dict", "MRC\nframe"]
    vals = [61, 67, 46]
    bars = ax2.bar(labels, vals, color=[GREY, "#9b59b6", ACC2])
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v} B", ha="center", fontsize=8.5)
    ax2.axhline(64, color="k", lw=0.8, ls="--")
    ax2.text(2.45, 64.5, "input: 64 B", fontsize=7.5, ha="right")
    ax2.set_ylabel("encoded size (bytes)")
    ax2.set_title("64-byte audio frame\n(transport-framed mode)")
    fig.tight_layout()
    save(fig, "fig_bench.png")

# ----------------------------------------------------- fig 8: regime + chunking
def fig_regime():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax = axes[0]
    sizes = [64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144]
    audio_sav = [28, 33, 36.3, 27.4, 15.5, 4.2, 2.1, 1.8, 2.5]  # frames <256, files >=256 (v2.3)
    ax.semilogx(sizes, audio_sav, "o-", color=ACC, lw=1.6, label="audio (textured)")
    ax.semilogx([256, 512, 1024, 4096], [-5.6, -0.4, 6.1, 7.3], "s--", color=GREY, lw=1.2,
                label="coastline (smooth)")
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_betweenx([-10, 40], 40, 1100, color=ACC, alpha=0.07)
    ax.text(200, 37, "fractal-engine regime", fontsize=8, color=ACC, ha="center")
    ax.fill_betweenx([-10, 40], 4096, 300000, color=ACC3, alpha=0.06)
    ax.text(48000, 37, "variable-block extension", fontsize=8, color=ACC3, ha="center")
    ax.text(48000, 33.5, "(fractal engine wins on audio,\nfallback ties elsewhere)",
            fontsize=6.8, color=ACC3, ha="center", va="top")
    ax.set_xlabel("payload size (bytes)"); ax.set_ylabel("saving vs best classical (%)")
    ax.set_title("Where the edge lives: saving vs payload size (v2.3)")
    ax.legend(fontsize=8, loc="center right"); ax.set_ylim(-10, 40)
    ax2 = axes[1]
    fsz = ["4 KB", "16 KB", "64 KB", "256 KB"]
    whole = [2271, 8704, 34349, 137169]
    chk256 = [2974, 11895, 47525, 190141]
    chk1k = [2580, 10315, 41197, 164935]
    base = [2316, 8748, 34392, 137212]
    x = np.arange(4); w = 0.2
    for i, (vals, lab, c) in enumerate([(whole, "whole-file MRC", ACC),
                                        (base, "best classical", GREY),
                                        (chk1k, "chunked 1 KB", "#e59866"),
                                        (chk256, "chunked 256 B", ACC2)]):
        ax2.bar(x + (i - 1.5) * w, np.array(vals) / np.array(whole), w, label=lab, color=c)
    ax2.set_xticks(x, fsz)
    ax2.set_ylabel("size relative to whole-file MRC")
    ax2.set_title("Chunking a large file into MRC-sized pieces\nloses 14–39% (audio corpus)")
    ax2.legend(fontsize=7.5); ax2.set_ylim(0.9, 1.45)
    fig.tight_layout()
    save(fig, "fig_regime.png")

# ------------------------------------------------- fig 9: variable blocks
def fig_varblock():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import mrc3
    data = open("/home/patrick/clawd/projects/fractal-compress/test_audio_1m.bin",
                "rb").read(2048)
    buf = np.frombuffer(data, dtype=np.uint8)
    segments, levels = mrc3._segment(buf, side_bits=96.0)
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.plot(buf, color="#1b2631", lw=0.7)
    colors = {32: ACC2, 64: "#e59866", 128: ACC, 256: ACC3, 512: "#9b59b6"}
    pos = 0
    seen = set()
    for c, _j in segments:
        size = mrc3.SIZES[c]
        lab = f"{size} B" if size not in seen else None
        seen.add(size)
        ax.axvspan(pos, pos + size, color=colors[size], alpha=0.13, label=lab)
        ax.axvline(pos, color="#7f8c8d", lw=0.5, alpha=0.6)
        pos += size
    ax.set_xlim(0, len(buf)); ax.set_xlabel("byte offset")
    ax.set_ylabel("sample value")
    ax.legend(fontsize=8, ncol=len(seen), loc="upper right", framealpha=0.9)
    ax.set_title("Variable-block segmentation of 2 KB of real audio: the bit-cost DP cuts "
                 "long blocks where one atlas shape fits, short blocks where the signal is busy",
                 fontsize=9.5)
    save(fig, "fig_varblock.png")

# --------------------------------------- fig 10: the codebook-content study
def fig_zoo():
    import json
    V2 = "/home/patrick/clawd/projects/wietse-fractal-compression/v2/"
    r1 = json.load(open(V2 + "fractal-probe-results.json"))
    r2 = json.load(open(V2 + "fractal-probe2-results.json"))
    ra = json.load(open(V2 + "ablation-results.json"))

    def tot(rec):
        return sum(s["engine"] for s in rec["slices"])

    arms = {}
    for src in (ra, r1, r2):
        for k, v in src.items():
            arms.setdefault(k, tot(v))
    arms.pop("fractal-v2", None)   # tuned set, superseded (overfit; see text)
    nice = {"fractal-v1": "Mandelbrot", "newton": "Newton $z^3{-}1$",
            "julia-zoo": "Julia (4 sets)", "sierpinski": "Sierpinski (Pascal mod 256)",
            "flower": "flower of life", "cantor": "Cantor staircase",
            "fern": "Barnsley fern", "burning-ship": "Burning Ship",
            "lorenz": "Lorenz attractor", "walk": "Brownian walk (PRNG)",
            "dragon": "dragon curve", "takagi": "Takagi curve",
            "koch": "Koch curve", "fbm": "1/f noise (PRNG)",
            "hilbert-v1": "Mandelbrot, Hilbert scan", "levy": u"Lévy C curve",
            "weier": "Weierstrass", "logistic": "logistic map",
            "white": "white noise (PRNG)"}
    order = sorted(arms, key=lambda a: arms[a])
    base = arms["fractal-v1"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = []
    for a in order:
        if a in ("white", "walk", "fbm"):
            cols.append(GREY)
        elif a == "fractal-v1":
            cols.append(ACC2)
        elif arms[a] < base:
            cols.append(ACC3)
        else:
            cols.append(ACC)
    y = np.arange(len(order))
    ax.barh(y, [arms[a] for a in order], color=cols, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([nice[a] for a in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(base, color=ACC2, lw=1.0, ls="--", alpha=0.8)
    for yi, a in zip(y, order):
        d = 100 * (arms[a] - base) / base
        ax.text(arms[a] + 60, yi, f"{d:+.1f}%", va="center", fontsize=7.5,
                color="#333333")
    ax.set_xlim(24000, 27600)
    ax.set_xlabel("total engine-coded bytes, 72 test slices (lower is better)")
    ax.set_title("19 regenerable codebooks through the identical pipeline. Grey = seeded-PRNG "
                 "baselines; red = Mandelbrot;\ngreen = fractals that beat it. Content matters "
                 "(white noise $+7.7\\%$) but no single fractal dominates every domain",
                 fontsize=9.5)
    save(fig, "fig_zoo.png")

# ------------------------------------------ fig: the full codebook registry
def fig_registry():
    side = 362
    # (id, label, chunk index to display — picked for visual structure)
    specs = [
        (0, "ID 0 — Mandelbrot (legacy)", 0),
        (1, "ID 1 — Newton $z^3{-}1$ basins", 0),
        (2, "ID 2 — flower-of-life lattice", 0),
        (3, "ID 3 — Sierpinski (Pascal mod 256)", 0),
        (4, "ID 4 — orbit-trap Mandelbrot (hunted)", 6),
        (5, "ID 5 — designed: audio\n(two-point-trapped Mandelbrot)", 1),
        (6, "ID 6 — designed: sensor\n(two-point-trapped Julia)", 1),
        (7, "ID 7 — designed: vibration\n(ring-trapped Julia)", 0),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(9, 5.0))
    for ax, (aid, lab, ci) in zip(axes.ravel(), specs):
        if aid == 0:
            g = mrc._render_grids()[0]
        else:
            bank = mrc.ATLAS_GENERATORS[aid]()
            g = np.asarray(bank[ci * side * side:(ci + 1) * side * side]
                           ).reshape(side, side)
        ax.imshow(g, cmap="magma", interpolation="nearest")
        ax.set_title(lab, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The full codebook registry (one representative grid each). IDs 0--3: the v2.4 "
                 "roster; ID 4: the hunted\norbit-trap survivor; IDs 5--7: machine-designed "
                 "per-domain recipes. Each regenerates from ${\\sim}30$ bytes",
                 fontsize=10, y=1.02)
    save(fig, "fig_registry.png")

# ----------------------------------------------- fig 11: v2.4 multi-atlas
def fig_v24():
    # held-out numbers from v2/v24-bench-results.txt + unified-frame bench
    srcs = ["audio", "pcm16", "seismo40", "seismo100"]
    slc23 = [6261, 7055, 4571, 4447]; slc24 = [6058, 6784, 4571, 4444]
    frm23 = [1816, 1925, 2328, 2322]; frm24 = [1779, 1890, 2115, 2087]
    # unified frame mode, 64-byte held-out frames: atlas family / sdz family / unified
    uni = {"audio": (627, 627, 611), "pcm16": (650, 682, 646),
           "seismo40": (731, 403, 403), "seismo100": (670, 414, 414)}
    short = ["audio", "pcm16", "seis40", "seis100"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    x = np.arange(len(srcs)); w = 0.38
    for ax, a23, a24, ttl in ((axes[0], slc23, slc24, "slices (256 B--1 KB)"),
                              (axes[1], frm23, frm24, "frames (64/128 B)")):
        ax.bar(x - w / 2, a23, w, color=GREY, label="v2.3 (one atlas)")
        ax.bar(x + w / 2, a24, w, color=ACC3, label="v2.4 (atlas roster)")
        for xi, (b, c) in enumerate(zip(a23, a24)):
            d = 100 * (c - b) / b
            ax.text(xi, max(b, c) * 1.015, f"{d:+.1f}%", ha="center", va="bottom",
                    fontsize=7.5)
        ax.set_ylim(0, max(a23) * 1.22)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
        ax.set_ylabel("bytes"); ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=7.5, loc="lower left")
    ax = axes[2]
    for i, (lab, col) in enumerate((("atlas family", ACC), ("sample-delta family", GREY),
                                    ("unified header", ACC3))):
        ax.bar(x + (i - 1) * 0.27, [uni[s][i] for s in srcs], 0.27, color=col, label=lab)
    ax.set_ylim(0, 900)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("bytes (12 frames)"); ax.set_title("unified 64 B frame header", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.suptitle("v2.4 on held-out data: per-item codebook selection (1 header byte) and the "
                 "unified frame mode that always picks the better family", fontsize=9.5, y=1.04)
    save(fig, "fig_v24.png")

if __name__ == "__main__":
    fig_atlas(); fig_bank(); fig_multiscale(); fig_match()
    fig_pipeline(); fig_format(); fig_bench(); fig_regime(); fig_varblock()
    fig_zoo(); fig_v24(); fig_registry()
    print("all figures done")
