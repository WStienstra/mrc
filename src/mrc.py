#!/usr/bin/env python3
"""
WFC v2 — Wietse Fractal Codec (address + correction)

Wietse's architecture, made lossless:
  1. Both encoder and decoder deterministically regenerate the SAME Mandelbrot
     atlas from ~tens of bytes of parameters. The codebook is never transmitted.
  2. A file is stored as ADDRESSES on the fractal: for each block, the position
     of the best-matching fractal segment + a 2-parameter affine fit (s, o).
  3. The probability wall of exact matching (P ~ 10^-149) is removed by storing
     a small CORRECTION (residual) per block. Every block always has a best
     match, so encoding always succeeds; corrections carry what the fractal
     didn't predict. Lossless by construction.

Format (after header): lzma streams of [indices][scales][offsets][residuals].
"""
import sys, struct, lzma
import numpy as np

MAGIC = b"WFC2"

# ---------------------------------------------------------------- atlas ----

# Deterministic viewports, multiple zoom levels. ("m", cx, cy, span) =
# Mandelbrot escape times; ("j", cre, cim, cx, cy, span) = Julia set for c.
# The multi-scale set is what makes this *fractal*: the same generator yields
# structure at every scale, so the atlas contains slow ramps, plateaus,
# oscillations and rough textures — a universal pattern bank.
# Default = original hand-picked set: beat the greedy-tuned set on real
# vibration data (2026-06-10 ablation + fractal-zoo probe; the tuning had
# overfit its synthetic TRAIN slices).
VIEWPORTS = [
    ("m", -0.75, 0.0, 3.0),
    ("m", -0.7435, 0.1314, 0.02),
    ("m", -0.7453, 0.1127, 0.0065),
    ("m", -0.16, 1.0405, 0.04),
    ("m", -1.25066, 0.02012, 0.0017),
    ("m", -0.745428, 0.113009, 0.00003),
    ("m", 0.27322, 0.00595, 0.0005),
    ("m", -1.74995768, 0.0, 0.0000003),
]
VIEWPORTS_V1 = VIEWPORTS   # alias kept so existing A/B scripts keep working

VIEWPORTS_V2_TUNED = [  # greedy-selected 2026-06-09 (tune_atlas.py), kept for A/B
    ("m", -1.7864402555, 0.0, 0.00001),
    ("m", -0.745428, 0.113009, 0.00003),
    ("j", 0.285, 0.01, 0.0, 0.0, 3.0),
    ("m", 0.27322, 0.00595, 0.0005),
    ("m", -0.235125, 0.827215, 0.0004),
    ("m", -0.16, 1.0405, 0.04),
    ("m", 0.001643721971153, 0.822467633298876, 0.0001),
    ("m", -1.74995768, 0.0, 0.0000003),
]

def set_viewports(vps):
    global VIEWPORTS, _ATLAS_CACHE, _CURRENT_ATLAS
    VIEWPORTS = vps
    _ATLAS_BANKS[0].clear()        # viewports only ever shape atlas ID 0
    _ATLAS_CACHE = _ATLAS_BANKS[0]
    _CURRENT_ATLAS = 0

def render_atlas(side=362, max_iter=255):
    """Render each viewport as side x side escape-time grid, concat rows -> 1D."""
    chunks = []
    for spec in VIEWPORTS:
        if spec[0] == "m":
            _, cx, cy, span = spec
            xs = np.linspace(cx - span / 2, cx + span / 2, side)
            ys = np.linspace(cy - span / 2, cy + span / 2, side)
            X, Y = np.meshgrid(xs, ys)
            C = X + 1j * Y
            Z = np.zeros_like(C)
        else:
            _, cre, cim, cx, cy, span = spec
            xs = np.linspace(cx - span / 2, cx + span / 2, side)
            ys = np.linspace(cy - span / 2, cy + span / 2, side)
            X, Y = np.meshgrid(xs, ys)
            Z = X + 1j * Y
            C = np.full_like(Z, complex(cre, cim))
        it = np.zeros(Z.shape, dtype=np.int32)
        alive = np.ones(Z.shape, dtype=bool)
        for i in range(max_iter):
            Z[alive] = Z[alive] * Z[alive] + C[alive]
            esc = np.abs(Z) > 2.0
            newly = esc & alive
            it[newly] = i
            alive &= ~esc
        it[alive] = max_iter
        chunks.append((it & 0xFF).astype(np.uint8).ravel())
    return np.concatenate(chunks)

_ATLAS_CACHE = {}

def _windows(arr, count, block):
    n = min(count, (len(arr) - block))
    stride = max(1, (len(arr) - block) // n)
    n = min(n, (len(arr) - block) // stride)
    idx = (np.arange(n)[:, None] * stride) + np.arange(block)[None, :]
    return arr[idx]

def get_candidates(block, stride):
    """Multi-scale sliding windows over the atlas: candidate matrix (n x block).

    Three dilations of the same atlas (1x, 2x, 4x) — fractal self-similarity
    means texture recurs across scales, so dilated reads add genuinely new
    shapes for free. Total capped at 65535 (uint16 addresses); `stride` selects
    the legacy single-scale bank (ver 1) vs multi-scale (ver 2, stride==0).
    """
    key = (block, stride)
    if key in _ATLAS_CACHE:
        return _ATLAS_CACHE[key]
    atlas = _ATLAS_CACHE.get("atlas")
    if atlas is None:
        atlas = render_atlas()
        _ATLAS_CACHE["atlas"] = atlas
    if stride == 0:  # multi-scale bank
        C = np.concatenate([
            _windows(atlas, 40000, block),
            _windows(atlas[::2], 16000, block),
            _windows(atlas[::4], 9535, block),
        ])[:65535].astype(np.float32)
    else:
        n = min((len(atlas) - block) // stride, 65535)
        idx = (np.arange(n)[:, None] * stride) + np.arange(block)[None, :]
        C = atlas[idx].astype(np.float32)
    Cm = C.mean(axis=1, keepdims=True)
    Cc = C - Cm
    Cn = np.sqrt((Cc * Cc).sum(axis=1))
    Cn[Cn == 0] = 1e-9
    out = (C, Cm.ravel(), Cc, Cn)
    _ATLAS_CACHE[key] = out
    return out

# -------------------------------------------------------------- 2D mode ----
# Images: tile into t x t blocks, candidate bank = true 2D patches from the
# viewport grids (the 1D path wastes the atlas's vertical structure). Same
# affine + correction machinery — only the bank and a byte reorder differ.

def _render_grids(side=362, max_iter=255):
    if "grids" not in _ATLAS_CACHE:
        flat = _ATLAS_CACHE.get("atlas")
        if flat is None:
            flat = render_atlas(side, max_iter)
            _ATLAS_CACHE["atlas"] = flat
        _ATLAS_CACHE["grids"] = flat.reshape(len(VIEWPORTS), side, side)
    return _ATLAS_CACHE["grids"]

def _patches(grid, t, stride):
    h, w = grid.shape
    ys = np.arange(0, h - t + 1, stride)
    xs = np.arange(0, w - t + 1, stride)
    out = np.empty((len(ys) * len(xs), t * t), dtype=np.uint8)
    k = 0
    for y in ys:
        for x in xs:
            out[k] = grid[y:y + t, x:x + t].ravel()
            k += 1
    return out

def get_candidates_2d(t):
    key = ("2d", t)
    if key in _ATLAS_CACHE:
        return _ATLAS_CACHE[key]
    grids = _render_grids()
    parts = []
    for g in grids:
        parts.append(_patches(g, t, 6))
        parts.append(_patches(g[::2, ::2], t, 4))
    C = np.concatenate(parts)[:65535].astype(np.float32)
    Cm = C.mean(axis=1, keepdims=True)
    Cc = C - Cm
    Cn = np.sqrt((Cc * Cc).sum(axis=1))
    Cn[Cn == 0] = 1e-9
    out = (C, Cm.ravel(), Cc, Cn)
    _ATLAS_CACHE[key] = out
    return out

def _tile_order(n, w, t):
    """Permutation mapping raster bytes -> tile-major bytes (and its length)."""
    h = n // w
    ht, wt = (h // t) * t, (w // t) * t
    idx = []
    for ty in range(0, ht, t):
        for tx in range(0, wt, t):
            for dy in range(t):
                idx.extend(range((ty + dy) * w + tx, (ty + dy) * w + tx + t))
    # leftover bytes (right/bottom margins) appended raster-order
    used = np.zeros(n, dtype=bool)
    a = np.array(idx)
    used[a] = True
    rest = np.nonzero(~used)[0]
    return np.concatenate([a, rest])

def encode_2d(data: bytes, width: int, t: int = 8) -> bytes:
    n = len(data)
    perm = _tile_order(n, width, t)
    reordered = bytes(np.frombuffer(data, dtype=np.uint8)[perm].tobytes())
    _ATLAS_CACHE[(t * t, 0)] = get_candidates_2d(t)   # bank swap for block=t*t
    body = encode(reordered, block=t * t)
    return b"WF2D" + struct.pack("<IH", width, t) + body

def decode_2d(blob: bytes) -> bytes:
    assert blob[:4] == b"WF2D"
    width, t = struct.unpack("<IH", blob[4:10])
    _ATLAS_CACHE[(t * t, 0)] = get_candidates_2d(t)
    reordered = np.frombuffer(decode(blob[10:]), dtype=np.uint8)
    n = len(reordered)
    perm = _tile_order(n, width, t)
    out = np.empty(n, dtype=np.uint8)
    out[perm] = reordered
    return out.tobytes()

# ------------------------------------------------------- entropy coding ----

_RAW_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]

def _lzma_raw(b):
    return lzma.compress(b, format=lzma.FORMAT_RAW, filters=_RAW_FILTERS)

def _unlzma_raw(b):
    return lzma.decompress(b, format=lzma.FORMAT_RAW, filters=_RAW_FILTERS)

_TOP, _BOT = 1 << 24, 1 << 16
_MASK = 0xFFFFFFFF

_GEO_PRIOR = [max(1, int(256 * 2 ** (-i / 3))) for i in range(256)]

class _AdaptiveModel:
    """Adaptive byte model with a geometric prior matched to zigzag residuals
    (small magnitudes dominate). Tuned 2026-06-09: half-life 3, increment 64
    beat flat/1+32 by 4% on real correction streams."""
    def __init__(self):
        self.freq = list(_GEO_PRIOR)
        self.total = sum(self.freq)
    def update(self, sym):
        self.freq[sym] += 64
        self.total += 64
        if self.total > _BOT - 256:
            self.total = 0
            for i in range(256):
                self.freq[i] = (self.freq[i] + 1) >> 1
                self.total += self.freq[i]

def _ac_encode(data):
    low, rng, out = 0, _MASK, bytearray()
    m = _AdaptiveModel()
    for sym in data:
        cum = sum(m.freq[:sym])
        f, tot = m.freq[sym], m.total
        rng //= tot
        low = (low + cum * rng) & _MASK
        rng *= f
        while True:
            if (low ^ (low + rng)) & _MASK < _TOP:
                pass
            elif rng < _BOT:
                rng = (-low) & (_BOT - 1)
            else:
                break
            out.append((low >> 24) & 0xFF)
            low = (low << 8) & _MASK
            rng = (rng << 8) & _MASK
        m.update(sym)
    for _ in range(4):
        out.append((low >> 24) & 0xFF)
        low = (low << 8) & _MASK
    return bytes(out)

def _ac_decode(blob, count):
    low, rng = 0, _MASK
    code = int.from_bytes(blob[:4], "big")
    pos = 4
    m = _AdaptiveModel()
    out = bytearray()
    for _ in range(count):
        rng //= m.total
        v = ((code - low) & _MASK) // rng
        cum, sym = 0, 0
        while cum + m.freq[sym] <= v:
            cum += m.freq[sym]
            sym += 1
        f = m.freq[sym]
        low = (low + cum * rng) & _MASK
        rng *= f
        while True:
            if (low ^ (low + rng)) & _MASK < _TOP:
                pass
            elif rng < _BOT:
                rng = (-low) & (_BOT - 1)
            else:
                break
            code = ((code << 8) & _MASK) | (blob[pos] if pos < len(blob) else 0)
            pos += 1
            low = (low << 8) & _MASK
            rng = (rng << 8) & _MASK
        m.update(sym)
        out.append(sym)
    return bytes(out)

# ---------------------------------------------------------------- codec ----

def _predict(data: bytes, block, stride):
    """Shared prediction core: returns per-block (mode, idxs, s, o_tx, zigzag)."""
    n = len(data)
    pad = (-n) % block
    buf = np.frombuffer(data + b"\x00" * pad, dtype=np.uint8)
    X = buf.reshape(-1, block).astype(np.float32)
    m = X.shape[0]

    C, Cmean, Cc, Cn = get_candidates(block, stride)

    Xm = X.mean(axis=1, keepdims=True)
    Xc = X - Xm
    Xn = np.sqrt((Xc * Xc).sum(axis=1))
    Xn[Xn == 0] = 1e-9

    idxs = np.empty(m, dtype=np.uint16)
    # chunk the correlation matrix to bound memory
    step = max(1, (1 << 25) // C.shape[0])
    for a in range(0, m, step):
        b = min(m, a + step)
        corr = (Xc[a:b] @ Cc.T) / (Xn[a:b, None] * Cn[None, :])
        idxs[a:b] = np.abs(corr).argmax(axis=1).astype(np.uint16)

    sel = C[idxs]                      # m x block
    sel_c = Cc[idxs]
    var = (sel_c * sel_c).sum(axis=1)
    var[var == 0] = 1e-9
    s = (Xc * sel_c).sum(axis=1) / var          # least-squares scale
    s_q = np.clip(np.round(s * 16), -127, 127).astype(np.int8)
    s_d = s_q.astype(np.float32) / 16.0
    o = Xm.ravel() - s_d * Cmean[idxs]
    o_q = np.clip(np.round(o), -32768, 32767).astype(np.int16)
    pred_f = np.round(s_d[:, None] * sel + o_q[:, None].astype(np.float32))

    # alternative per-block predictor: linear ramp fit (handles smooth blocks
    # better than any texture patch; the mode flag costs ~0 after lzma)
    t = np.arange(block, dtype=np.float32)
    tc = t - t.mean()
    a = (Xc * tc[None, :]).sum(axis=1) / (tc * tc).sum()
    a_q = np.clip(np.round(a * 16), -127, 127).astype(np.int8)
    a_d = a_q.astype(np.float32) / 16.0
    b_ = Xm.ravel() - a_d * t.mean()
    b_q = np.clip(np.round(b_), -32768, 32767).astype(np.int16)
    pred_l = np.round(a_d[:, None] * t[None, :] + b_q[:, None].astype(np.float32))

    cost_f = np.abs(X - pred_f).sum(axis=1)
    cost_l = np.abs(X - pred_l).sum(axis=1)
    use_l = cost_l < cost_f
    mode = use_l.astype(np.uint8)

    pred = np.where(use_l[:, None], pred_l, pred_f)
    s_out = np.where(use_l, a_q, s_q).astype(np.int8)
    o_out = np.where(use_l, b_q, o_q).astype(np.int16)
    idxs = np.where(use_l, 0, idxs).astype(np.uint16)

    resid = (X.astype(np.int32) - pred.astype(np.int32))
    r_s = ((resid + 128) & 0xFF) - 128                 # wrap to [-128,127]
    zz = np.where(r_s >= 0, 2 * r_s, -2 * r_s - 1)     # zigzag -> small uints

    # block offsets form a smooth signal themselves -> delta them
    o_tx = np.diff(o_out.astype(np.int32), prepend=0).astype(np.int16)
    return mode, idxs, s_out, o_tx, zz

def encode(data: bytes, block=32, stride=0) -> bytes:
    n = len(data)
    mode, idxs, s_out, o_tx, zz = _predict(data, block, stride)

    side = mode.tobytes() + idxs.tobytes() + s_out.tobytes() + o_tx.tobytes()
    zzb = zz.astype(np.uint8).tobytes()

    # container candidates — pick smallest, self-described by fmt byte
    cands = [(1, _lzma_raw(side + zzb))]
    if len(zzb) <= 16384:  # adaptive range coder pays off on small payloads
        sc = _lzma_raw(side)
        cands.append((2, struct.pack("<I", len(sc)) + sc + _ac_encode(zzb)))
    # fallbacks: raw delta+lzma (smooth data) and stored (incompressible) —
    # the codec never does worse than its best competitor or the input itself
    a8 = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    dlt = np.diff(a8, prepend=np.int16(0)).astype(np.uint8).tobytes() if n else b""
    cands.append((3, _lzma_raw(dlt)))
    cands.append((4, data))
    fmt, payload = min(cands, key=lambda t: len(t[1]))
    header = MAGIC + struct.pack("<IHBB", n, block, stride // 16 if stride else 0, fmt)
    return header + payload

def decode(blob: bytes) -> bytes:
    assert blob[:4] == MAGIC
    n, block, stride16, fmt = struct.unpack("<IHBB", blob[4:12])
    stride = stride16 * 16
    m = ((n + block - 1) // block)
    body = blob[12:]
    if fmt == 3:   # raw delta+lzma fallback
        dlt = np.frombuffer(_unlzma_raw(body), dtype=np.uint8)
        return np.cumsum(dlt.astype(np.int64)).astype(np.uint8).tobytes()[:n]
    if fmt == 4:   # stored
        return body[:n]
    if fmt == 1:
        raw = _unlzma_raw(body)
    elif fmt == 2:
        (slen,) = struct.unpack("<I", body[:4])
        raw = _unlzma_raw(body[4:4 + slen]) + _ac_decode(body[4 + slen:], m * block)
    else:  # legacy container
        raw = lzma.decompress(body)
    p = 0
    mode = np.frombuffer(raw[p:p + m], dtype=np.uint8); p += m
    idxs = np.frombuffer(raw[p:p + 2 * m], dtype=np.uint16); p += 2 * m
    s_q = np.frombuffer(raw[p:p + m], dtype=np.int8); p += m
    o_tx = np.frombuffer(raw[p:p + 2 * m], dtype=np.int16); p += 2 * m
    o_q = np.cumsum(o_tx.astype(np.int64)).astype(np.int16)
    zz = np.frombuffer(raw[p:p + m * block], dtype=np.uint8).reshape(m, block).astype(np.int32)

    C, _, _, _ = get_candidates(block, stride)
    sel = C[idxs]
    s_d = s_q.astype(np.float32) / 16.0
    pred_f = np.round(s_d[:, None] * sel + o_q[:, None].astype(np.float32))
    t = np.arange(block, dtype=np.float32)
    pred_l = np.round(s_d[:, None] * t[None, :] + o_q[:, None].astype(np.float32))
    pred = np.where((mode == 1)[:, None], pred_l, pred_f)

    r_s = np.where(zz % 2 == 0, zz // 2, -(zz + 1) // 2)   # un-zigzag
    out = ((pred.astype(np.int32) + r_s) & 0xFF).astype(np.uint8)
    return out.ravel().tobytes()[:n]

# ------------------------------------------------------------ frame mode ---
# For transport-framed payloads (radio/BLE/UDP frames): the transport already
# carries the length, so the header is 1 byte (4-bit fmt | 4-bit log2 block).
# Frame fmt 1 = WFC: packed mode bits + raw addresses + raw scales, then one
# adaptive-AC pass over zigzag(o_tx) escapes + corrections. Frame fmt 4 = stored.
# v2.4: fmt codes 1/2/3/5 = same WFC layout under atlas 0/1/2/3 — the atlas
# ID rides in spare fmt-code space, so multi-atlas frames cost zero bytes.

_FRAME_FMT_OF_ATLAS = {0: 1, 1: 2, 2: 3, 3: 5, 4: 8, 5: 9, 6: 10, 7: 11}
_ATLAS_OF_FRAME_FMT = {v: k for k, v in _FRAME_FMT_OF_ATLAS.items()}


def encode_frame(data: bytes, block=16) -> bytes:
    n = len(data)
    mode, idxs, s_out, o_tx, zz = _predict(data, block, 0)
    m = len(mode)
    mode_bits = np.packbits(mode).tobytes()
    ozz = bytearray()
    for v in o_tx.astype(int):
        v = int(v)
        z = (v << 1) ^ (v >> 31) if v >= 0 else ((-v) << 1) - 1  # zigzag
        if z < 255:
            ozz.append(z)
        else:
            ozz += bytes((255, (z >> 8) & 0xFF, z & 0xFF))
    body = (mode_bits + idxs.tobytes() + s_out.tobytes()
            + struct.pack("<H", len(ozz))
            + _ac_encode(bytes(ozz) + zz.astype(np.uint8).tobytes()))
    blk_log = block.bit_length() - 1
    if len(body) + 1 >= n + 1:
        return bytes([(blk_log << 4) | 4]) + data    # stored
    return bytes([(blk_log << 4) | _FRAME_FMT_OF_ATLAS[_CURRENT_ATLAS]]) + body

def encode_frame_multi(data: bytes, block=16, roster=(0, 1, 2, 3)) -> bytes:
    """v2.4 frame encode: try every roster atlas, keep the smallest frame."""
    best = None
    for aid in roster:
        use_atlas(aid)
        e = encode_frame(data, block)
        if best is None or len(e) < len(best):
            best = e
    use_atlas(0)
    return best

def decode_frame(blob: bytes, n: int) -> bytes:
    blk_log, fmt = blob[0] >> 4, blob[0] & 0xF
    if fmt == 4:
        return blob[1:1 + n]
    use_atlas(_ATLAS_OF_FRAME_FMT[fmt])
    block = 1 << blk_log
    m = (n + block - 1) // block
    p = 1
    mode = np.unpackbits(np.frombuffer(blob[p:p + (m + 7) // 8], dtype=np.uint8))[:m]
    p += (m + 7) // 8
    idxs = np.frombuffer(blob[p:p + 2 * m], dtype=np.uint16); p += 2 * m
    s_q = np.frombuffer(blob[p:p + m], dtype=np.int8); p += m
    (olen,) = struct.unpack("<H", blob[p:p + 2]); p += 2
    dec = _ac_decode(blob[p:], olen + m * block)
    ozz, zzb = dec[:olen], dec[olen:]
    o_tx, q = [], 0
    while len(o_tx) < m:
        z = ozz[q]; q += 1
        if z == 255:
            z = (ozz[q] << 8) | ozz[q + 1]; q += 2
        o_tx.append((z >> 1) ^ -(z & 1))
    o_q = np.cumsum(np.array(o_tx, dtype=np.int64)).astype(np.int16)
    zz = np.frombuffer(zzb, dtype=np.uint8).reshape(m, block).astype(np.int32)

    C, _, _, _ = get_candidates(block, 0)
    sel = C[idxs]
    s_d = s_q.astype(np.float32) / 16.0
    pred_f = np.round(s_d[:, None] * sel + o_q[:, None].astype(np.float32))
    t = np.arange(block, dtype=np.float32)
    pred_l = np.round(s_d[:, None] * t[None, :] + o_q[:, None].astype(np.float32))
    pred = np.where((mode == 1)[:, None], pred_l, pred_f)
    r_s = np.where(zz % 2 == 0, zz // 2, -(zz + 1) // 2)
    out = ((pred.astype(np.int32) + r_s) & 0xFF).astype(np.uint8)
    return out.ravel().tobytes()[:n]

# ----------------------------------------------- bit-tree coder (fmt 5) ----
# LZMA-style adaptive binary coding: per-node 12-bit probabilities, shift-5
# update, byte coded as 8 decisions down a 255-node tree. Context = top 3
# bits of previous byte. Adapts faster than Laplace counts on small payloads.

_KBITS, _KTOP = 12, 1 << 12

def _bt_encode(data, ctxbits=3):
    nctx = 1 << ctxbits
    probs = [[2048] * 256 for _ in range(nctx)]
    low, rng, out = 0, _MASK, bytearray()
    prev = 0
    for sym in data:
        tree = probs[prev >> (8 - ctxbits)]
        node = 1
        for i in range(7, -1, -1):
            bit = (sym >> i) & 1
            p = tree[node]
            bound = (rng >> _KBITS) * p
            if bit == 0:
                rng = bound
                tree[node] = p + ((_KTOP - p) >> 5)
            else:
                low = (low + bound) & _MASK
                rng -= bound
                tree[node] = p - (p >> 5)
            while True:
                if (low ^ (low + rng)) & _MASK < _TOP:
                    pass
                elif rng < _BOT:
                    rng = (-low) & (_BOT - 1)
                else:
                    break
                out.append((low >> 24) & 0xFF)
                low = (low << 8) & _MASK
                rng = (rng << 8) & _MASK
            node = (node << 1) | bit
        prev = sym
    for _ in range(4):
        out.append((low >> 24) & 0xFF)
        low = (low << 8) & _MASK
    return bytes(out)

def _bt_decode(blob, count, ctxbits=3):
    nctx = 1 << ctxbits
    probs = [[2048] * 256 for _ in range(nctx)]
    low, rng = 0, _MASK
    code = int.from_bytes(blob[:4], "big")
    pos = 4
    out = bytearray()
    prev = 0
    for _ in range(count):
        tree = probs[prev >> (8 - ctxbits)]
        node = 1
        for _i in range(8):
            p = tree[node]
            bound = (rng >> _KBITS) * p
            if ((code - low) & _MASK) < bound:
                bit = 0
                rng = bound
                tree[node] = p + ((_KTOP - p) >> 5)
            else:
                bit = 1
                low = (low + bound) & _MASK
                rng -= bound
                tree[node] = p - (p >> 5)
            while True:
                if (low ^ (low + rng)) & _MASK < _TOP:
                    pass
                elif rng < _BOT:
                    rng = (-low) & (_BOT - 1)
                else:
                    break
                code = ((code << 8) & _MASK) | (blob[pos] if pos < len(blob) else 0)
                pos += 1
                low = (low << 8) & _MASK
                rng = (rng << 8) & _MASK
            node = (node << 1) | bit
        sym = node & 0xFF
        out.append(sym)
        prev = sym
    return bytes(out)

# ----------------------------------------------------------------- cli -----

# ------------------------------------------- v2.4 multi-atlas registry ----
# Four specialist codebooks, each regenerated deterministically from its
# generator (zero dictionary bytes shipped, same as ever). Selection per
# encode: container carries a 1-byte atlas ID (MRC4), frames carry it in
# spare fmt-code space (free). Roster chosen by the 2026-06-10 fractal-zoo
# probes: Newton wins seismic+overall, flower-of-life wins audio, Sierpinski
# wins frame mode, Mandelbrot stays as ID 0 (legacy default + frame all-rounder).

def _newton_atlas():
    side = 362
    roots = np.array([1, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)])
    chunks = []
    for cx, cy, span in [(0, 0, 4.0), (0, 0, 1.0), (0.5, 0.5, 0.5), (0, 0, 0.25),
                         (-0.5, 0.3, 0.3), (0.2, 0, 0.1), (0, 0.6, 0.2), (0.33, 0.33, 0.05)]:
        xs = np.linspace(cx - span / 2, cx + span / 2, side)
        ys = np.linspace(cy - span / 2, cy + span / 2, side)
        X, Y = np.meshgrid(xs, ys)
        Z = (X + 1j * Y).astype(np.complex128)
        Z[Z == 0] = 1e-9
        it = np.zeros(Z.shape, dtype=np.int32)
        done = np.zeros(Z.shape, dtype=bool)
        rid = np.zeros(Z.shape, dtype=np.int32)
        for i in range(60):
            Z = np.where(done, Z, Z - (Z ** 3 - 1) / (3 * Z ** 2 + 1e-12))
            d = np.abs(Z[..., None] - roots)
            close = d.min(axis=-1) < 1e-6
            newly = close & ~done
            it[newly] = i
            rid[newly] = d.argmin(axis=-1)[newly]
            done |= close
        chunks.append(((it * 8 + rid * 85) & 0xFF).astype(np.uint8).ravel())
    return np.concatenate(chunks)

def _flower_atlas():
    side = 362
    cy, cx = np.mgrid[0:side, 0:side]
    rad = 1.0
    centers = [(rad * i + rad * 0.5 * j, rad * np.sqrt(3) / 2 * j)
               for i in range(-12, 13) for j in range(-12, 13)]
    cxs = np.array([c[0] for c in centers])
    cys = np.array([c[1] for c in centers])
    out = []
    for span in (8.0, 4.0, 2.0, 1.0, 0.5, 6.0, 3.0, 1.5):
        X = (cx / side - 0.5) * span
        Y = (cy / side - 0.5) * span
        d = np.abs(np.sqrt((X[..., None] - cxs) ** 2
                           + (Y[..., None] - cys) ** 2) - rad).min(axis=-1)
        v = np.exp(-d * 4.0).ravel()
        lo, hi = v.min(), v.max()
        out.append(((v - lo) / (hi - lo) * 255.999).astype(np.uint8))
    return np.concatenate(out)

def _sierpinski_atlas():
    target = 8 * 362 * 362
    rows, total, row = [], 0, np.array([1], dtype=np.uint8)
    while total < target:
        rows.append(row)
        total += len(row)
        row = np.concatenate([[1], (row[:-1].astype(np.uint16)
                                    + row[1:].astype(np.uint16)) & 0xFF, [1]]
                             ).astype(np.uint8)
    return np.concatenate(rows)[:target]

def _orbittrap_atlas():
    # zoo round 3 winner (held-out -0.9% engine bytes vs best reference);
    # byte-exact port of fractal_probe3.orbit_trap incl. its 8-chunk scaler
    target = 8 * 362 * 362
    side = 362

    def chunks_to_u8(x):
        x = np.asarray(x, dtype=np.float64)[:target]
        if len(x) < target:
            x = np.pad(x, (0, target - len(x)), mode="wrap")
        u8 = np.empty(target, dtype=np.uint8)
        c = target // 8
        for i in range(8):
            seg = x[i * c:(i + 1) * c] if i < 7 else x[7 * c:]
            lo, hi = seg.min(), seg.max()
            if hi - lo < 1e-12:
                hi = lo + 1
            u8[i * c:i * c + len(seg)] = ((seg - lo) / (hi - lo) * 255.0
                                          ).astype(np.uint8)
        return u8

    out = []
    for cx, cy, span in [(-0.75, 0, 3.0), (-0.7453, 0.1127, 0.0065),
                         (-0.16, 1.0405, 0.04), (-1.25066, 0.02012, 0.0017),
                         (-0.75, 0.1, 0.05), (0.275, 0.005, 0.01),
                         (-0.7435, 0.1314, 0.02), (-1.4002, 0.0, 0.005)]:
        xs = np.linspace(cx - span / 2, cx + span / 2, side)
        ys = np.linspace(cy - span / 2, cy + span / 2, side)
        X, Y = np.meshgrid(xs, ys)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        dist = np.full(C.shape, 1e9)
        alive = np.ones(C.shape, dtype=bool)
        for _ in range(64):
            Z[alive] = Z[alive] ** 2 + C[alive]
            d = np.abs(Z - (0.25 + 0.5j))
            np.minimum(dist, np.where(alive, d, 1e9), out=dist)
            alive &= np.abs(Z) <= 4.0
        out.append(chunks_to_u8(np.log1p(dist.ravel()))[:side * side])
    return np.concatenate(out)[:target]


# Zoo round 5 (2026-06-12): machine-DESIGNED atlases, one per use case.
# Random search over parameterized recipe families, fitness = real encoded
# bytes, winners validated on held-out data vs the champion roster.
# Audio -0.7% / sensor -0.6% / vibration -0.6% vs best champion; recipes are
# pure parameter blobs - the decoder regenerates everything below from them.
_DESIGNED_RECIPES = {
    5: {'family': 'mandel_trap', 'seed': 339430209, 'shape': 'twopt', 'tp': [0.3259, 0.79], 'ring_r': 0.727, 'views': [[-0.1587155, 1.0324796, 0.00468123], [-0.7451413, 0.1304193, 0.03749222], [-0.1520605, 1.0397111, 0.01507126], [-0.7452669, 0.1130453, 7.68e-05], [-1.2503291, 0.0198773, 0.0040867], [-0.7436672, 0.1317822, 0.00014697], [-0.7899572, 0.4717558, 7.36676409], [-1.2573046, 0.0369001, 0.00109387]]},   # audio
    6: {'family': 'julia_trap', 'seed': 46640651, 'shape': 'twopt', 'tp': [0.2382, 0.002], 'ring_r': 0.653, 'cs': [[0.24666, 0.00382], [-0.66933, -0.35406], [-0.14917, 0.67246], [0.28671, -0.42225], [-0.16504, 0.646], [-0.68718, -0.18393], [0.33878, 0.0949], [0.38332, -0.11619]]},   # sensor
    7: {'family': 'julia_trap', 'seed': 659068635, 'shape': 'ring', 'tp': [-0.6308, 0.2598], 'ring_r': 0.36, 'cs': [[-0.52601, 0.49907], [0.1476, 0.59037], [-0.02631, 0.63723], [0.29198, -0.03834], [-0.59396, 0.5078], [-0.62428, 0.37442], [0.37921, -0.1477], [0.32228, 0.07988]]},   # vibration
}

def _designed_atlas(aid):
    rec = _DESIGNED_RECIPES[aid]
    target = 8 * 362 * 362
    side = 362

    def chunks_to_u8(x):
        x = np.asarray(x, dtype=np.float64)[:target]
        if len(x) < target:
            x = np.pad(x, (0, target - len(x)), mode="wrap")
        u8 = np.empty(target, dtype=np.uint8)
        c = target // 8
        for i in range(8):
            seg = x[i * c:(i + 1) * c] if i < 7 else x[7 * c:]
            lo, hi = seg.min(), seg.max()
            if hi - lo < 1e-12:
                hi = lo + 1
            u8[i * c:i * c + len(seg)] = ((seg - lo) / (hi - lo) * 255.0
                                          ).astype(np.uint8)
        return u8

    tp = rec["tp"][0] + 1j * rec["tp"][1]
    if rec["shape"] == "twopt":
        trap = lambda Z: np.minimum(np.abs(Z - tp), np.abs(Z + tp))
    elif rec["shape"] == "ring":
        rr = rec["ring_r"]
        trap = lambda Z: np.abs(np.abs(Z - tp) - rr)
    else:
        trap = lambda Z: np.abs(Z - tp)

    out = []
    if rec["family"] == "mandel_trap":
        for cx, cy, span in rec["views"]:
            xs = np.linspace(cx - span / 2, cx + span / 2, side)
            ys = np.linspace(cy - span / 2, cy + span / 2, side)
            X, Y = np.meshgrid(xs, ys)
            C = X + 1j * Y
            Z = np.zeros_like(C)
            dist = np.full(C.shape, 1e9)
            alive = np.ones(C.shape, dtype=bool)
            for _ in range(64):
                Z[alive] = Z[alive] ** 2 + C[alive]
                np.minimum(dist, np.where(alive, trap(Z), 1e9), out=dist)
                alive &= np.abs(Z) <= 4.0
            out.append(chunks_to_u8(np.log1p(dist.ravel()))[:side * side])
    else:                                    # julia_trap
        for cr, ci in rec["cs"]:
            c = cr + 1j * ci
            xs = np.linspace(-1.6, 1.6, side)
            ys = np.linspace(-1.2, 1.2, side)
            X, Y = np.meshgrid(xs, ys)
            Z = X + 1j * Y
            dist = np.full(Z.shape, 1e9)
            alive = np.ones(Z.shape, dtype=bool)
            for _ in range(64):
                Z[alive] = Z[alive] ** 2 + c
                np.minimum(dist, np.where(alive, trap(Z), 1e9), out=dist)
                alive &= np.abs(Z) <= 4.0
            out.append(chunks_to_u8(np.log1p(dist.ravel()))[:side * side])
    return np.concatenate(out)[:target]

ATLAS_GENERATORS = {
    0: render_atlas,        # Mandelbrot (current VIEWPORTS)
    1: _newton_atlas,
    2: _flower_atlas,
    3: _sierpinski_atlas,
    4: _orbittrap_atlas,
    5: lambda: _designed_atlas(5),   # designed: audio
    6: lambda: _designed_atlas(6),   # designed: sensor
    7: lambda: _designed_atlas(7),   # designed: vibration
}
_ATLAS_BANKS = {0: _ATLAS_CACHE}    # ID 0 shares the legacy cache dict
_CURRENT_ATLAS = 0

def use_atlas(aid: int):
    """Switch the active codebook bank; renders the atlas on first use."""
    global _ATLAS_CACHE, _CURRENT_ATLAS
    if aid not in _ATLAS_BANKS:
        _ATLAS_BANKS[aid] = {}
    _ATLAS_CACHE = _ATLAS_BANKS[aid]
    _CURRENT_ATLAS = aid
    if "atlas" not in _ATLAS_CACHE:
        _ATLAS_CACHE["atlas"] = ATLAS_GENERATORS[aid]()


if __name__ == "__main__":
    op, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    data = open(src, "rb").read()
    if op == "c":
        out = encode(data)
    elif op == "d":
        out = decode(data)
    else:
        sys.exit("usage: mrc.py c|d in out")
    open(dst, "wb").write(out)
    print(f"{len(data)} -> {len(out)} bytes")
