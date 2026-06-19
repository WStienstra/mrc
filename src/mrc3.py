#!/usr/bin/env python3
"""
MRC v2.3 — variable-block segmentation ("fractal-determined snippets").

Wietse's 2026-06-10 idea, made rigorous: instead of cutting a file into
independent chunks (measured loser: +14..39%), let the encoder choose the
*block size* inside one whole-file encode. Long blocks where one atlas shape
fits well (less side-info), short blocks where the signal is busy (better
fits). Hierarchical choice per 128-byte quad: 1x128 vs 2x64 vs 4x32, by
estimated bit cost. The size stream costs ~2 bits/segment before LZMA.

Same atlas, same affine fit, same corrections, same containers as mrc.py.
"""
import struct, lzma
import numpy as np
import mrc
from mrc import (_lzma_raw, _unlzma_raw, _ac_encode, _ac_decode,
                 get_candidates)

MAGIC3 = b"MRC3"          # legacy v2.3 container (atlas 0 implied)
MAGIC4 = b"MRC4"          # v2.4 container: +1 header byte = atlas ID
SIZES = (32, 64, 128, 256, 512)  # power-of-two ladder, size code = index
# DP aggressiveness grid: estimated side-info bits per segment. Higher values
# push toward larger blocks. The level predictions are computed once; trying
# several settings and keeping the smallest encoding is nearly free, and the
# stream is self-describing so the decoder never needs to know which won.
SB_GRID = (36.0, 96.0, 160.0, 224.0)
SIDE_BITS = 36.0               # used by seg_stats/_segment default
_AC_CAP = 1 << 18              # range-coder cap (pure-python speed guard)


def _predict_blocks(X, block):
    """Core predictor on a gathered m x block matrix. Returns per-block
    (mode, idxs, s_out int8, o_out int16 ABSOLUTE, zz int32 m x block)."""
    m = X.shape[0]
    C, Cmean, Cc, Cn = get_candidates(block, 0)

    Xm = X.mean(axis=1, keepdims=True)
    Xc = X - Xm
    Xn = np.sqrt((Xc * Xc).sum(axis=1))
    Xn[Xn == 0] = 1e-9

    idxs = np.empty(m, dtype=np.uint16)
    step = max(1, (1 << 25) // C.shape[0])
    for a in range(0, m, step):
        b = min(m, a + step)
        corr = (Xc[a:b] @ Cc.T) / (Xn[a:b, None] * Cn[None, :])
        idxs[a:b] = np.abs(corr).argmax(axis=1).astype(np.uint16)

    sel = C[idxs]
    sel_c = Cc[idxs]
    var = (sel_c * sel_c).sum(axis=1)
    var[var == 0] = 1e-9
    s = (Xc * sel_c).sum(axis=1) / var
    s_q = np.clip(np.round(s * 16), -127, 127).astype(np.int8)
    s_d = s_q.astype(np.float32) / 16.0
    o = Xm.ravel() - s_d * Cmean[idxs]
    o_q = np.clip(np.round(o), -32768, 32767).astype(np.int16)
    pred_f = np.round(s_d[:, None] * sel + o_q[:, None].astype(np.float32))

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
    r_s = ((resid + 128) & 0xFF) - 128
    zz = np.where(r_s >= 0, 2 * r_s, -2 * r_s - 1)
    return mode, idxs, s_out, o_out, zz


def _est_bits(zz):
    """Estimated coded bits per block row: side info + Elias-gamma-ish
    residual cost (matches the geometric-prior AC within a few %)."""
    return SIDE_BITS + (2.0 * np.log2(zz.astype(np.float64) + 1.0) + 1.0).sum(axis=1)


def _levels(buf):
    """Predict every level of the ladder once (the expensive part)."""
    levels, rescosts = [], []
    for code, size in enumerate(SIZES):
        mfull = (len(buf) // size)
        if mfull == 0:
            levels.append(None)
            rescosts.append(None)
            continue
        X = buf[:mfull * size].reshape(-1, size).astype(np.float32)
        lv = _predict_blocks(X, size)
        levels.append(lv)
        rescosts.append(_est_bits(lv[4]) - SIDE_BITS)  # residual bits only
    return levels, rescosts


def _choose(rescosts, nq, side_bits):
    """Choose the segmentation by bottom-up DP over the power-of-two ladder:
    best[c][i] = min(cost of one size-c block at i, best[c-1][2i]+best[c-1][2i+1]).
    Returns segments = list of (code, index into that level's arrays) in order."""
    costs = [None if rc is None else rc + side_bits for rc in rescosts]
    top = max(c for c, cost in enumerate(costs) if cost is not None)
    # best[c][i]: cheapest encoding of the size-SIZES[c] span starting at block i
    best = [None] * (top + 1)
    best[0] = costs[0].copy()
    for c in range(1, top + 1):
        pair = best[c - 1][0::2][:len(best[c - 1]) // 2] + best[c - 1][1::2][:len(best[c - 1]) // 2]
        n_c = len(costs[c]) if costs[c] is not None else 0
        merged = np.minimum(costs[c][:min(n_c, len(pair))], pair[:min(n_c, len(pair))])
        best[c] = np.concatenate([merged, pair[len(merged):]])  # tail spans: split only

    def emit(c, i, segments):
        """Walk the DP back down: emit the choice for span (level c, index i)."""
        if c == 0:
            segments.append((0, i))
            return
        n_c = len(costs[c]) if costs[c] is not None else 0
        if i < n_c and costs[c][i] <= best[c - 1][2 * i] + best[c - 1][2 * i + 1]:
            segments.append((c, i))
        else:
            emit(c - 1, 2 * i, segments)
            emit(c - 1, 2 * i + 1, segments)

    segments = []
    i = 0
    while i < nq:
        # largest level whose full span fits at this (aligned) position
        for c in range(top, -1, -1):
            span = 1 << c
            if i % span == 0 and i + span <= nq and best[c] is not None and i // span < len(best[c]):
                emit(c, i // span, segments)
                i += span
                break
    return segments


def _segment(buf, side_bits=None):
    """Convenience wrapper: predict levels and choose at one side_bits value."""
    levels, rescosts = _levels(buf)
    segments = _choose(rescosts, len(buf) // SIZES[0], side_bits or SIDE_BITS)
    return segments, levels


def _assemble(segments, levels):
    """Build (side, zzb) byte streams for a chosen segmentation."""
    m = len(segments)
    size_codes = np.array([c for c, _ in segments], dtype=np.uint8)
    mode = np.empty(m, dtype=np.uint8)
    idxs = np.empty(m, dtype=np.uint16)
    s_out = np.empty(m, dtype=np.int8)
    o_abs = np.empty(m, dtype=np.int16)
    zz_parts = []
    for k, (c, j) in enumerate(segments):
        lmode, lidx, ls, lo, lzz = levels[c]
        mode[k] = lmode[j]
        idxs[k] = lidx[j]
        s_out[k] = ls[j]
        o_abs[k] = lo[j]
        zz_parts.append(lzz[j].astype(np.uint8))
    o_tx = np.diff(o_abs.astype(np.int32), prepend=0).astype(np.int16)
    zzb = np.concatenate(zz_parts).tobytes()
    side = (size_codes.tobytes() + mode.tobytes() + idxs.tobytes()
            + s_out.tobytes() + o_tx.tobytes())
    return side, zzb


def _deint(data: bytes) -> bytes:
    """Stride-2 byte-plane split (int16 streams: low plane ++ high plane)."""
    return data[0::2] + data[1::2]


def _reint(d: bytes, n: int) -> bytes:
    h = (n + 1) // 2
    out = bytearray(n)
    out[0::2] = d[:h]
    out[1::2] = d[h:n]
    return bytes(out)


def _plane_delta(data: bytes) -> bytes:
    """Per-plane lag-1 byte delta after stride-2 split (int16-aware fallback)."""
    a = np.frombuffer(data, dtype=np.uint8)
    return b"".join(np.diff(p.astype(np.int16), prepend=np.int16(0))
                    .astype(np.uint8).tobytes() for p in (a[0::2], a[1::2]))


def _unplane_delta(d: bytes, n: int) -> bytes:
    h = (n + 1) // 2
    a = np.frombuffer(d, dtype=np.uint8)
    lo = np.cumsum(a[:h].astype(np.int64)).astype(np.uint8)
    hi = np.cumsum(a[h:n].astype(np.int64)).astype(np.uint8)
    out = np.empty(n, dtype=np.uint8)
    out[0::2] = lo
    out[1::2] = hi
    return out.tobytes()


def _engine_cands(stream: bytes, base_fmt: int):
    """Fractal-engine container candidates (lzma / lzma+AC) for one byte
    orientation of the input. base_fmt is the fmt code of the lzma variant;
    base_fmt+1 is the AC variant."""
    pad = (-len(stream)) % 32
    buf = np.frombuffer(stream + b"\x00" * pad, dtype=np.uint8)
    levels, rescosts = _levels(buf)
    nq = len(buf) // SIZES[0]
    cands = []
    seen = set()
    for sb in SB_GRID:
        segments = _choose(rescosts, nq, sb)
        key = tuple(segments)
        if key in seen:           # same cut as a previous setting: skip
            continue
        seen.add(key)
        side, zzb = _assemble(segments, levels)
        cands.append((base_fmt, _lzma_raw(side + zzb)))
        if len(zzb) <= _AC_CAP:
            sc = _lzma_raw(side)
            cands.append((base_fmt + 1, struct.pack("<I", len(sc)) + sc + _ac_encode(zzb)))
    return cands


def _sdz(data: bytes) -> bytes:
    """int16 sample-delta -> zigzag -> byte-plane split. Even length only."""
    s = np.frombuffer(data, dtype="<i2").astype(np.int32)
    sd = np.diff(s, prepend=0)
    # wrap delta into int16 range (congruent mod 2^16) so zigzag fits uint16;
    # decoder's cumsum + int16 cast undoes the wrap exactly
    sd = ((sd + 32768) & 0xFFFF) - 32768
    zz = np.where(sd >= 0, 2 * sd, -2 * sd - 1).astype("<u2").tobytes()
    return zz[0::2] + zz[1::2]


def _unsdz(t: bytes, n: int) -> bytes:
    h = n // 2
    zz = np.empty(n, dtype=np.uint8)
    zz[0::2] = np.frombuffer(t[:h], dtype=np.uint8)
    zz[1::2] = np.frombuffer(t[h:], dtype=np.uint8)
    z = zz.view("<u2").astype(np.int64)
    sd = np.where(z % 2 == 0, z // 2, -(z + 1) // 2)
    return np.cumsum(sd).astype("<i2").tobytes()


ROSTER = (0, 1, 2, 3)   # v2.4 atlas roster: Mandelbrot/Newton/flower/Sierpinski


def encode(data: bytes, roster=ROSTER, **_ignored) -> bytes:
    n = len(data)
    # engine candidates under every roster atlas (fmt 1/2 raw, 7/8 byte-plane)
    best = None                                  # (atlas_id, fmt, payload)
    for aid in roster:
        mrc.use_atlas(aid)
        cands = _engine_cands(data, 1)
        if n >= 64:
            cands += _engine_cands(_deint(data), 7)
        for f, pl in cands:
            if best is None or len(pl) < len(best[2]):
                best = (aid, f, pl)
    mrc.use_atlas(0)
    # atlas-independent candidates, computed once
    cands = []
    if n >= 64:
        cands.append((6, _lzma_raw(_plane_delta(data))))   # int16-aware fallback
        if n % 2 == 0:
            t = _sdz(data)                       # int16 sample-delta transform
            cands.append((9, _lzma_raw(t)))
            if n <= _AC_CAP:
                cands.append((10, _ac_encode(t)))
    a8 = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    dlt = np.diff(a8, prepend=np.int16(0)).astype(np.uint8).tobytes() if n else b""
    cands.append((3, _lzma_raw(dlt)))            # byte-delta fallback (8-bit data)
    cands.append((5, _lzma_raw(data)))           # plain LZMA, no transform
    cands.append((4, data))                      # stored
    for f, pl in cands:
        if len(pl) < len(best[2]):
            best = (0, f, pl)
    aid, fmt, payload = best
    return MAGIC4 + struct.pack("<IBB", n, fmt, aid) + payload


# ----------------------------------------------- int16 frame mode (v2.3.1) --
# Transport-framed int16 sensor bursts: 1-byte header, per-frame choice of
# range-coded / lzma'd sample-delta planes, or stored. The AC's geometric
# prior is exactly matched to zigzag sample deltas of smooth telemetry.

def encode_frame16(data: bytes) -> bytes:
    assert len(data) % 2 == 0, "int16 frame must have even byte count"
    t = _sdz(data)
    cands = [(1, _ac_encode(t)), (2, _lzma_raw(t)), (4, data)]
    fmt, payload = min(cands, key=lambda c: len(c[1]))
    return bytes([fmt]) + payload


def decode_frame16(blob: bytes, n: int) -> bytes:
    fmt, body = blob[0], blob[1:]
    if fmt == 4:
        return body[:n]
    t = _ac_decode(body, n) if fmt == 1 else _unlzma_raw(body)
    return _unsdz(t, n)


# --------------------------------------------- v2.4 unified frame mode ----
# One 1-byte transport header for BOTH frame families, so the encoder can
# pick the better of atlas-engine frames and sample-delta frames per frame:
#   low nibble 1/2/3/5 = atlas-engine frame under atlas 0/1/2/3 (mrc layout,
#                        high nibble = log2 block, unchanged)
#   low nibble 6       = int16 sample-delta + range coder   (sdz family)
#   low nibble 7       = int16 sample-delta + raw LZMA
#   low nibble 4       = stored
# Decoders for old single-family frames keep working: their codes are a
# subset and unchanged.

def encode_frame_best(data: bytes, block=16, roster=(0, 1, 2, 3)) -> bytes:
    cands = [mrc.encode_frame_multi(data, block, roster)]
    if len(data) % 2 == 0 and len(data) >= 2:
        t = _sdz(data)
        cands.append(bytes([6]) + _ac_encode(t))
        cands.append(bytes([7]) + _lzma_raw(t))
    cands.append(bytes([4]) + data)
    return min(cands, key=len)


def decode_frame_best(blob: bytes, n: int) -> bytes:
    fmt = blob[0] & 0xF
    if fmt == 4:
        return blob[1:1 + n]
    if fmt == 6:
        return _unsdz(_ac_decode(blob[1:], n), n)
    if fmt == 7:
        return _unsdz(_unlzma_raw(blob[1:]), n)
    return mrc.decode_frame(blob, n)


def decode(blob: bytes) -> bytes:
    if blob[:4] == MAGIC4:
        n, fmt, aid = struct.unpack("<IBB", blob[4:10])
        body = blob[10:]
    else:
        assert blob[:4] == MAGIC3        # legacy container = atlas 0
        n, fmt = struct.unpack("<IB", blob[4:9])
        body, aid = blob[9:], 0
    if fmt in (1, 2, 7, 8):              # only engine formats read the atlas
        mrc.use_atlas(aid)
    if fmt == 3:
        dlt = np.frombuffer(_unlzma_raw(body), dtype=np.uint8)
        return np.cumsum(dlt.astype(np.int64)).astype(np.uint8).tobytes()[:n]
    if fmt == 4:
        return body[:n]
    if fmt == 5:
        return _unlzma_raw(body)[:n]
    if fmt == 6:
        return _unplane_delta(_unlzma_raw(body), n)
    if fmt == 9:
        return _unsdz(_unlzma_raw(body), n)
    if fmt == 10:
        return _unsdz(_ac_decode(body, n), n)
    deint = fmt >= 7
    if deint:
        fmt -= 6                  # 7/8 -> engine fmt 1/2 on the plane stream
    padded = ((n + 31) // 32) * 32

    if fmt == 1:
        raw = _unlzma_raw(body)
    else:  # fmt 2: lzma side-info + AC corrections; need side first
        (slen,) = struct.unpack("<I", body[:4])
        raw = _unlzma_raw(body[4:4 + slen])

    # parse size codes until they tile the padded length
    total, m = 0, 0
    while total < padded:
        total += SIZES[raw[m]]
        m += 1
    assert total == padded, "size stream does not tile file"
    size_codes = np.frombuffer(raw[:m], dtype=np.uint8)
    p = m
    mode = np.frombuffer(raw[p:p + m], dtype=np.uint8); p += m
    idxs = np.frombuffer(raw[p:p + 2 * m], dtype=np.uint16); p += 2 * m
    s_q = np.frombuffer(raw[p:p + m], dtype=np.int8); p += m
    o_tx = np.frombuffer(raw[p:p + 2 * m], dtype=np.int16); p += 2 * m
    o_q = np.cumsum(o_tx.astype(np.int64)).astype(np.int16)
    if fmt == 1:
        zzb = raw[p:p + padded]
    else:
        (slen,) = struct.unpack("<I", body[:4])
        zzb = _ac_decode(body[4 + slen:], padded)
    zz_all = np.frombuffer(zzb, dtype=np.uint8)

    seg_sizes = np.array([SIZES[c] for c in size_codes], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(seg_sizes)[:-1]])
    out = np.empty(padded, dtype=np.uint8)

    for code, size in enumerate(SIZES):
        sel_k = np.nonzero(size_codes == code)[0]
        if len(sel_k) == 0:
            continue
        C, _, _, _ = get_candidates(size, 0)
        sel = C[idxs[sel_k]]
        s_d = s_q[sel_k].astype(np.float32) / 16.0
        o_v = o_q[sel_k]
        t = np.arange(size, dtype=np.float32)
        pred_f = np.round(s_d[:, None] * sel + o_v[:, None].astype(np.float32))
        pred_l = np.round(s_d[:, None] * t[None, :] + o_v[:, None].astype(np.float32))
        pred = np.where((mode[sel_k] == 1)[:, None], pred_l, pred_f)
        zz_rows = np.stack([zz_all[starts[k]:starts[k] + size] for k in sel_k]).astype(np.int32)
        r_s = np.where(zz_rows % 2 == 0, zz_rows // 2, -(zz_rows + 1) // 2)
        rec = ((pred.astype(np.int32) + r_s) & 0xFF).astype(np.uint8)
        for row, k in enumerate(sel_k):
            out[starts[k]:starts[k] + size] = rec[row]
    res = out.tobytes()[:n]
    return _reint(res, n) if deint else res


def seg_stats(data: bytes):
    """Diagnostic: how the encoder cut a file (counts per block size)."""
    pad = (-len(data)) % 32
    buf = np.frombuffer(data + b"\x00" * pad, dtype=np.uint8)
    segments, _ = _segment(buf)
    counts = {s: 0 for s in SIZES}
    for c, _j in segments:
        counts[SIZES[c]] += 1
    return counts


if __name__ == "__main__":
    import sys
    op, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    data = open(src, "rb").read()
    out = encode(data) if op == "c" else decode(data)
    open(dst, "wb").write(out)
    print(f"{len(data)} -> {len(out)} bytes")
