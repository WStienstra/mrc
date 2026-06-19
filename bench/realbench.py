#!/usr/bin/env python3
"""MRC v2.3 on REAL vibration telemetry: seismometer (ground vibration,
Omnidots-analog) + CWRU machine-bearing accelerometer. Same protocol as
grind3: slices (5 offsets) + whole files, vs gzip-9/LZMA-9e/delta+LZMA,
plus a 64/128-byte frame-mode test. All roundtrip-verified."""
import json, time, zlib, lzma, sys, os, faulthandler
import numpy as np

faulthandler.enable()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import mrc, mrc3

SOURCES = {
    "seismo40": "seismo_40hz.bin",
    "seismo100": "seismo_100hz.bin",
    "bearing-ok": "bearing_normal.bin",
    "bearing-flt": "bearing_fault.bin",
}
OUT = "realbench-results"


def delta_lzma(d):
    a = np.frombuffer(d, dtype=np.uint8).astype(np.int16)
    dd = np.diff(a, prepend=a[:1]).astype(np.uint8) if len(a) else a.astype(np.uint8)
    return len(lzma.compress(dd.tobytes(), preset=9 | lzma.PRESET_EXTREME))


def sdelta16_lzma(d):
    """Strong int16-aware baseline: sample-delta + zigzag + plane split + LZMA
    (what a competent firmware engineer would hand-roll for int16 telemetry)."""
    if len(d) % 2:
        return 10**9
    s = np.frombuffer(d, dtype="<i2").astype(np.int32)
    sd = np.diff(s, prepend=0)
    zz = np.where(sd >= 0, 2 * sd, -2 * sd - 1).astype("<u2").tobytes()
    return len(lzma.compress(zz[0::2] + zz[1::2], preset=9 | lzma.PRESET_EXTREME))


def classics(d):
    return min(len(zlib.compress(d, 9)),
               len(lzma.compress(d, preset=9 | lzma.PRESET_EXTREME)),
               delta_lzma(d),
               sdelta16_lzma(d))


def v23(d):
    e = mrc3.encode(d)
    assert mrc3.decode(e) == d, "roundtrip fail"
    return len(e), e[8]


def log(f, line):
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


def main():
    results = {"slices": [], "whole": [], "frames": []}
    if os.path.exists(OUT + ".json.partial"):
        results = json.load(open(OUT + ".json.partial"))
    done_s = {(r["src"], r["size"], r["off"]) for r in results["slices"]}
    done_w = {(r["src"], r["size"]) for r in results["whole"]}
    done_f = {(r["src"], r["fsize"]) for r in results["frames"]}

    def ckpt():
        json.dump(results, open(OUT + ".json.partial", "w"))

    f = open(OUT + ".txt", "a")
    rng = np.random.default_rng(11)

    log(f, f"{'source':<13}{'size':>6}{'off':>8}{'classic':>9}{'v2.3':>7}{'fmt':>4}   saving")
    for name, path in SOURCES.items():
        full = open(path, "rb").read()
        for sz in (256, 512, 1024, 4096):
            offs = sorted(int(o) for o in rng.integers(0, len(full) - sz, 5))
            for off in offs:
                if (name, sz, off) in done_s:
                    continue
                d = full[off:off + sz]
                bc = classics(d)
                w3, fmt = v23(d)
                log(f, f"{name:<13}{sz:>6}{off:>8}{bc:>9}{w3:>7}{fmt:>4}   {100*(bc-w3)/bc:+.1f}%")
                results["slices"].append(dict(src=name, size=sz, off=off,
                                              classic=bc, v23=w3, fmt=fmt))
                ckpt()

    log(f, f"\n{'source':<13}{'size':>8}{'classic':>9}{'v2.3':>8}{'fmt':>4}   saving")
    for name, path in SOURCES.items():
        full = open(path, "rb").read()
        for sz in (16384, 65536, len(full)):
            if (name, sz) in done_w:
                continue
            d = full[:sz]
            t0 = time.time()
            bc = classics(d)
            w3, fmt = v23(d)
            log(f, f"{name:<13}{sz:>8}{bc:>9}{w3:>8}{fmt:>4}   {100*(bc-w3)/bc:+.1f}%  ({time.time()-t0:.0f}s)")
            results["whole"].append(dict(src=name, size=sz, classic=bc, v23=w3, fmt=fmt))
            ckpt()

    # frame mode: SWARM-style short bursts, transport-framed (no container)
    log(f, f"\nframe mode (transport-framed, 1-byte header) vs raw deflate / raw delta+lzma:")
    RAW = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
    for name, path in SOURCES.items():
        full = open(path, "rb").read()
        for fsize in (64, 128, 256):
            if (name, fsize) in done_f:
                continue
            offs = rng.integers(0, len(full) - fsize, 12)
            tot_m = tot_d = tot_l = 0
            for off in offs:
                off = int(off) & ~1          # int16 alignment
                d = full[off:off + fsize]
                e = mrc3.encode_frame16(d)
                assert mrc3.decode_frame16(e, fsize) == d
                co = zlib.compressobj(9, zlib.DEFLATED, -15)
                raw_deflate = co.compress(d) + co.flush()
                s = np.frombuffer(d, dtype="<i2").astype(np.int32)
                sd = np.diff(s, prepend=0)
                zz = np.where(sd >= 0, 2 * sd, -2 * sd - 1).astype("<u2").tobytes()
                raw_dlz = lzma.compress(zz[0::2] + zz[1::2], format=lzma.FORMAT_RAW, filters=RAW)
                tot_m += len(e); tot_d += len(raw_deflate); tot_l += len(raw_dlz)
            n = len(offs) * fsize
            log(f, f"{name:<13}{fsize:>4}B x12   MRC-planes {tot_m/12:6.1f}B ({100*(n/12-tot_m/12)/(n/12):+.0f}%)"
                   f"   deflate {tot_d/12:6.1f}B   sdelta16+lzma {tot_l/12:6.1f}B")
            results["frames"].append(dict(src=name, fsize=fsize, mrc=tot_m, deflate=tot_d, dlzma=tot_l))
            ckpt()

    json.dump(results, open(OUT + ".json", "w"), indent=1)
    log(f, "\nDONE realbench")
    f.close()


if __name__ == "__main__":
    main()
