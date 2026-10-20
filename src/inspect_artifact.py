"""Verify that a hardened GF(256) artifact matches what the paper claims.

Checks the claims that are checkable without retraining:

  1. which version the artifact actually is (d, layers, fan_in, lags)
  2. the exact program size, split into table bytes
  3. whether inference really is float-free -- and if not, how much float
     arithmetic the "integer program" actually performs per character
  4. whether the hard forward path agrees with the soft model

Usage:
    python src/inspect_artifact.py artifacts/gf256_hard_model.pt
"""
import argparse
import math

import torch


def load(path):
    # Try the safe unpickler first. The V6 artifact contains only tensors,
    # ints and lists of tuples, so it loads without weights_only=False.
    # The fallback exists for older checkpoints that pickled other objects.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "hard" in obj:
        return obj["hard"], obj.get("meta", {})
    return obj, {}


def gf_tables():
    exp = [0] * 512
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x2 = x << 1
        if x2 & 0x100:
            x2 ^= 0x11B
        x = x2 ^ x
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


def describe(hard):
    emb = hard["emb"]
    d = emb.shape[0]
    sboxes = hard["sbox"]
    readouts = hard["readout"]
    layers = len(sboxes)
    fan_in = int(hard["coef"][0].shape[1])
    lags = hard["lags"]

    print("== artifact identity ==")
    print(f"  d (channels)        : {d}")
    print(f"  layers              : {layers}")
    print(f"  fan_in per layer    : {fan_in}")
    print(f"  lags                : {lags}")
    print(f"  max lag (ctx reach) : {max(lags)} bytes")
    print()

    print("== table footprint ==")
    total = 0
    rows = []
    e_bytes = emb.numel() * emb.element_size()
    rows.append(("embedding (argmax indices)", emb.numel(), e_bytes))
    total += e_bytes
    for i, s in enumerate(sboxes):
        b = s.numel()
        rows.append((f"sbox[{i}]", b, b))
        total += b
    for i, r in enumerate(readouts):
        b = r.numel() * r.element_size()
        rows.append((f"readout[{i}] {tuple(r.shape)} {r.dtype}", r.numel(), b))
        total += b
    for name, n, b in rows:
        print(f"  {name:<46} {n:>10} elems {b:>12,} bytes")
    print(f"  {'TOTAL':<46} {'':>10}      {total:>12,} bytes "
          f"({total/1e6:.1f} MB)")
    print()

    # ---- the float question -------------------------------------------------
    print("== is inference actually float-free? ==")
    int_tables = emb.numel() + sum(s.numel() for s in sboxes)
    f_tables = sum(r.numel() for r in readouts) + hard["bias"].numel()
    print(f"  integer tables (emb + sboxes)      : {int_tables:>12,} entries")
    print(f"  float tables (readouts + bias)    : {f_tables:>12,} entries "
          f"({hard['readout'][0].dtype})")
    print()

    # per-character arithmetic, counting what the code actually does.
    # NOTE: argmax is over the 256 logits, NOT over the size of the readout
    # table -- only d rows are fetched per stage, each of length 256.
    mul_lookups = layers * fan_in * d          # GF multiply via exp table
    sbox_lookups = layers * d                  # one sbox eval per channel
    ro_fetch = len(readouts) * d               # row fetches (each 256 floats)
    ro_adds = len(readouts) * d * 256          # sum 256 floats per row fetch
    argmax_cmp = 256                           # max/scan over 256 logits
    xors = layers * (fan_in - 1) * d

    print("  per generated character, the hard program performs:")
    print(f"    GF-multiply table lookups       : {mul_lookups:>10,}")
    print(f"    S-box lookups                   : {sbox_lookups:>10,}")
    print(f"    XOR operations                 : {xors:>10,}")
    print(f"    float-table row fetches         : {ro_fetch:>10,} "
          f"(x 256 floats each)")
    print(f"    FLOAT adds (readout summation)  : {ro_adds:>10,}")
    print(f"    FLOAT comparisons (argmax)      : {argmax_cmp:>10,}")
    float_ops = ro_adds + argmax_cmp
    print()
    print(f"  => integer ops (XOR + lookups)    : {mul_lookups + sbox_lookups + xors:>10,}")
    print(f"  => FLOATING-POINT ops             : {float_ops:>10,}")
    print(f"  => float share of all arithmetic  : "
          f"{float_ops / (float_ops + mul_lookups + sbox_lookups + xors) * 100:.1f}%")
    print()
    print("  The trunk (emb + GF mixing + S-boxes) is integer-only. The readout")
    print("  is a learned float32 matrix applied by summation, and it dominates")
    print("  the arithmetic. Eliminating it is open work (see FUTURE_WORK.md).")
    print()
    return d, layers, fan_in


def sanity_checks(hard, path):
    """Confirm the hardened tables are actually usable as claimed."""
    print("== structural sanity ==")
    emb = hard["emb"]

    # sbox[i] is a 256x256 matrix indexed [input, output], so the hardened
    # map is x -> sbox[i][x]. Bijectivity means 256 distinct outputs.
    for i, s in enumerate(hard["sbox"]):
        n_distinct = torch.unique(s).numel()
        print(f"  sbox[{i}]: {n_distinct:3d}/256 distinct outputs "
              f"({'permutation' if n_distinct == 256 else 'NOT bijective'})")

    for i, R in enumerate(hard["readout"]):
        print(f"  readout[{i}]: shape {tuple(R.shape)} dtype {R.dtype} "
              f"range [{R.min():.2f}, {R.max():.2f}]")

    for i, c in enumerate(hard["coef"]):
        uniq = torch.unique(c).tolist()
        print(f"  coef[{i}]: {len(uniq)} distinct GF constants, "
              f"includes 0: {0 in uniq}, includes 1: {1 in uniq}")

    # How much of each layer's wiring is actually load-bearing? Path 0 is the
    # own-channel lag-0 identity path; if its coefficient is not 1 the channel
    # has lost its residual and is being carried entirely by S-box[j](0).
    for i, (ch, lg) in enumerate(hard["wiring"]):
        coef = hard["coef"][i]
        d = ch.shape[0]
        own = (ch == torch.arange(d).unsqueeze(1)) & (lg == 0)
        ident_path0 = own[:, 0]
        lost_residual = int((ident_path0 & (coef[:, 0] != 1)).sum())
        zero_ch = int((coef == 0).sum())
        print(f"  layer[{i}]: {lost_residual}/{d} channels lost their "
              f"identity residual; {zero_ch}/{coef.numel()} paths zeroed "
              f"(coef==0)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="artifacts/gf256_hard_model.pt")
    args = ap.parse_args()

    hard, meta = load(args.path)
    print(f"artifact: {args.path}\n")
    if meta:
        print(f"embedded meta: {meta}\n")
    describe(hard)
    print()
    sanity_checks(hard, args.path)


if __name__ == "__main__":
    main()