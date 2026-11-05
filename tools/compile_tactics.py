"""Write Connect Four tactics directly into the integer trunk of a hard checkpoint.

The 32 learned channels are kept bit-for-bit. New channels are appended and
programmed by hand; each is an ordinary trunk channel:

    state = SBOX[ XOR_f  coef[f] (x) h_prev[ch[f]] shifted by lag[f] ]

with `coef` a GF(256) constant and SBOX a 256-entry integer table. Nothing here
needs a float except the readout, which (as in every variant) stays float.

Circuit, evaluated at the TURN byte of a ply (column k sits at lag 8-k):
  layer 0 (emb)   per-token cell state of row r:
                  0 empty/other, 1 P1, 2 P2, 3 playable, 4 empty directly above
                  the playable cell; plus a turn channel and a vertical-3 flag.
  layer 1         "pair" nodes: two cells + turn -> mover-relative summary
                  (both mine / both theirs / one special cell + owner of other);
                  "vertical" nodes: column flag + turn.
  layer 2         "line" nodes: two pair summaries -> which cell of a 4-cell line
                  is a win (mine), a block (theirs), or an unsafe cell (theirs,
                  directly above the playable one).
  readout         adds WIN / BLOCK / UNSAFE weights to the 7 move-column logits.

Usage:
    .venv-c4\\Scripts\\python.exe tools\\compile_tactics.py \\
        --src runs/_probe/v24/checkpoints/best_hard.pt \\
        --out runs/_probe/v24_tactics/best_hard.pt
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.gf256_markov_lm_v6 import gf_mul  # noqa: E402

COLS, ROWS = 7, 6
TURN_P1, TURN_P2 = 0x58, 0x4F
COL_OFF = 0x41
FIRST_NEW = 32

# cell state codes (absolute)
EMPTY, P1C, P2C, PLAYABLE, NEXT = 0, 1, 2, 3, 4
# relative codes inside pair nodes
OTHER, ME, OPP, PLAY, NXT = 0, 1, 2, 3, 4


def col_cell(byte, r):
    """Absolute state of row r for a V2 column byte."""
    if not byte & 0x80:
        return EMPTY
    v = byte & 0x7F
    h = v.bit_length() - 1
    if r < h:
        return P1C if (v >> r) & 1 else P2C
    if r == h:
        return PLAYABLE
    if r == h + 1:
        return NEXT
    return EMPTY


def col_vertical(byte):
    """1/2 if the top three stones of a playable column belong to P1/P2."""
    if not byte & 0x80:
        return 0
    v = byte & 0x7F
    h = v.bit_length() - 1
    if not 3 <= h <= 5:
        return 0
    bits = [(v >> r) & 1 for r in (h - 1, h - 2, h - 3)]
    return 1 if all(bits) else 2 if not any(bits) else 0


def rel(cell, turn):
    """Absolute cell state -> mover-relative state."""
    if cell in (P1C, P2C):
        mover = P1C if turn == 1 else P2C
        return ME if cell == mover else OPP
    return {EMPTY: OTHER, PLAYABLE: PLAY, NEXT: NXT}[cell]


def pair_summary(x, y):
    """Summary of two relative cells that is sufficient for any 4-cell line."""
    if x == ME and y == ME:
        return 1
    if x == OPP and y == OPP:
        return 2
    for pos, (s, o) in enumerate(((x, y), (y, x))):
        if s in (PLAY, NXT) and o in (ME, OPP):
            return 3 + (((s - PLAY) * 2 + pos) * 2 + (o - ME))
    return 0


def line_code(s1, s2):
    """Pair summaries -> 0 none, else 1 + ((kind*2 + who)*4 + j)."""
    def decode(s):
        if s in (1, 2):
            return ("full", ME if s == 1 else OPP)
        if s >= 3:
            k = s - 3
            return ("spec", (k // 4) % 2, (k // 2) % 2, ME if k % 2 == 0 else OPP)
        return None
    a, b = decode(s1), decode(s2)
    if a is None or b is None:
        return 0
    if a[0] == "full" and b[0] == "spec" and a[1] == b[3]:
        kind, pos, who, base = b[1], b[2], a[1], 2
    elif a[0] == "spec" and b[0] == "full" and b[1] == a[3]:
        kind, pos, who, base = a[1], a[2], b[1], 0
    else:
        return 0
    return 1 + ((kind * 2 + (who - ME)) * 4 + base + pos)


class Net:
    """Collects emb columns and per-layer nodes, then emits export tensors."""

    def __init__(self, d, layers):
        self.d, self.layers = d, layers
        self.emb = {}                          # ch -> list[256]
        self.nodes = [dict() for _ in range(layers)]   # layer -> ch -> node
        self.next_ch = [FIRST_NEW] * (layers + 1)

    def add_emb(self, fn):
        ch = self.next_ch[0]
        self.next_ch[0] += 1
        self.emb[ch] = [fn(t) for t in range(256)]
        return ch

    def add_node(self, layer, parents, widths, fn):
        """parents: [(ch, lag)] from layer-1 outputs; widths: bits per parent."""
        ch = self.next_ch[layer]
        self.next_ch[layer] += 1
        self.nodes[layer - 1][ch] = (parents, widths, fn)
        return ch


def build(src):
    lags = [int(x) for x in src["lags"]]
    lag_idx = {l: i for i, l in enumerate(lags)}
    net = Net(0, 3)

    cell_ch = [net.add_emb(lambda t, r=r: col_cell(t, r)) for r in range(ROWS)]
    vert_ch = net.add_emb(col_vertical)
    turn_ch = net.add_emb(lambda t: 1 if t == TURN_P1 else 2 if t == TURN_P2 else 0)

    out_rows = []   # (layer, ch, state, col, weight) readout writes
    pair_cache = {}

    def lag_of(k):
        return 8 - k

    def pair_node(c1, c2):
        key = (c1, c2)
        if key in pair_cache:
            return pair_cache[key]
        (r1, k1), (r2, k2) = c1, c2
        parents = [(cell_ch[r1], lag_of(k1)), (cell_ch[r2], lag_of(k2)),
                   (turn_ch, 0)]

        def fn(a, b, t):
            if t not in (1, 2) or a > NXT or b > NXT:
                return 0
            return pair_summary(rel(a, t), rel(b, t))
        ch = net.add_node(1, parents, (3, 3, 2), fn)
        pair_cache[key] = ch
        return ch

    lines = []
    for r in range(ROWS):
        for a in range(COLS - 3):
            lines.append([(r, a + i) for i in range(4)])
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            lines.append([(r + i, c + i) for i in range(4)])
            lines.append([(r + i, c + 3 - i) for i in range(4)])

    weights = {(0, ME): WIN, (0, OPP): BLOCK, (1, OPP): UNSAFE, (1, ME): 0.0}
    for cells in lines:
        p1 = pair_node(cells[0], cells[1])
        p2 = pair_node(cells[2], cells[3])
        ch = net.add_node(2, [(p1, 0), (p2, 0)], (4, 4), line_code)
        for code in range(1, 17):
            k = code - 1
            kind, who, j = (k // 4) // 2, (k // 4) % 2, k % 4
            w = weights[(kind, ME + who)]
            if w:
                out_rows.append((2, ch, code, cells[j][1], w))

    for k in range(COLS):
        def vfn(v, t):
            if v not in (1, 2) or t not in (1, 2):
                return 0
            return 1 if v == t else 2
        ch = net.add_node(1, [(vert_ch, lag_of(k)), (turn_ch, 0)], (2, 2), vfn)
        out_rows.append((1, ch, 1, k, WIN))
        out_rows.append((1, ch, 2, k, BLOCK))
    return net, out_rows, lag_idx


WIN, BLOCK, UNSAFE = 1000.0, 300.0, -40.0


def emit(src, net, out_rows, lag_idx):
    d_old = src["emb"].shape[1]
    d = max(net.next_ch)
    L = len(src["coef"])
    fan = src["coef"][0].shape[1]
    assert d_old == FIRST_NEW and L >= 2 and fan >= 3

    emb = torch.zeros(256, d, dtype=torch.long)
    emb[:, :d_old] = src["emb"]
    for ch, col in net.emb.items():
        emb[:, ch] = torch.tensor(col)

    coef, sbox, wiring = [], [], []
    for l in range(L):
        c = torch.zeros(d, fan, dtype=torch.long)
        s = torch.zeros(d, 256, dtype=torch.long)
        ch_t = torch.zeros(d, fan, dtype=torch.long)
        lg_t = torch.zeros(d, fan, dtype=torch.long)
        c[:d_old] = src["coef"][l]
        s[:d_old] = src["sbox"][l]
        ch_t[:d_old] = src["wiring"][l][0]
        lg_t[:d_old] = src["wiring"][l][1]
        if l < 2:
            for ch, (parents, widths, fn) in net.nodes[l].items():
                shift = 0
                doms = []
                for f, ((pch, lag), w) in enumerate(zip(parents, widths)):
                    c[ch, f] = 1 << shift
                    ch_t[ch, f] = pch
                    lg_t[ch, f] = lag_idx[lag]
                    doms.append(range(1 << w))
                    shift += w
                assert shift <= 8
                seen = {}
                for vals in itertools.product(*doms):
                    x = 0
                    for f, v in enumerate(vals):
                        x ^= gf_mul(int(c[ch, f]), v)
                    out = fn(*vals)
                    assert seen.setdefault(x, out) == out, "packing collision"
                    s[ch, x] = out
        coef.append(c)
        sbox.append(s)
        wiring.append((ch_t, lg_t))

    readout = []
    for R in src["readout"]:
        new = torch.zeros(d, 256, 256, dtype=R.dtype)
        new[:d_old] = R
        readout.append(new)
    for layer, ch, state, col, w in out_rows:
        readout[layer][ch, state, COL_OFF + col] += w

    out = dict(src)
    out.update(emb=emb, coef=coef, sbox=sbox, wiring=wiring, readout=readout,
               d=d, tactics_compiled=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = torch.load(a.src, map_location="cpu", weights_only=False)
    net, rows, lag_idx = build(src)
    out = emit(src, net, rows, lag_idx)
    print(f"channels: {src['emb'].shape[1]} -> {out['emb'].shape[1]}; "
          f"layer1 nodes {len(net.nodes[0])}, layer2 nodes {len(net.nodes[1])}")
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, p)
    print("wrote", p, f"{p.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
