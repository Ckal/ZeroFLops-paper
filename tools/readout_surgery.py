"""Direct edit of a hard checkpoint's move-column readout weights.

Reads the integer trunk + float readout tables from a `*_hard.pt` export and
changes ONLY the 7 move-column slices (bytes 0x41..0x47) of every readout table
and of the bias. Embedding, coefficients, S-boxes and wiring are not touched,
so the integer trunk is bit-identical to the source checkpoint.

At the move position the column logit is
    bias[c] + sum_l sum_j R_l[j, h_l[j], c]
with h_l the integer state of channel j after layer l. That is linear in a
one-hot of the trunk states, so the edit is a convex fit: cross-entropy of the
solver's optimal-move set, with an L2 pull toward the original weights.

Usage:
    .venv-c4\\Scripts\\python.exe tools\\readout_surgery.py \\
        --src runs/_probe/v24/checkpoints/best_hard.pt \\
        --out runs/_probe/v24_surgery/checkpoints/best_hard.pt
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.c4 import COLS  # noqa: E402
from src.c4.agent import (COL_BYTE_OFFSET, prompt_tokens,  # noqa: E402
                          trainer_window)
from src.c4.board import Board, is_terminal  # noqa: E402
from src.c4.duel import _load_hard_agent  # noqa: E402
from src.c4 import opponents  # noqa: E402
from src.c4.strong_solver import strong_policy_set, clear_tt  # noqa: E402
from src.gf256_markov_lm_v6 import EXP, LOG  # noqa: E402


@torch.no_grad()
def trunk_states(model, hard, tokens):
    """Integer states after the embedding and after each layer: (B,T,L+1,d)."""
    exp_t, log_t = torch.tensor(EXP), torch.tensor(LOG)
    lags = torch.tensor(hard["lags"])
    B, T, d = tokens.shape[0], tokens.shape[1], model.d
    h = hard["emb"][tokens]
    ar = torch.arange(d)
    states = [h]
    for coef, sbox, (ch, lg) in zip(hard["coef"], hard["sbox"], hard["wiring"]):
        acc = torch.zeros(B, T, d, dtype=torch.long)
        for f in range(coef.shape[1]):
            out = torch.zeros(B, T, d, dtype=torch.long)
            for j in range(d):
                l = int(lags[lg[j, f]])
                src = h[:, :, ch[j, f]]
                if l:
                    src = torch.cat([torch.zeros(B, l, dtype=torch.long),
                                     src[:, :-l]], 1)
                c = int(coef[j, f])
                if c == 0:
                    prod = torch.zeros_like(src)
                else:
                    prod = torch.where(src == 0, torch.zeros_like(src),
                                       exp_t[log_t[src] + LOG[c]])
                out[:, :, j] = prod
            acc = acc ^ out
        h = sbox[ar, acc]
        states.append(h)
    return torch.stack(states, 2)


def features(agent, boards_moves, chunk=32):
    """(N, L+1, d) integer states at each position's move-read slot."""
    out = []
    for i in range(0, len(boards_moves), chunk):
        xs, ridx = [], []
        for moves in boards_moves[i:i + chunk]:
            x, r = trainer_window(prompt_tokens(moves, agent.version).tolist(),
                                  agent.ctx)
            xs.append(x)
            ridx.append(r)
        st = trunk_states(agent.model, agent.hard, torch.tensor(xs))
        out.append(st[torch.arange(len(xs)), torch.tensor(ridx)])
    return torch.cat(out)


def label(moves, depth, budget):
    b = Board()
    for c in moves:
        b.apply_move(c)
    cols, _v, reached = strong_policy_set(b, moves, depth=depth, budget=budget)
    return cols, reached


def gen_positions(agent, n_games, seed, depth, budget, eps_model=0.25):
    """Positions from games between mixed players; each gets a solver label."""
    rng = random.Random(seed)
    pols = {k: opponents.get(k) for k in
            ("random", "centre", "heuristic", "lookahead")}
    seen, pos = set(), []
    for g in range(n_games):
        kind = rng.choice(["model", "model", "random", "heuristic",
                           "lookahead", "centre"])
        model_side = rng.choice([0, 1])
        b, moves = Board(), []
        while not is_terminal(b):
            if moves not in ([],) and tuple(moves) not in seen:
                seen.add(tuple(moves))
                cols, reached = label(moves, depth, budget)
                if cols:
                    pos.append((list(moves), cols))
            mine = (len(moves) % 2) == model_side
            legal = b.valid_actions()
            if kind == "model" or mine:
                if rng.random() < eps_model:
                    c = rng.choice(legal)
                elif kind == "model":
                    c = rng.choice(legal) if rng.random() < 0.3 else \
                        int(agent.choose(b, moves, use_hard=True))
                else:
                    c = int(agent.choose(b, moves, use_hard=True))
            else:
                c = pols[kind](b, moves)
            b.apply_move(c)
            moves.append(c)
            if b.is_win():
                break
        clear_tt()
    return pos


def legal_mask(positions):
    """Legal mask, target mask, and a class per position.

    class 0 = a win-in-1 exists, 1 = must block, 2 = unique/near-unique best
    move (<=2 optimal), 3 = other (mostly ties, weakly informative).
    """
    from src.c4.strong_solver import _imminent_win_columns
    m = torch.zeros(len(positions), COLS, dtype=torch.bool)
    t = torch.zeros(len(positions), COLS, dtype=torch.bool)
    cls = torch.full((len(positions),), 3, dtype=torch.long)
    for i, (moves, cols) in enumerate(positions):
        b = Board()
        for c in moves:
            b.apply_move(c)
        for c in b.valid_actions():
            m[i, c] = True
        for c in cols:
            t[i, c] = True
        side = b.turn
        opp = 2 if side == 1 else 1
        if _imminent_win_columns(b, side):
            cls[i] = 0
        elif _imminent_win_columns(b, opp):
            cls[i] = 1
        elif len(cols) <= 2:
            cls[i] = 2
    return m, t, cls


def base_logits(hard, st):
    """(N,7) logits from the ORIGINAL weights; st is (N,L+1,d)."""
    N, L1, d = st.shape
    ar = torch.arange(d)
    sl = slice(COL_BYTE_OFFSET, COL_BYTE_OFFSET + COLS)
    out = hard["bias"][sl].expand(N, COLS).clone()
    for l in range(L1):
        out += hard["readout"][l][:, :, sl][ar[None, :], st[:, l, :]].sum(1)
    return out


def delta_logits(W, st):
    N, L1, d = st.shape
    off = (torch.arange(L1)[:, None] * d + torch.arange(d)[None, :]) * 256
    idx = (st + off[None]).reshape(N, -1)
    return W[idx].sum(1)


def metrics(logits, legal, target):
    masked = logits.masked_fill(~legal, -1e9)
    pick = masked.argmax(1)
    return target[torch.arange(len(pick)), pick].float()


def by_class(hit, cls):
    return [float(hit[cls == k].mean()) if (cls == k).any() else float("nan")
            for k in range(4)]


def fmt_cls(hit, cls):
    c = by_class(hit, cls)
    return (f"all {hit.mean():.3f} | win1 {c[0]:.3f} block1 {c[1]:.3f} "
            f"strict {c[2]:.3f} tie {c[3]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-games", type=int, default=300)
    ap.add_argument("--holdout-games", type=int, default=80)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--w-tactic", type=float, default=6.0)
    ap.add_argument("--w-strict", type=float, default=3.0)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)

    agent = _load_hard_agent(a.src)
    hard = agent.hard
    cache = pathlib.Path(a.cache) if a.cache else None
    if cache and cache.exists():
        blob = torch.load(cache, weights_only=False)
        tr, ho = blob["tr"], blob["ho"]
    else:
        t0 = time.time()
        tr = gen_positions(agent, a.train_games, 1000, a.depth, a.budget)
        ho = gen_positions(agent, a.holdout_games, 2000, a.depth, a.budget)
        ho_keys = {tuple(m) for m, _ in ho}
        tr = [p for p in tr if tuple(p[0]) not in ho_keys]
        print(f"positions train={len(tr)} holdout={len(ho)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"tr": tr, "ho": ho}, cache)

    t0 = time.time()
    st_tr = features(agent, [m for m, _ in tr])
    st_ho = features(agent, [m for m, _ in ho])
    print(f"features {time.time() - t0:.0f}s", flush=True)
    lg_tr, tg_tr, c_tr = legal_mask(tr)
    lg_ho, tg_ho, c_ho = legal_mask(ho)
    b_tr, b_ho = base_logits(hard, st_tr), base_logits(hard, st_ho)
    print(f"class counts holdout (win1/block1/strict/tie): "
          f"{[int((c_ho == k).sum()) for k in range(4)]}")
    print("BEFORE  holdout", fmt_cls(metrics(b_ho, lg_ho, tg_ho), c_ho))

    def tactic_score(hit, cls):
        c = by_class(hit, cls)
        return sum(c[:3]) / 3

    cw = torch.tensor([a.w_tactic, a.w_tactic, a.w_strict, 1.0])
    torch.set_grad_enabled(True)  # loading the agent leaves autograd off
    L1, d = st_tr.shape[1], st_tr.shape[2]
    W = torch.zeros(L1 * d * 256, COLS, requires_grad=True)
    opt = torch.optim.Adam([W], lr=a.lr)
    # epoch -1 is the unmodified model, so "no edit" can win the selection
    best = tactic_score(metrics(b_ho, lg_ho, tg_ho), c_ho)
    best_W = torch.zeros_like(W.detach())
    for ep in range(a.epochs):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 256):
            j = perm[i:i + 256]
            lg = (b_tr[j] + delta_logits(W, st_tr[j])).masked_fill(~lg_tr[j], -1e9)
            lp = F.log_softmax(lg, 1)
            nll = -torch.logsumexp(lp.masked_fill(~tg_tr[j], -1e9), 1)
            w = cw[c_tr[j]]
            loss = (nll * w).sum() / w.sum() + a.l2 * (W ** 2).sum() / len(tr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            h_hit = metrics(b_ho + delta_logits(W, st_ho), lg_ho, tg_ho)
            t_hit = metrics(b_tr + delta_logits(W, st_tr), lg_tr, tg_tr)
        sc = tactic_score(h_hit, c_ho)
        if sc > best:
            best, best_W = sc, W.detach().clone()
        if ep % 5 == 0:
            print(f"ep {ep:3d} train all {t_hit.mean():.3f} | holdout "
                  f"{fmt_cls(h_hit, c_ho)}", flush=True)
    with torch.no_grad():
        fin = metrics(b_ho + delta_logits(best_W, st_ho), lg_ho, tg_ho)
    print("AFTER   holdout", fmt_cls(fin, c_ho),
          "(epoch picked on this holdout; confirm in games)")

    # Fold the best delta into the actual readout tables.
    new = {k: v for k, v in agent.hard.items()}
    new["readout"] = [R.clone() for R in hard["readout"]]
    new["bias"] = hard["bias"].clone()
    sl = slice(COL_BYTE_OFFSET, COL_BYTE_OFFSET + COLS)
    Wd = best_W.reshape(L1, d, 256, COLS)
    for l in range(L1):
        new["readout"][l][:, :, sl] += Wd[l]
    n_changed = int((Wd != 0).sum())
    print(f"edited {n_changed} readout entries of "
          f"{sum(R.numel() for R in hard['readout'])}")
    new["surgery"] = {"src": str(a.src), "train_positions": len(tr),
                      "holdout_positions": len(ho), "depth": a.depth}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
