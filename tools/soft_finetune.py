"""compile -> small-margin soft import -> STE fine-tune -> hard export.

Takes a hard payload (e.g. from tools/compile_tactics.py), builds a soft V6
model of the same shape whose coefficient set is the union of every constant in
the payload, and writes each integer table entry as a logit bonus of `margin`.
Training runs with the straight-through estimator, so the FORWARD pass is the
integer program (argmax tables) and gradients flow through the softmax. A small
margin keeps those gradients alive; a large one (import_hard uses 10) freezes
the tables.

Loss: solver tie-set NLL on the move columns, weighted by tactic class, read at
the move slot of a 64-byte window (receptive field of 3 layers x lag<=20 = 60).

Usage:
    .venv-c4\\Scripts\\python.exe tools\\soft_finetune.py \\
        --src runs/_probe/v24_tactics/best_hard.pt --positions runs/_probe/v24_surgery/positions.pt \\
        --out runs/_probe/v24_soft/best_hard.pt
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from readout_surgery import fmt_cls, legal_mask  # noqa: E402
from src.c4 import COLS  # noqa: E402
from src.c4.agent import (COL_BYTE_OFFSET, prompt_tokens,  # noqa: E402
                          trainer_window)
from src.gf256_markov_lm_v6 import GFMarkovLM, gf_mul  # noqa: E402

WIN_T = 64
SL = slice(COL_BYTE_OFFSET, COL_BYTE_OFFSET + COLS)


def build_soft(payload, margin, device):
    d = int(payload["emb"].shape[1])
    L = len(payload["coef"])
    fan = int(payload["coef"][0].shape[1])
    lags = [int(x) for x in payload["lags"]]
    m = GFMarkovLM(d=d, layers=L, fan_in=fan, lags=tuple(lags), n_coef=8)

    vals = sorted({int(v) for c in payload["coef"] for v in c.flatten().tolist()}
                  | {0, 1})
    m.coef_set = vals
    pm = torch.zeros(len(vals), 256, 256)
    for k, c in enumerate(vals):
        for i in range(256):
            pm[k, i, gf_mul(c, i)] = 1.0
    m.perm_mats = pm
    pos = {v: i for i, v in enumerate(vals)}

    with torch.no_grad():
        emb = torch.zeros(256, d, 256)
        emb[torch.arange(256)[:, None], torch.arange(d)[None, :],
            payload["emb"]] = margin
        m.emb_logits.copy_(emb)
        ar_d, ar256 = torch.arange(d), torch.arange(256)
        for l, layer in enumerate(m.layers):
            cl = torch.zeros(d, fan, len(vals))
            idx = torch.tensor([[pos[int(v)] for v in row]
                                for row in payload["coef"][l].tolist()])
            cl.scatter_(2, idx.unsqueeze(2), margin)
            layer.coef_logits = nn.Parameter(cl)
            sb = torch.zeros(d, 256, 256)
            sb[ar_d[:, None], ar256[None, :], payload["sbox"][l]] = margin
            layer.sbox_logits.copy_(sb)
            ch, lg = payload["wiring"][l]
            layer.ch = ch.long()
            layer.lag = lg.long()
            layer.lag_vals = torch.tensor(lags)[lg.long()]
            layer.fan_in = fan
        for R, Rn in zip(m.readout, payload["readout"]):
            R.copy_(Rn)
        m.readout_bias.copy_(payload["bias"])
    m.set_ste(True)
    return m.to(device)


def export(m, payload):
    hard = m.export_hard()
    out = dict(payload)
    out.update({k: ([x.cpu() if torch.is_tensor(x) else x for x in v]
                    if isinstance(v, list) else v.cpu())
                for k, v in hard.items() if k not in ("wiring", "lags")})
    out["coef"] = [c.cpu() for c in hard["coef"]]
    out["sbox"] = [s.cpu() for s in hard["sbox"]]
    out["wiring"] = [(a.cpu(), b.cpu()) for a, b in hard["wiring"]]
    out["readout"] = [r.cpu() for r in hard["readout"]]
    out["emb"] = hard["emb"].cpu()
    out["bias"] = hard["bias"].cpu()
    return out


def windows(positions, version):
    xs, ri = [], []
    for moves, _ in positions:
        x, r = trainer_window(prompt_tokens(moves, version).tolist(), WIN_T)
        xs.append(x)
        ri.append(r)
    return torch.tensor(xs), torch.tensor(ri)


def col_logits(m, x, ri):
    lg = m(x)
    return lg[torch.arange(len(x), device=x.device), ri][:, SL]


@torch.no_grad()
def evaluate(m, x, ri, legal, target, cls, bs=128):
    m.eval()
    outs = [col_logits(m, x[i:i + bs], ri[i:i + bs]) for i in range(0, len(x), bs)]
    lg = torch.cat(outs).masked_fill(~legal, -1e9)
    pick = lg.argmax(1)
    return target[torch.arange(len(pick), device=lg.device), pick].float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--w-tactic", type=float, default=1.0)
    ap.add_argument("--w-strict", type=float, default=3.0)
    ap.add_argument("--parity", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)

    payload = torch.load(a.src, map_location="cpu", weights_only=False)
    version = payload.get("transcript", "v2")
    blob = torch.load(a.positions, weights_only=False)
    tr, ho = blob["tr"], blob["ho"]
    x_tr, r_tr = windows(tr, version)
    x_ho, r_ho = windows(ho, version)
    lg_tr, tg_tr, c_tr = legal_mask(tr)
    lg_ho, tg_ho, c_ho = legal_mask(ho)
    x_tr, r_tr, lg_tr, tg_tr, c_tr = [t.to(dev) for t in (x_tr, r_tr, lg_tr, tg_tr, c_tr)]
    x_ho, r_ho, lg_ho, tg_ho = [t.to(dev) for t in (x_ho, r_ho, lg_ho, tg_ho)]

    m = build_soft(payload, a.margin, dev)
    print(f"soft model d={m.d} coef_set={len(m.coef_set)} "
          f"params={sum(p.numel() for p in m.parameters()) / 1e6:.0f}M", flush=True)

    if a.parity:
        from src.c4.duel import _load_hard_agent
        ag = _load_hard_agent(a.src)
        n = a.parity
        with torch.no_grad():
            hard = ag.model.hard_forward(x_ho[:n].cpu(), ag.hard)
            hh = hard[torch.arange(n), r_ho[:n].cpu()][:, SL]
            ss = col_logits(m.eval(), x_ho[:n], r_ho[:n]).cpu()
        agree = (hh.argmax(1) == ss.argmax(1)).float().mean().item()
        print(f"parity soft(STE)-vs-hard on {n}: max|diff|={(hh - ss).abs().max():.4g} "
              f"argmax agree={agree:.4f}", flush=True)

    cw = torch.tensor([a.w_tactic, a.w_tactic, a.w_strict, 1.0], device=dev)
    c_ho_cpu = c_ho

    def report(tag):
        hit = evaluate(m, x_ho, r_ho, lg_ho, tg_ho, c_ho_cpu)
        print(f"{tag} holdout {fmt_cls(hit, c_ho_cpu)}", flush=True)
        return sum(float(hit[c_ho_cpu == k].mean()) for k in range(3)) / 3

    best = report("step 0   ")
    best_out = export(m, payload)
    best_step = 0
    m.train()
    m.use_ckpt = False
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        j = torch.randint(0, len(x_tr), (a.bs,), device=dev)
        lg = col_logits(m, x_tr[j], r_tr[j]).masked_fill(~lg_tr[j], -1e9)
        lp = F.log_softmax(lg, 1)
        nll = -torch.logsumexp(lp.masked_fill(~tg_tr[j], -1e9), 1)
        w = cw[c_tr[j]]
        loss = (nll * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0:
            print(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
            sc = report(f"step {step:<4d}")
            if sc > best:
                best, best_step, best_out = sc, step, export(m, payload)
            m.train()

    print(f"selected step {best_step} (mean of win1/block1/strict = {best:.3f}; "
          f"chosen on this holdout, so confirm in games)")
    out = best_out
    out["soft_finetune"] = {"margin": a.margin, "steps": best_step, "lr": a.lr}
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, p)
    print("wrote", p)


if __name__ == "__main__":
    main()
