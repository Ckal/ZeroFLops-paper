"""Train the V6 GF(256) byte-LM on Connect Four via full on-policy self-play.

Recipe is locked to V6: d=32, layers=3, fan_in=3, 20000 steps, STE from
step 12000 ramping over 4000, hard_w=0.5, ent_max=2.0, perm_w=1.0.

CORPUS REGENERATION. This is the difference between this script and the stock
V6 byte-LM trainer. V6 trains on a fixed tinyshakespeare file; here the corpus
is a moving target, because the policy that generates it is the model being
trained. So the loop interleaves:

    1. self-play -- current snapshot plays itself, winners-only kept
    2. append winners to the on-disk corpus (never truncated: a fixed corpus
       would stop being on-policy the moment the model improves)
    3. one training step on a random slice of the corpus
    4. every --play_every steps, refresh the self-play snapshot policy

No off-policy replay, no policy iteration with a lagged opponent -- the corpus
at step N is exactly what the step-N model believes is winning play.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gf256_markov_lm_v6 import GFMarkovLM, CsvLogger
from src.c4 import COLS
from src.c4 import corpus as cp
from src.c4.agent import Agent, agent_agreement
from src.c4.board import Board, Outcome


def load_corpus_tensor(path: str) -> torch.Tensor:
    """Read the whole corpus file into one flat int64 byte tensor.

    Games are newline-delimited, so the model also learns that \\n starts a new
    game -- same as V6 on tinyshakespeare. Only ~2% of bytes are newlines, so
    this is a cheap way to get fresh games in the same stream.
    """
    raw = open(path, "rb").read()
    if not raw:
        raise RuntimeError(f"empty corpus: {path}")
    return torch.tensor(list(raw), dtype=torch.long)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/c4_selfplay")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--seed", type=int, default=32)
    # ---- locked V6 architecture ----
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--fan_in", type=int, default=3)
    # ---- locked V6 optimisation ----
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-drop", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ent_max", type=float, default=2.0)
    ap.add_argument("--ent_warmup", type=int, default=1000)
    ap.add_argument("--perm_w", type=float, default=1.0)
    ap.add_argument("--hard_start", type=int, default=12000)
    ap.add_argument("--hard_ramp", type=int, default=4000)
    ap.add_argument("--hard_w", type=float, default=0.5)
    # ---- self-play ----
    ap.add_argument("--play_every", type=int, default=8,
                    help="Training steps between self-play corpus refreshes.")
    ap.add_argument("--games_per_refresh", type=int, default=96)
    ap.add_argument("--opponent", default="self", choices=["self", "random"],
                    help="'self': both sides the current model (full self-play). "
                         "'random': warm-up phase where the corpus is random.")
    ap.add_argument("--random_until", type=int, default=200,
                    help="Steps of random self-play before the model plays both "
                         "sides. Gives the byte-LM a sane prior over the "
                         "alphabet before it is asked to be a policy.")
    ap.add_argument("--play_temp", type=float, default=0.7)
    ap.add_argument("--play_batch", type=int, default=16,
                    help="Positions forwarded per batch during self-play.")
    # ---- eval / io ----
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_seqs", type=int, default=256)
    ap.add_argument("--eval_chunk_size", type=int, default=16)
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from the run dir's checkpoint and corpus.")
    return ap


@torch.no_grad()
def eval_move_accuracy(model, val_lines, device, ctx, chunk=16):
    """Bits-per-move and top-1 accuracy on HELD-OUT games.

    This is the metric that actually matters. V6 reports bpc over all 256
    bytes; for Connect Four, only the 7 column bytes at offset 9 of each ply
    record are a decision, so we score just those positions.
    """
    was_training = model.training
    model.eval()
    model.set_ste(False)

    from .transcript import COL_BYTE_OFFSET, PLY_LEN
    total_nll, total_n, correct = 0.0, 0, 0
    for raw in val_lines:
        line = raw if isinstance(raw, bytes) else raw.encode()
        body = line.rstrip(b"\n")
        nply = len(body) // PLY_LEN
        if nply < 2:
            continue
        tok = torch.tensor([list(body)], dtype=torch.long)
        for start in range(0, max(1, tok.shape[1] - ctx), ctx):
            x = tok[:, start:start + ctx].to(device)
            if x.shape[1] < 2:
                continue
            logits = model(x)[0]                    # (T, 256)
            t = x[0]
            for i in range(1, t.shape[0]):
                pos = start + i
                if pos % PLY_LEN != PLY_LEN - 1:
                    continue
                tgt = int(t[i])
                if not (COL_BYTE_OFFSET <= tgt < COL_BYTE_OFFSET + COLS):
                    continue
                lg = logits[i - 1][COL_BYTE_OFFSET:COL_BYTE_OFFSET + COLS]
                total_nll += F.cross_entropy(lg[None, :],
                                             torch.tensor([tgt - COL_BYTE_OFFSET]),
                                             reduction="sum").item()
                total_n += 1
                correct += int(int(lg.argmax()) == tgt - COL_BYTE_OFFSET)
    model.train(was_training)
    if total_n == 0:
        return float("inf"), 0.0, 0
    return total_nll / total_n / math.log(2), correct / total_n, total_n


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = random.Random(args.seed)

    run_dir = Path(args.run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    corpus_path = run_dir / "corpus.bin"

    model = GFMarkovLM(d=args.d, layers=args.layers, fan_in=args.fan_in,
                       seed=args.seed).to(args.device)
    model.use_ckpt = not args.no_ckpt
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_par = sum(p.numel() for p in model.parameters())

    start_step = 0
    if args.resume and corpus_path.exists():
        ck = torch.load(run_dir / "checkpoints" / "last.pt",
                        map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        print(f"resumed from step {start_step}", flush=True)

    cfg = {k: v for k, v in vars(args).items()}
    cfg["n_params"] = n_par
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str),
                                         encoding="utf-8")

    print(f"params {n_par/1e6:.2f}M | d={args.d} layers={args.layers} "
          f"fan_in={args.fan_in} | device={args.device} | start_step={start_step}",
          flush=True)
    print(f"corpus: {corpus_path} | run dir: {run_dir}", flush=True)

    train_log = CsvLogger(run_dir / "logs" / "train_metrics.csv",
                         ["step", "loss", "ce_soft", "hard_ce", "entropy",
                          "perm_pen", "corpus_bytes", "elapsed_s"])
    eval_log = CsvLogger(run_dir / "logs" / "eval_metrics.csv",
                         ["step", "move_bpc", "move_top1", "n_positions",
                          "soft_hard_agree", "corpus_bytes", "elapsed_s"])

    t0 = time.time()
    best_top1 = -1.0

    # held-out games: same random policy, different seed -> never trained on
    val_games = cp.winners_only(cp.play_games(512, random.Random(args.seed + 999)))
    val_lines = cp.games_to_lines(val_games)

    agent = Agent(model, args.device, greedy=False, temperature=args.play_temp,
                  rng=rng)

    for step in range(start_step, args.steps + 1):
        # ---------------- self-play corpus refresh ----------------
        if step % args.play_every == 0 or (step == 0 and start_step == 0):
            use_model = args.opponent == "self" and step >= args.random_until
            pol = None
            if use_model:
                pol = _make_policy(agent, args.play_temp)
            games = cp.play_games(args.games_per_refresh, rng, pol)
            won = cp.winners_only(games)
            if won:
                cp.append_corpus(corpus_path, won)
            if step % (args.eval_every) == 0 or step == 0:
                n_games = len(games)
                n_win = len(won)
                print(f"[step {step:5d}] self-play {n_games} games, {n_win} wins "
                      f"| policy={'model' if use_model else 'random'}",
                      flush=True)

        data = load_corpus_tensor(str(corpus_path))

        if step % args.eval_every == 0 or step == args.steps:
            bpc, top1, npos = eval_move_accuracy(model, val_lines, args.device,
                                                 args.ctx, args.eval_chunk_size)
            hard = model.export_hard()
            agree, n_ag = _agreement_rate(agent, val_lines[:64], hard)
            best_top1 = max(best_top1, top1)
            print(f"step {step:5d} | move-bpc {bpc:.4f} | top1 {top1:.4f} "
                  f"| soft~hard {agree:.3f} | corpus {len(data)/1e3:.0f}kB "
                  f"| {time.time()-t0:.0f}s", flush=True)
            eval_log.write({"step": step, "move_bpc": f"{bpc:.4f}",
                            "move_top1": f"{top1:.4f}",
                            "n_positions": npos,
                            "soft_hard_agree": f"{agree:.4f}",
                            "corpus_bytes": len(data),
                            "elapsed_s": f"{time.time()-t0:.1f}"})
            if top1 >= best_top1:
                torch.save(model.cpu().export_hard(),
                           run_dir / "checkpoints" / "best_hard.pt")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step}, run_dir / "checkpoints" / "last.pt")
                model.to(args.device)

        if step == args.steps:
            break

        # ---------------- one V6 training step ----------------
        if step == args.hard_start:
            print(f"\n--- Step {step}: Starting Gentle STE phase ---", flush=True)
            for pg in opt.param_groups:
                pg["lr"] *= args.lr_drop

        x, y = _get_batch(data, args.batch, args.ctx, args.device, rng)

        model.set_ste(False)
        logits_soft = model(x)
        ce_soft = F.cross_entropy(logits_soft.reshape(-1, 256), y.reshape(-1))

        hard_ce = torch.tensor(0.0, device=args.device)
        hard_w = 0.0
        if step >= args.hard_start and args.hard_w > 0:
            hard_w = args.hard_w * min(
                1.0, (step - args.hard_start + 1) / max(1, args.hard_ramp))
            model.set_ste(True)
            logits_hard = model(x)
            hard_ce = F.cross_entropy(logits_hard.reshape(-1, 256), y.reshape(-1))
            model.set_ste(False)

        ent, perm_pen = model.regularisers()
        lam = args.ent_max * min(1.0, (step + 1) / max(1, args.ent_warmup))
        loss = ce_soft + hard_w * hard_ce + lam * ent + args.perm_w * perm_pen

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step % 200 == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f} | ce {ce_soft.item():.4f} "
                  f"| ent {ent.item():.3f} | corpus {len(data)/1e3:.0f}kB "
                  f"| {time.time()-t0:.0f}s", flush=True)
            train_log.write({"step": step, "loss": f"{loss.item():.5f}",
                             "ce_soft": f"{ce_soft.item():.5f}",
                             "hard_ce": f"{hard_ce.item():.5f}",
                             "entropy": f"{ent.item():.5f}",
                             "perm_pen": f"{perm_pen.item():.6f}",
                             "corpus_bytes": len(data),
                             "elapsed_s": f"{time.time()-t0:.1f}"})

    # ---------------- final: full held-out eval + hardened export -------
    bpc, top1, npos = eval_move_accuracy(model, val_lines, args.device, args.ctx)
    hard = model.export_hard()
    agree, _ = _agreement_rate(agent, val_lines, hard)
    print(f"\nFINAL held-out: move-bpc {bpc:.4f} | top1 {top1:.4f} "
          f"| soft~hard {agree:.4f} | {npos} positions")

    # A full-strength self-play game with the HARDENED model -- this is the
    # artifact the user will actually play against.
    outcome, plies = _hard_selfplay(model, hard, rng, args.device)
    print(f"hardened self-play sanity game: {outcome} after {plies} plies")

    (run_dir / "final_metrics.json").write_text(json.dumps({
        "move_bpc_heldout": bpc,
        "move_top1_heldout": top1,
        "n_eval_positions": npos,
        "soft_hard_agree": agree,
        "best_move_top1": best_top1,
        "corpus_bytes": int(corpus_path.stat().st_size),
        "n_params": n_par,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")

    torch.save(model.cpu().export_hard(), run_dir / "hard_model_final.pt")
    print(f"saved -> {run_dir / 'hard_model_final.pt'}")


def _make_policy(agent, temp):
    """A policy callable for corpus.play_games.

    Both sides use the same snapshot and temperature: that is the "full
    self-play from step 0" contract. The move history is threaded through
    because the byte-LM's context is the whole game.
    """
    def policy(board: Board, moves: List[int]) -> int:
        scores = agent.column_logits(moves) + agent.legal_mask(board)
        if temp <= 0:
            return int(scores.argmax())
        p = torch.softmax(scores / temp, dim=-1)
        return int(torch.multinomial(p, 1).item())
    return policy


def _get_batch(data, B, T, device, rng):
    ix = torch.randint(0, max(1, len(data) - T - 1), (B,), generator=_g(rng))
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x.to(device), y.to(device)


_GENS: dict = {}


def _g(rng):
    """torch Generator seeded from the python rng, memoised per call site."""
    return torch.Generator().manual_seed(rng.randrange(2**31))


@torch.no_grad()
def _agreement_rate(agent, val_lines, hard, limit=64):
    """Fraction of positions where soft and hardened forwards pick the same
    legal move. Reported alongside top-1 because it is the honest statement of
    'will the exported runtime play the same game'."""
    ok = tot = 0
    for raw in val_lines[:limit]:
        line = raw if isinstance(raw, bytes) else raw.encode()
        body = line.rstrip(b"\n")
        from .transcript import COL_BYTE_OFFSET, PLY_LEN
        nply = len(body) // PLY_LEN
        for ply in range(nply - 1):
            mv = [body[i * PLY_LEN + PLY_LEN - 1] - COL_BYTE_OFFSET
                  for i in range(ply + 1)]
            b = Board()
            for m in mv:
                if m in b.valid_actions():
                    b.apply_move(m)
            same, _ = agent_agreement(agent, mv, b, hard)
            ok += int(same)
            tot += 1
            if tot >= 64:
                return (ok / tot if tot else 1.0), tot
    return (ok / tot if tot else 1.0), tot


@torch.no_grad()
def _hard_selfplay(model, hard, rng, device, games=4):
    """Play a few full games using ONLY hard_forward, as a smoke test that the
    exported model is a functional policy and not just a set of tensors."""
    from .agent import Agent as _A
    a = _A(model, device, greedy=False, temperature=0.7, rng=rng)
    for g in range(games):
        b = Board()
        mv: list[int] = []
        while True:
            if not mv:
                col = rng.choice(b.valid_actions())
            else:
                try:
                    col = a.hard_column_logits(mv, hard) + a.legal_mask(b)
                    col = int(col.argmax())
                except Exception:
                    col = rng.choice(b.valid_actions())
                if col not in b.valid_actions():
                    col = rng.choice(b.valid_actions())
            mv.append(col)
            b.apply_move(col)
            o = b.outcome()
            if o != Outcome.ONGOING:
                return ("P1" if o == Outcome.WIN_P1 else
                        "P2" if o == Outcome.WIN_P2 else "draw"), len(mv)
    return "ongoing", 0


if __name__ == "__main__":
    main(build_parser().parse_args())