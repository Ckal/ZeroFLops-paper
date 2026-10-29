"""Measure the user's real scenario: BOTH sides pick the same column.

The user reported: "me and the model picked column 7 only ... then it was
draw there were many open fields."

Three questions, measured separately:

  A. ECHO RATE -- how often does the model play the SAME column the
     opponent just played? A model that "follows" the opponent up a
     column wastes the column and can never build a horizontal four.
     Compared against the centre bot and random for reference.

  B. WINNABILITY -- if the opponent always repeats column X, does the
     model still win? Compared against a strong reference policy in the
     SAME game. If the strong reference wins and the model draws, the
     draw is the model's fault, not the opponent's.

  C. BOARD FILL AT DRAW -- "many open fields" is quantified as the
     fraction of the 42 cells that were filled when the game ended.

Everything uses the SAME canonical hard path as the live server.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys

sys.path.insert(0, ".")

import torch

from src.c4.board import Board, Outcome, P1, P2
from src.c4.ui_server import _load_agent


def hard_row(agent, board: Board, moves: list[int]):
    legal = torch.tensor([c < 6 for c in board.heights], dtype=torch.bool)
    lg = agent.hard_column_logits(board, list(moves))
    neg = torch.tensor(-1e9, dtype=lg.dtype)
    masked = torch.where(legal, lg, neg)
    probs = torch.softmax(masked, dim=-1)
    return probs.detach().numpy().tolist(), [bool(x) for x in legal.tolist()]


# ---------------------------------------------------------------- opponents
def opp_repeat_col(x: int):
    """Always play column x (the user's 'we both picked column 7')."""
    def f(board, moves, rng):
        legal = board.valid_actions()
        if x in legal:
            return x
        return legal[0]
    return f


def opp_centre(board, moves, rng):
    legal = board.valid_actions()
    for c in (3, 2, 4, 1, 5, 0, 6):
        if c in legal:
            return c
    return legal[0]


def opp_random(board, moves, rng):
    return rng.choice(board.valid_actions())


def opp_echo(board, moves, rng):
    """Always play the column the opponent just played (if legal)."""
    legal = board.valid_actions()
    if moves:
        last = moves[-1]
        if last in legal:
            return last
    return legal[0]


# ------------------------------------------------------------------ strong
def strong_ref(agent, board, moves):
    """A stronger reference policy: the model's hard row, but with the
    exact solver consulted when it is cheap enough to prove a win."""
    from src.c4.exact_solver import ExactSolver
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = ExactSolver(max_empty=8, budget=300_000)
    # Try the exact solver first on shallow positions.
    try:
        status = _SOLVER.solve(board, board.turn)
        if status is not None:
            best = None
            best_val = None
            for c in board.valid_actions():
                import copy
                b2 = copy.deepcopy(board)
                b2.apply_move(c)
                if b2.outcome() != Outcome.ONGOING:
                    val = 1
                else:
                    v2 = _SOLVER.solve(b2, 3 - board.turn)
                    if v2 is None:
                        continue
                    val = -v2
                if best_val is None or val > best_val:
                    best_val, best = val, c
            if best is not None:
                return best
    except Exception:
        pass
    return agent.choose(board, list(moves), use_hard=True)


_SOLVER = None


def make_strong(agent):
    """Adapt strong_ref to the (board, moves) policy signature."""
    def pol(board, moves):
        return strong_ref(agent, board, moves)
    return pol


def play_game(agent, model_policy, opp_policy, rng):
    """model_policy(board, moves) -> col. Returns a summary dict."""
    board = Board()
    moves: list[int] = []
    model_moves: list[int] = []
    opp_moves: list[int] = []
    echo_count = 0
    echo_possible = 0
    while board.outcome() == Outcome.ONGOING and len(moves) < 42:
        if board.turn == P2:
            c = model_policy(board, moves)
            model_moves.append(c)
            if moves:
                echo_possible += 1
                if moves[-1] == c:
                    echo_count += 1
            board.apply_move(c)
            moves.append(c)
        else:
            c = opp_policy(board, moves, rng)
            opp_moves.append(c)
            board.apply_move(c)
            moves.append(c)
    o = board.outcome()
    filled = bin(board.mask).count("1")
    return {
        "outcome": o,
        "winner": None if o == Outcome.ONGOING else (P1 if o == Outcome.WIN_P1 else P2),
        "filled": filled,
        "total": 42,
        "echo_count": echo_count,
        "echo_possible": echo_possible,
        "model_moves": model_moves,
        "opp_moves": opp_moves,
        "moves": moves,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default="runs/_probe/echo_col_diag.json")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    agent = _load_agent(args.model, device="cpu")

    def model_greedy(board, moves):
        return agent.choose(board, list(moves), use_hard=True)

    results = {}

    # ---- A. ECHO RATE -------------------------------------------------
    print("=" * 68)
    print("A. ECHO RATE  (model plays the same column the opponent just played)")
    print("=" * 68)
    echo_rows = []
    for name, opp in (("random", opp_random),
                      ("centre", opp_centre),
                      ("echo", opp_echo),
                      ("repeat c6 (col 7)", opp_repeat_col(6))):
        ech = ech_pos = 0
        n = args.games
        for _ in range(n):
            r = play_game(agent, model_greedy, opp, rng)
            ech += r["echo_count"]
            ech_pos += r["echo_possible"]
        rate = 100.0 * ech / max(1, ech_pos)
        echo_rows.append({"opponent": name, "echo_rate_pct": round(rate, 2),
                          "echoes": ech, "opportunities": ech_pos})
        print(f"  vs {name:20s}  echo rate {rate:5.2f}%   "
              f"({ech}/{ech_pos})")

    # Random reference: if the opponent plays a random legal column, the
    # chance the model echoes it is ~1/7 on average (fewer as columns fill).
    print(f"  (chance level for a random opponent ~ 1/7 = 14.3%)")

    # ---- B. WINNABILITY ------------------------------------------------
    print()
    print("=" * 68)
    print("B. WINNABILITY  (opponent always repeats one column)")
    print("=" * 68)
    win_rows = []
    strong = make_strong(agent)
    for x in (6, 3, 0):
        for pol_name, pol in (("model greedy", model_greedy),
                              ("strong reference", strong)):
            wins = draws = losses = 0
            fills = []
            for _ in range(max(40, args.games // 2)):
                r = play_game(agent, pol, opp_repeat_col(x), rng)
                if r["winner"] == P2:
                    wins += 1
                elif r["winner"] is None:
                    draws += 1
                else:
                    losses += 1
                fills.append(r["filled"])
            n = wins + draws + losses
            avg_fill = sum(fills) / max(1, len(fills))
            win_rows.append({"opp_col": x, "policy": pol_name, "wins": wins,
                             "draws": draws, "losses": losses,
                             "win_pct": round(100.0 * wins / max(1, n), 1),
                             "avg_filled_cells": round(avg_fill, 1),
                             "avg_fill_pct": round(100.0 * avg_fill / 42, 1)})
            print(f"  opponent repeats c{x} (col {x+1}) | {pol_name:17s} "
                  f"W{wins:3d} D{draws:3d} L{losses:3d}  "
                  f"win {100.0*wins/max(1,n):5.1f}%  "
                  f"board {avg_fill:4.1f}/42 filled")

    # ---- C. BOARD FILL AT DRAW ----------------------------------------
    print()
    print("=" * 68)
    print("C. BOARD FILL  (how 'open' was the board when the game ended)")
    print("=" * 68)
    fill_rows = []
    for name, opp in (("random", opp_random), ("centre", opp_centre),
                      ("repeat c6 (col 7)", opp_repeat_col(6))):
        fills = []
        outs = collections.Counter()
        for _ in range(max(40, args.games // 2)):
            r = play_game(agent, model_greedy, opp, rng)
            fills.append(r["filled"])
            outs["draw" if r["winner"] is None else
                 ("model_win" if r["winner"] == P2 else "opp_win")] += 1
        avg = sum(fills) / max(1, len(fills))
        fill_rows.append({"opponent": name,
                          "avg_filled": round(avg, 1),
                          "avg_fill_pct": round(100.0 * avg / 42, 1),
                          "outcomes": dict(outs)})
        print(f"  vs {name:20s}  avg {avg:4.1f}/42 cells filled "
              f"({100.0*avg/42:4.1f}%)   {dict(outs)}")

    # ---- D. MODEL'S OWN COLUMN PROFILE --------------------------------
    # The user's report was "me and the model picked column 7 only". That
    # has TWO possible causes, and they are very different:
    #   (i)  the MODEL has a column-7 fixation (a model defect), or
    #   (ii) the model merely FOLLOWS the opponent up the column it just
    #        played, so it looks like a c7 fixation whenever the human
    #        plays c7 (an interaction effect, not a model defect).
    # Measuring the model's column distribution against a NON-echo
    # opponent separates the two: if the model only plays c7 when the
    # opponent does, the fixation is an interaction, not a bias.
    print()
    print("=" * 68)
    print("D. MODEL'S OWN COLUMN PROFILE  (is c7 a bias or an echo?)")
    print("=" * 68)
    col_rows = []
    for name, opp in (("random", opp_random),
                      ("centre", opp_centre),
                      ("repeat c6 (col 7)", opp_repeat_col(6))):
        counts = collections.Counter()
        n_model_moves = 0
        for _ in range(max(40, args.games // 2)):
            r = play_game(agent, model_greedy, opp, rng)
            for c in r["model_moves"]:
                counts[c] += 1
                n_model_moves += 1
        share = {c: round(100.0 * counts[c] / max(1, n_model_moves), 1)
                 for c in range(7)}
        col_rows.append({"opponent": name, "model_moves": n_model_moves,
                         "col_share_pct": share})
        dist = "  ".join(f"c{c}={share[c]:5.1f}%" for c in range(7))
        print(f"  vs {name:20s}  {dist}")
    print("  (if c7 is only high for 'repeat c6', the model FOLLOWS the")
    print("   opponent rather than having a c7 bias of its own)")

    out = {"model": args.model, "games": args.games,
           "echo": echo_rows, "winnability": win_rows, "fill": fill_rows,
           "model_column_profile": col_rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
