"""Diagnostics for: (a) decision-vs-next distribution, (b) repeat-column bias,
(c) tactical pattern ground truth. Read-only."""
import sys, json, collections, torch
ROOT = r"C:/Users/Chris4K/projects/ZeroFLops-paper"
sys.path.insert(0, ROOT)
from src.c4.ui_server import _load_agent
from src.c4.board import Board, Outcome

CK = "runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt"
agent = _load_agent(CK, "cpu")

def hard_probs(board, moves):
    lg = agent.hard_column_logits(board, moves)
    legal = torch.tensor([c < 6 for c in board.heights], dtype=torch.bool)
    NEG = torch.tensor(-1e9, dtype=lg.dtype)
    p = torch.softmax(torch.where(legal, lg, NEG), -1)
    return p, int(torch.argmax(torch.where(legal, lg, NEG)).item())

def L(p):
    return " ".join("%.1f" % (100*float(p[i])) for i in range(7))

print("=" * 74)
print("A) YOUR EXACT SCENARIO: you play F(5), model plays G(6)")
print("=" * 74)
b = Board(); moves = []
b.apply_move(5); moves.append(5)
dec_p, dec_arg = hard_probs(b, moves)
print("DECISION board (after you play F)   :", L(dec_p))
print("  argmax = %s  <-- THIS is what the model actually chose from" % chr(65+dec_arg))
b.apply_move(6); moves.append(6)
post_p, post_arg = hard_probs(b, moves)
print("POST board (after model plays G)     :", L(post_p))
print("  argmax = %s  <-- THIS is what the panel showed as 'picked'" % chr(65+post_arg))
print()
print("At the DECISION position, F was %.1f%% and G was %.1f%%." % (
    100*float(dec_p[5]), 100*float(dec_p[6])))
print("The model played G because G was the argmax THERE. The 26.1%% for F")
print("in the panel was measured on the board AFTER G was already played.")
print()
print("=" * 74)
print("B) GREEDY 1-PLY BLINDNESS: how often does the model's next-preference")
print("   differ from what it just played?")
print("=" * 74)
same = diff = 0
gaps = []
for seed in range(40):
    b = Board(); moves = []
    while b.outcome() == Outcome.ONGOING and len(moves) < 20:
        legal = b.valid_actions()
        um = legal[(seed + len(moves)) % len(legal)]
        b.apply_move(um); moves.append(um)
        if b.outcome() != Outcome.ONGOING:
            break
        dec_p, dec_arg = hard_probs(b, moves)
        ai = dec_arg
        b.apply_move(ai); moves.append(ai)
        post_p, post_arg = hard_probs(b, moves)
        if post_arg == ai:
            same += 1
        else:
            diff += 1
            # how much did the model give up by not taking its own next-best?
            gaps.append(float(post_p[post_arg]) - float(post_p[ai]))
print("model's next-move argmax == move just played : %d" % same)
print("model's next-move argmax != move just played : %d" % diff)
if gaps:
    gaps.sort()
    print("mean confidence given up by not playing next-best: %.1f pts" % (sum(gaps)/len(gaps)))
    print("median: %.1f  max: %.1f" % (gaps[len(gaps)//2], gaps[-1]))
print()
print("=" * 74)
print("C) REPEAT-COLUMN BIAS: distribution of the model's chosen column")
print("=" * 74)
cnt = collections.Counter(); pos_cnt = collections.Counter()
n = 0
for seed in range(60):
    b = Board(); moves = []
    while b.outcome() == Outcome.ONGOING and len(moves) < 24:
        legal = b.valid_actions()
        um = legal[(seed * 7 + len(moves)) % len(legal)]
        b.apply_move(um); moves.append(um)
        if b.outcome() != Outcome.ONGOING:
            break
        p, ai = hard_probs(b, moves)
        cnt[ai] += 1; n += 1
        # is it the same column as the model's previous move?
        pos_cnt[(len(moves)//2)] += 1
        b.apply_move(ai); moves.append(ai)
print("model's chosen column over %d decisions:" % n)
for c in range(7):
    print("  %s : %4d  (%5.1f%%)" % (chr(65+c), cnt[c], 100*cnt[c]/max(1,n)))
print()
print("=" * 74)
print("D) REPEAT: does it replay the SAME column consecutively?")
print("=" * 74)
rep = tot = 0
for seed in range(60):
    b = Board(); moves = []; prev_ai = None
    while b.outcome() == Outcome.ONGOING and len(moves) < 24:
        legal = b.valid_actions()
        um = legal[(seed * 7 + len(moves)) % len(legal)]
        b.apply_move(um); moves.append(um)
        if b.outcome() != Outcome.ONGOING:
            break
        p, ai = hard_probs(b, moves)
        if prev_ai is not None:
            tot += 1
            if ai == prev_ai:
                rep += 1
        prev_ai = ai
        b.apply_move(ai); moves.append(ai)
print("consecutive same-column model moves: %d / %d  (%.1f%%)" % (rep, tot, 100*rep/max(1,tot)))
print("uniform baseline would be ~16.7%%")
