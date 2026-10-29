"""Diagnostic for the three reported UI/model bugs. Read-only: writes nothing."""
import sys, torch, json
ROOT = r"C:/Users/Chris4K/projects/ZeroFLops-paper"
sys.path.insert(0, ROOT)
from src.c4.ui_server import _load_agent
from src.c4.board import Board, Outcome

CK = "runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt"
agent = _load_agent(CK, "cpu")

def masked_hard(board, moves):
    lg = agent.hard_column_logits(board, moves)
    legal = torch.tensor([c < 6 for c in board.heights], dtype=torch.bool)
    NEG = torch.tensor(-1e9, dtype=lg.dtype)
    return torch.where(legal, lg, NEG)

print("=" * 70)
print("A) IS THE FLOAT PATH TRAINED ON THIS V6 CHECKPOINT?")
print("=" * 70)
b = Board()
fl = agent.column_logits(b, [])
print("float raw logits :", [round(float(x), 8) for x in fl])
print("float uniform?   :", bool(torch.allclose(fl, fl[0].expand_as(fl), atol=1e-7)))
print("=> soft argmax is ALWAYS column A (index of max in a flat array)")
print("has import_hard? :", hasattr(agent.model, "import_hard"))
print("model class      :", type(agent.model).__name__)

print()
print("=" * 70)
print("B) DOES THE MODEL PLAY THE ARGMAX OF THE POSITION IT SAW?")
print("=" * 70)
b = Board(); moves = []
rows = []
for i, umove in enumerate([0, 1, 2, 3, 4, 5]):
    b.apply_move(umove); moves.append(umove)
    if b.outcome() != Outcome.ONGOING:
        break
    pre = masked_hard(b, moves)                 # the decision position
    pre_arg = int(torch.argmax(pre).item())
    pre_p = torch.softmax(pre, -1)
    ai = int(torch.argmax(pre).item())
    b.apply_move(ai); moves.append(ai)          # model actually plays this
    post = masked_hard(b, moves)                # the position the PANEL sees
    post_arg = int(torch.argmax(post).item())
    post_p = torch.softmax(post, -1)
    rows.append((umove, ai, pre_arg, post_arg))
    print("user=%s | model PLAYED=%s (decision argmax=%s) | PANEL argmax=%s | "
          "played==panel? %s | panel shows %.1f%% for the played col"%(
              chr(65+umove), chr(65+ai), chr(65+pre_arg), chr(65+post_arg),
              ai == post_arg, 100*float(post_p[ai])))
print()
print("model honours its own decision-position argmax:",
      all(ai == pre_arg for _, ai, pre_arg, _ in rows))
print("panel argmax == move actually played:",
      sum(1 for _, ai, _, post_arg in rows if ai == post_arg), "/", len(rows))

print()
print("=" * 70)
print("C) WHAT DOES THE PANEL SAY vs WHAT IT SHOWS AS 'picked'?")
print("=" * 70)
b = Board(); moves = []
b.apply_move(5); moves.append(5)   # user F
pre = masked_hard(b, moves)
ai = int(torch.argmax(pre).item())
b.apply_move(ai); moves.append(ai)
post = masked_hard(b, moves)
post_p = torch.softmax(post, -1)
post_arg = int(torch.argmax(post).item())
print("user picked F, model actually played %s" % chr(65+ai))
print("panel 'picked' (argmax of post-AI board) = %s" % chr(65+post_arg))
print("panel would report: 'Model picked column %s · %.1f%%'" % (
    chr(65+post_arg), 100*float(post_p[post_arg])))
print("ACTUALLY played = column %s (%.1f%% under the post board)" % (
    chr(65+ai), 100*float(post_p[ai])))
print("=> the headline column is the model's NEXT preference, not the played move.")

print()
print("=" * 70)
print("D) WHAT DOES THE TRAINER ACTUALLY OPTIMISE?")
print("=" * 70)
import re
src = open(ROOT + "/src/c4/train.py", encoding="utf-8").read()
for pat in ["hard_forward", "column_logits", "import_hard", "export_hard",
            "loss", "cross_entropy"]:
    hits = [i for i, l in enumerate(src.splitlines(), 1) if pat in l]
    print("%-16s -> lines %s" % (pat, hits[:14]))
