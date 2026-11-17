import json, subprocess, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
L = []
def P(s=""):
    L.append(str(s)); print(s)

P("=" * 78); P("1. CHECKPOINT STRUCTURE (where do the lags live?)"); P("=" * 78)
p = Path("runs/c4/v7mixed_v18_seed32/checkpoints/best_hard.pt")
raw = torch.load(p, map_location="cpu", weights_only=False)
P("top-level keys: " + str(sorted(raw.keys())))
for k in sorted(raw.keys()):
    v = raw[k]
    if isinstance(v, dict):
        P("  %-16s dict(%d) sample=%s" % (k, len(v), list(v.keys())[:8]))
    elif hasattr(v, "shape"):
        P("  %-16s tensor %s" % (k, tuple(v.shape)))
    else:
        P("  %-16s %r" % (k, v))
P()
# find lag-bearing tensors in the state dict
sd = raw.get("state") or raw.get("model") or raw.get("model_state") or {}
if isinstance(sd, dict):
    P("state-dict entries mentioning 'lag':")
    for k, v in sd.items():
        if "lag" in k.lower():
            P("   %-30s shape=%s dtype=%s" % (k, tuple(v.shape), v.dtype))
            P("      values: %s" % (v.flatten()[:16].tolist()))
    P()
    P("state-dict entries mentioning 'bias'/'table':")
    for k, v in sd.items():
        if "bias" in k.lower() or "table" in k.lower():
            P("   %-30s shape=%s" % (k, tuple(v.shape)))
P()

P("=" * 78); P("2. OLD final.json WRITER (git, at the time of the 2026-10-23 run)"); P("=" * 78)
def g(*a):
    return subprocess.run(["git"] + list(a), capture_output=True, text=True, cwd=".").stdout
# find the commit that produced the run / the writer before the rename
for rev in ("HEAD", "60e48b4~1", "f20e42b", "60e48b4"):
    txt = g("show", "%s:src/c4/train.py" % rev)
    if not txt:
        continue
    hit = None
    for i, ln in enumerate(txt.splitlines()):
        if "final.json" in ln:
            hit = i; break
    if hit is None:
        P("%-10s no final.json writer" % rev); continue
    P("--- %s ---" % rev)
    for ln in txt.splitlines()[hit - 3: hit + 16]:
        P("   " + ln)
    P()
P("log for the rename commit:")
P(g("show", "--stat", "--oneline", "60e48b4")[:800])
P()

P("=" * 78); P("3. WHICH RUN PRODUCED THE RELEASED step18000 ARTIFACT?"); P("=" * 78)
for d in sorted(Path("runs/c4").iterdir()):
    if not d.is_dir():
        continue
    for f in ("checkpoints/best_hard.pt", "hard_model_final.pt", "final_float.pt"):
        fp = d / f
        if not fp.exists():
            continue
        try:
            r = torch.load(fp, map_location="cpu", weights_only=False)
        except Exception:
            continue
        m = r.get("meta") or {}
        st = m.get("step")
        if st in (18000,):
            P("MATCH %s/%s  step=%s" % (d.name, f, st))
            P("   top-level keys: %s" % sorted(r.keys()))
            P("   meta: %s" % json.dumps({k: str(v)[:120] for k, v in m.items()})[:1200])
P()
P("search all runs for a step-18000 checkpoint:")
found = False
for d in sorted(Path("runs").rglob("*.pt")):
    try:
        r = torch.load(d, map_location="cpu", weights_only=False)
    except Exception:
        continue
    m = r.get("meta") or {}
    if m.get("step") == 18000:
        P("  %s  step=18000  meta_keys=%s" % (d, sorted(m.keys())))
        found = True
if not found:
    P("  none found via meta.step")
P()

P("=" * 78); P("4. RELEASED ARTIFACT: does it embed the lag set?"); P("=" * 78)
for rel in ("release/artifacts/connect_four_gf256_best_hard_step18000.pt",
            "runs/_probe/v24/checkpoints/best_hard.pt"):
    fp = Path(rel)
    if not fp.exists():
        P("%s MISSING" % rel); continue
    r = torch.load(fp, map_location="cpu", weights_only=False)
    P("--- %s ---" % rel)
    P("  top-level keys: %s" % sorted(r.keys()))
    m = r.get("meta") or {}
    P("  meta: %s" % json.dumps({k: str(v)[:200] for k, v in m.items()})[:1500])
    sd = r.get("state") or r.get("model") or {}
    if isinstance(sd, dict):
        for k, v in sd.items():
            if "lag" in k.lower():
                P("  %-34s shape=%s" % (k, tuple(v.shape)))
                P("     %s" % (v.flatten()[:16].tolist()))
    P()

Path("tmp/v18_release_prov.txt").write_text("\n".join(L), encoding="utf-8")
print("\n[written] tmp/v18_release_prov.txt")
