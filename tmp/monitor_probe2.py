import os, time, datetime, re

def show(label, *parts):
    print(label, *parts)

print("=== ALL PYTHON PROCESSES (cmdline) ===")
try:
    import psutil
    any_py = False
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cl = p.cmdline() or []
            name = (p.name() or '').lower()
            if 'python' in name or 'uv' in name:
                any_py = True
                joined = ' '.join(cl)
                tag = ""
                if 'src.c4.train' in joined: tag = "  <== TRAINER"
                print(f"pid={p.pid} name={p.name()} cmd={joined[:200]}{tag}")
        except Exception:
            pass
    if not any_py:
        print("  no python processes at all")
except ImportError as e:
    print("NO_PSUTIL", e)

print()
print("=== LOG TAIL (last 15 lines) ===")
log = "runs/_probe/v24.log"
if os.path.exists(log):
    mtime = os.path.getmtime(log)
    print(f"LOG mtime: {datetime.datetime.fromtimestamp(mtime).isoformat()}")
    print(f"NOW:       {datetime.datetime.fromtimestamp(time.time()).isoformat()}")
    with open(log, 'r', errors='replace') as f:
        lines = f.readlines()
    print(f"TOTAL_LOG_LINES={len(lines)}")
    for ln in lines[-15:]:
        print(ln.rstrip())
else:
    print("LOG MISSING")

print()
print("=== COMPLETION ARTIFACTS ===")
arts = [
    "runs/_probe/v24/final_float.pt",
    "runs/_probe/v24/hard_model_final.pt",
    "runs/_probe/v24/checkpoints/best_hard.pt",
]
for a in arts:
    if os.path.exists(a):
        sz = os.path.getsize(a)
        print(f"PRESENT {a}  size={sz}  ({sz/1e6:.1f} MB)")
    else:
        print(f"MISSING {a}")

print()
print("=== DIR runs/_probe/v24 (all files w/ sizes) ===")
d = "runs/_probe/v24"
if os.path.isdir(d):
    for root, dirs, files in os.walk(d):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                sz = os.path.getsize(fp)
                print(f"  {fp}  {sz/1e6:.2f} MB")
            except Exception:
                pass
else:
    print("DIR MISSING", d)
