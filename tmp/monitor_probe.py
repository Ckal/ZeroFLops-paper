import os, time, re, sys

# 1) Identify trainer PIDs
try:
    import psutil
    trainers = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cl = p.cmdline() or []
            joined = ' '.join(cl)
            if 'src.c4.train' in joined:
                trainers.append((p.pid, joined[:160]))
        except Exception:
            pass
    if trainers:
        print("TRAINER_MATCHES:")
        for pid, cl in trainers:
            print(f"  pid={pid}  cmd={cl}")
    else:
        print("TRAINER_MATCHES: NONE")
except ImportError as e:
    print("NO_PSUTIL:", e)

# 2) Log step counter + mtime
log = "runs/_probe/v24.log"
if os.path.exists(log):
    mtime = os.path.getmtime(log)
    now = time.time()
    age_s = now - mtime
    with open(log, 'r', errors='replace') as f:
        data = f.read()
    steps = re.findall(r'^step (\d+)', data, re.M)
    last_step = steps[-1] if steps else "none"
    print(f"LOG: last_step={last_step} mtime_age_s={age_s:.0f} ({age_s/60:.1f} min)")
    print(f"NOW_epoch={now:.0f}")
    # check for failure markers
    for marker in ["Traceback", "ARM FAILED", "[ABORT]", "elapsed:"]:
        if marker in data:
            idx = data.rfind(marker)
            print(f"MARKER_FOUND: {marker!r} ... {data[idx:idx+200]!r}")
else:
    print("LOG: MISSING", log)

# 3) UI port check
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)
try:
    s.connect(("127.0.0.1", 7860))
    print("UI_7860: CONNECTED")
except Exception as e:
    print("UI_7860: FAILED", repr(e))
finally:
    s.close()
