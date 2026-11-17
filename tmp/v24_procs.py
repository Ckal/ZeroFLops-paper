import psutil
for p in psutil.process_iter(['pid', 'cmdline']):
    try:
        cl = ' '.join(p.cmdline() or [])
        if 'python' in cl.lower() or 'uv' in cl.lower():
            print(p.pid, '|', cl[:160])
    except Exception:
        pass
