import psutil
for p in psutil.process_iter(['pid', 'cmdline']):
    try:
        cl = p.cmdline() or []
        if 'src.c4.train' in ' '.join(cl) and '-c' not in cl[:2]:
            print('TRAINER', p.pid, ' '.join(cl)[:120])
    except Exception:
        pass
