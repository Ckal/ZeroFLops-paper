import psutil, json
out = []
for p in sorted(psutil.process_iter(['pid','ppid','create_time','name','cmdline','cpu_times','memory_info'])):
    try:
        i = p.info
        cmd = ' '.join(i['cmdline'][:5]) if i['cmdline'] else i['name']
        out.append({'pid': i['pid'], 'ppid': i['ppid'], 'name': i['name'], 'cmd': cmd[:300],
                    'cpu_user_min': round(i['cpu_times'].user/60, 1),
                    'rss_mb': round(i['memory_info'].rss/1e6, 1) if i['memory_info'] else None,
                    'born': psutil.boot_time() and __import__('datetime').datetime.fromtimestamp(i['create_time']).isoformat()})
    except Exception as e:
        out.append({'pid': getattr(i,'pid',None), 'err': str(e)})
print(json.dumps([o for o in out if o.get('name','').startswith('python') or o['pid'] in (1669,1667,12792,12824) or 'ollama' in str(o.get('cmd',''))], indent=1, default=str))
