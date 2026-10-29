import torch

for name in [r'runs/_probe/mir_a_s6000/final_float.pt',
             r'runs/_probe/mir_a_s6000/checkpoints/step6000_soft.pt',
             r'runs/_probe/mir_a_s6000/checkpoints/step6000_hard.pt']:
    try:
        ck = torch.load(name, map_location='cpu', weights_only=False)
    except Exception as e:
        print(name, 'LOAD FAIL', e)
        continue
    print('==', name)
    print('  top keys:', list(ck.keys())[:12] if isinstance(ck, dict) else type(ck))
    if isinstance(ck, dict):
        for k in ['state', 'meta', 'model', 'state_dict']:
            if k in ck:
                v = ck[k]
                if isinstance(v, dict):
                    ks = list(v.keys())
                    print(f'   {k} -> {len(ks)} keys; first: {ks[:8]}')
                    if 'transcript' in v:
                        print('      transcript:', v['transcript'])
                    nz = 0
                    tot = 0
                    for kk, vv in v.items():
                        if hasattr(vv, 'numel'):
                            tot += vv.numel()
                            if float(vv.abs().sum()) > 0:
                                nz += 1
                    print('      nonzero tensors:', nz, '/', len(v), 'elems', tot)
                else:
                    print(f'   {k} -> {type(v)}')
