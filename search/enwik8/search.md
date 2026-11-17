# Settings search (enwik8)

36 experiments, 11.5 hours. Each: 20,000 steps, scored on the validation windows of the trainer. Lower is better.

- start: 1.2379  (two seeds: 1.2379, 1.2403; chance wobble 0.0030)
- best:  1.2000  (+0.0379 better)

Best settings: arch classic -> modern, cache_orders 2,3,4,6,8,12,16 -> 2,3,4,5,6,8,10,12,16,24, lr 0.00045 -> 0.0009, mem_drop 0.0 -> 0.05

For a full run, use these options (the warmup scales with the steps):

```
--d 512 --layers 8 --heads 8 --lr 0.0009 --warmup 2000 --min_lr_frac 0.1 --weight_decay 0.1 --dropout 0.2 --arch modern --doc_hist 8192 --cache_keep 8 --cache_orders 2,3,4,5,6,8,10,12,16,24 --mem_drop 0.05 --cache_decay 0 --class_cache 0
```

| # | score | kept | change |
|---|---|---|---|
| 1 | 1.2379 | no | start |
| 2 | 1.2403 | no | start |
| 3 | 1.2393 | no | doc_hist 8192 -> 4096 |
| 4 | 1.2367 | no | cache_keep 8 -> 4 |
| 5 | 1.2298 | yes | lr 0.00045 -> 0.0006 |
| 6 | 1.2104 | yes | arch classic -> modern |
| 7 | 1.2068 | yes | cache_orders 2,3,4,6,8,12,16 -> 2,3,4,5,6,8,10,12,16,24 |
| 8 | 1.2074 | no | cache_keep 8 -> 16 |
| 9 | 1.2076 | no | doc_hist 8192 -> 4096 |
| 10 | 1.2041 | no | mem_drop 0.0 -> 0.05 |
| 11 | 1.2065 | no | doc_hist 8192 -> 16384 |
| 12 | 1.2102 | no | min_lr_frac 0.1 -> 0.3 |
| 13 | 1.2183 | no | dropout 0.2 -> 0.3 |
| 14 | 1.2180 | no | weight_decay 0.1 -> 0.3 |
| 15 | 1.2050 | no | cache_keep 8 -> 4 |
| 16 | 1.2068 | no | min_lr_frac 0.1 -> 0.03 |
| 17 | 1.2068 | no | warmup 2000 -> 1000 |
| 18 | 1.2075 | no | weight_decay 0.1 -> 0.03 |
| 19 | 1.2030 | yes | lr 0.0006 -> 0.0009 |
| 20 | 1.2102 | no | min_lr_frac 0.1 -> 0.3 |
| 21 | 1.2117 | no | shape 512x8x8 -> 640x5x10 |
| 22 | 1.2000 | yes | mem_drop 0.0 -> 0.05 |
| 23 | 1.2032 | no | weight_decay 0.1 -> 0.03 |
| 24 | 1.1995 | no | shape 512x8x8 -> 448x10x7 |
| 25 | 1.1971 | no | warmup 2000 -> 1000 |
| 26 | 1.2025 | no | cache_keep 8 -> 16 |
| 27 | 1.2000 | no | min_lr_frac 0.1 -> 0.03 |
| 28 | 1.2034 | no | cache_orders 2,3,4,5,6,8,10,12,16,24 -> 2,3,4,6,8,12,16 |
| 29 | 1.2177 | no | arch modern -> classic |
| 30 | 1.2003 | no | doc_hist 8192 -> 16384 |
| 31 | 1.1984 | no | mem_drop 0.05 -> 0.1 |
| 32 | 1.1989 | no | lr 0.0009 -> 0.0013 |
| 33 | 1.2072 | no | cache_decay 0 -> 500 |
| 34 | 1.1986 | no | cache_decay 0 -> 4000 |
| 35 | 1.2025 | no | class_cache 0 -> 1 |
| 36 | 1.2032 | no | cache_decay 0 -> 1500 |
