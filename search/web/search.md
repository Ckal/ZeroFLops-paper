# Settings search (web)

30 experiments, 7.9 hours. Each: 20,000 steps, scored on the validation windows of the trainer. Lower is better.

- start: 1.3136  (two seeds: 1.3140, 1.3136; chance wobble 0.0030)
- best:  1.2893  (+0.0243 better)

Best settings: arch classic -> modern, cache_keep 8 -> 4, warmup 1000 -> 500

For a full run, use these options (the warmup scales with the steps):

```
--d 512 --layers 8 --heads 8 --lr 0.0006 --warmup 500 --min_lr_frac 0.1 --weight_decay 0.1 --dropout 0.05 --arch modern --doc_hist 8192 --cache_keep 4 --cache_orders 2,3,4,6,8,12,16 --mem_drop 0.0
```

| # | score | kept | change |
|---|---|---|---|
| 1 | 1.3140 | no | start |
| 2 | 1.3136 | no | start |
| 3 | 1.3128 | no | shape 512x8x8 -> 448x10x7 |
| 4 | 1.3197 | no | doc_hist 8192 -> 16384 |
| 5 | 1.3130 | no | lr 0.0006 -> 0.0009 |
| 6 | 1.3166 | no | dropout 0.05 -> 0.1 |
| 7 | 1.3122 | no | mem_drop 0.0 -> 0.05 |
| 8 | 1.3276 | no | weight_decay 0.1 -> 0.3 |
| 9 | 1.3168 | no | cache_orders 2,3,4,6,8,12,16 -> 3,6,12 |
| 10 | 1.3141 | no | min_lr_frac 0.1 -> 0.3 |
| 11 | 1.3183 | no | shape 512x8x8 -> 640x5x10 |
| 12 | 1.3121 | no | dropout 0.05 -> 0.0 |
| 13 | 1.3116 | no | cache_orders 2,3,4,6,8,12,16 -> 2,3,4,5,6,8,10,12,16,24 |
| 14 | 1.3139 | no | min_lr_frac 0.1 -> 0.03 |
| 15 | 1.3143 | no | warmup 1000 -> 2000 |
| 16 | 1.3128 | no | weight_decay 0.1 -> 0.03 |
| 17 | 1.3158 | no | warmup 1000 -> 500 |
| 18 | 1.3146 | no | lr 0.0006 -> 0.00045 |
| 19 | 1.3099 | yes | cache_keep 8 -> 4 |
| 20 | 1.3083 | no | mem_drop 0.0 -> 0.05 |
| 21 | 1.3093 | no | min_lr_frac 0.1 -> 0.03 |
| 22 | 1.2934 | yes | arch classic -> modern |
| 23 | 1.2974 | no | warmup 1000 -> 2000 |
| 24 | 1.2893 | yes | warmup 1000 -> 500 |
| 25 | 1.3016 | no | shape 512x8x8 -> 640x5x10 |
| 26 | 1.2889 | no | cache_orders 2,3,4,6,8,12,16 -> 2,3,4,5,6,8,10,12,16,24 |
| 27 | 1.2904 | no | min_lr_frac 0.1 -> 0.03 |
| 28 | 1.2885 | no | doc_hist 8192 -> 4096 |
| 29 | 1.2936 | no | dropout 0.05 -> 0.1 |
| 30 | 1.2984 | no | cache_orders 2,3,4,6,8,12,16 -> 3,6,12 |
