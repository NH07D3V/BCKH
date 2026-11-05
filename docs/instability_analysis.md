# Instability analysis (long pilots)

Sinh tá»± Ä‘á»™ng bá»Ÿi `code/analyze_instability.py`. NhÃ£n theo `docs/gate2b_protocol.md`:
spike khi robust z â‰¥ 4 vÃ  giÃ¡ trá»‹ â‰¥ 2Ã— median warm-up; module báº¥t á»•n Ä‘á»‹nh khi cÃ³ â‰¥ 3 record spike.

## Tá»«ng run

| Run | Records | Modules | Valid | Unstable | Prevalence | Max z | Loss spikes |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen_seed13 | 100 | 168 | 168 | 19 | 11.3% | 37.3 | 1/100 |
| qwen_seed37 | 100 | 168 | 168 | 19 | 11.3% | 51.0 | 1/100 |
| qwen_seed73 | 100 | 168 | 168 | 18 | 10.7% | 37.1 | 1/100 |
| sailor2_seed13 | 100 | 336 | 336 | 63 | 18.8% | 73.2 | 0/100 |
| sailor2_seed37 | 100 | 336 | 336 | 80 | 23.8% | 63.3 | 0/100 |
| sailor2_seed73 | 100 | 336 | 336 | 74 | 22.0% | 123.8 | 0/100 |

## Qwen2.5-0.5B-Instruct

- Prevalence theo seed: 13: 11.3%, 37: 11.3%, 73: 10.7%.
- Äá»“ng thuáº­n táº­p báº¥t á»•n Ä‘á»‹nh giá»¯a seed (Jaccard): 13_37: 0.73 (16 module chung), 13_73: 0.76 (16 module chung), 37_73: 0.54 (13 module chung).

| Kind | Modules | Unstable | Rate |
|---|---:|---:|---:|
| gate_proj | 24 | 17 | 70.8% |
| q_proj | 24 | 3 | 12.5% |
| up_proj | 24 | 2 | 8.3% |
| k_proj | 24 | 2 | 8.3% |
| down_proj | 24 | 0 | 0.0% |
| o_proj | 24 | 0 | 0.0% |
| v_proj | 24 | 0 | 0.0% |

| Layers | Modules | Unstable | Rate |
|---|---:|---:|---:|
| 0â€“5 | 42 | 9 | 21.4% |
| 6â€“11 | 42 | 7 | 16.7% |
| 12â€“17 | 42 | 5 | 11.9% |
| 18â€“23 | 42 | 3 | 7.1% |

## Sailor2-1B-Chat

- Prevalence theo seed: 13: 18.8%, 37: 23.8%, 73: 22.0%.
- Äá»“ng thuáº­n táº­p báº¥t á»•n Ä‘á»‹nh giá»¯a seed (Jaccard): 13_37: 0.59 (53 module chung), 13_73: 0.61 (52 module chung), 37_73: 0.71 (64 module chung).

| Kind | Modules | Unstable | Rate |
|---|---:|---:|---:|
| gate_proj | 48 | 46 | 95.8% |
| down_proj | 48 | 23 | 47.9% |
| k_proj | 48 | 15 | 31.2% |
| q_proj | 48 | 7 | 14.6% |
| up_proj | 48 | 6 | 12.5% |
| o_proj | 48 | 0 | 0.0% |
| v_proj | 48 | 0 | 0.0% |

| Layers | Modules | Unstable | Rate |
|---|---:|---:|---:|
| 0â€“11 | 84 | 32 | 38.1% |
| 12â€“23 | 84 | 36 | 42.9% |
| 24â€“35 | 84 | 18 | 21.4% |
| 36â€“47 | 84 | 11 | 13.1% |
