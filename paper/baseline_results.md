# Baseline summary

Generated from run_metrics and dev metric artifacts.

| Model | Run | Seed | EM | F1 | Train loss | Runtime (s) | Peak VRAM (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Sailor2-1B-Chat | sailor2_seed13 | 13 | 0.3343 | 0.4257 | 1.2607370149135406 | 6976.6354531 | 5.58 |
| Sailor2-1B-Chat | sailor2_seed37 | 37 | 0.3416 | 0.4388 | 1.3092293146003666 | 6549.096584199999 | 5.58 |
| Sailor2-1B-Chat | sailor2_seed73 | 73 | 0.3402 | 0.4404 | 1.3095538331288286 | 8608.3979378 | 5.58 |
| Qwen2.5-0.5B-Instruct | qwen_seed13 | 13 | 0.3193 | 0.3722 | 1.9950419430334383 | 3463.3698398000015 | 4.67 |
| Qwen2.5-0.5B-Instruct | qwen_seed37 | 37 | 0.3217 | 0.3701 | 1.994553147801052 | 3627.910824399998 | 4.67 |
| Qwen2.5-0.5B-Instruct | qwen_seed73 | 73 | 0.3196 | 0.3700 | 1.9953163232356257 | 3651.6523646999995 | 4.67 |

*Sailor2 seed 13 was resumed from checkpoint-250: runtime and train loss cover steps 250-2,837 only. EM/F1 and final eval loss are computed on the full grouped dev.*
