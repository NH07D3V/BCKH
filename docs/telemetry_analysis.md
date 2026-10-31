# Telemetry analysis

Sinh tự động bởi `code/analyze_telemetry.py`; chỉ đọc artifact, không chạy lại training.

## Chất lượng pilot

| Run | Records | Empty | Steps | Modules | |g| mean | |g| max | update/weight mean | Loss spikes |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| qwen_seed13 | 20 | 0 | 10–200 | 168 | 1.4635 | 1.9330 | 2.84e-02 | 0 (0.0%) |
| qwen_seed37 | 20 | 0 | 10–200 | 168 | 1.4211 | 1.8863 | 2.82e-02 | 0 (0.0%) |
| qwen_seed73 | 20 | 0 | 10–200 | 168 | 1.5729 | 2.1817 | 2.83e-02 | 0 (0.0%) |
| sailor2_seed13 | 20 | 0 | 10–200 | 336 | 1.3407 | 2.0228 | 2.72e-02 | 0 (0.0%) |
| sailor2_seed37 | 20 | 0 | 10–200 | 336 | 1.3197 | 1.8998 | 2.68e-02 | 0 (0.0%) |
| sailor2_seed73 | 20 | 0 | 10–200 | 336 | 1.5804 | 4.6490 | 2.72e-02 | 0 (0.0%) |

## Qwen2.5-0.5B-Instruct (seeds 13, 37, 73)

- Cross-seed rank agreement of module gradient norms: 13↔37: ρ=0.986, 13↔73: ρ=0.983, 37↔73: ρ=0.984.
- Modules observed: 168.

| Module | Depth | |g| mean | within-run CV | cross-seed CV | update/weight | risk |
|---|---:|---:|---:|---:|---:|---:|
| `base_model.model.model.layers.20.mlp.gate_proj` | 20 | 0.2508 | 0.346 | 0.087 | 4.18e-02 | 0.913 |
| `base_model.model.model.layers.23.mlp.up_proj` | 23 | 0.3775 | 0.317 | 0.244 | 3.57e-02 | 0.894 |
| `base_model.model.model.layers.14.mlp.gate_proj` | 14 | 0.1244 | 0.389 | 0.092 | 4.03e-02 | 0.882 |
| `base_model.model.model.layers.9.mlp.gate_proj` | 9 | 0.1187 | 0.415 | 0.092 | 3.93e-02 | 0.874 |
| `base_model.model.model.layers.12.mlp.gate_proj` | 12 | 0.1176 | 0.353 | 0.170 | 3.78e-02 | 0.873 |
| `base_model.model.model.layers.6.mlp.gate_proj` | 6 | 0.1813 | 0.380 | 0.068 | 4.01e-02 | 0.867 |
| `base_model.model.model.layers.1.mlp.gate_proj` | 1 | 0.1970 | 0.352 | 0.084 | 3.72e-02 | 0.850 |
| `base_model.model.model.layers.10.mlp.gate_proj` | 10 | 0.1117 | 0.396 | 0.088 | 3.87e-02 | 0.850 |
| `base_model.model.model.layers.23.mlp.gate_proj` | 23 | 0.3136 | 0.281 | 0.057 | 4.02e-02 | 0.838 |
| `base_model.model.model.layers.22.mlp.up_proj` | 22 | 0.2822 | 0.259 | 0.073 | 3.83e-02 | 0.829 |
| `base_model.model.model.layers.18.mlp.gate_proj` | 18 | 0.2089 | 0.396 | 0.046 | 4.33e-02 | 0.822 |
| `base_model.model.model.layers.1.mlp.up_proj` | 1 | 0.2082 | 0.288 | 0.094 | 3.44e-02 | 0.819 |
| `base_model.model.model.layers.2.mlp.gate_proj` | 2 | 0.1583 | 0.343 | 0.083 | 3.68e-02 | 0.819 |
| `base_model.model.model.layers.16.mlp.gate_proj` | 16 | 0.1610 | 0.350 | 0.053 | 4.11e-02 | 0.811 |
| `base_model.model.model.layers.11.mlp.gate_proj` | 11 | 0.1056 | 0.387 | 0.077 | 3.77e-02 | 0.801 |

Module-kind aggregate:

| Kind | Modules | |g| mean | risk mean |
|---|---:|---:|---:|
| down_proj | 24 | 0.0693 | 0.514 |
| gate_proj | 24 | 0.1795 | 0.796 |
| k_proj | 24 | 0.0356 | 0.379 |
| o_proj | 24 | 0.0665 | 0.383 |
| q_proj | 24 | 0.0323 | 0.390 |
| up_proj | 24 | 0.1588 | 0.678 |
| v_proj | 24 | 0.0779 | 0.360 |

Depth buckets:

| Layers | Modules | |g| mean | risk mean |
|---|---:|---:|---:|
| 0–5 | 42 | 0.1009 | 0.482 |
| 6–11 | 42 | 0.0722 | 0.511 |
| 12–17 | 42 | 0.0723 | 0.469 |
| 18–23 | 42 | 0.1090 | 0.538 |

## Sailor2-1B-Chat (seeds 13, 37, 73)

- Cross-seed rank agreement of module gradient norms: 13↔37: ρ=0.986, 13↔73: ρ=0.981, 37↔73: ρ=0.985.
- Modules observed: 336.

| Module | Depth | |g| mean | within-run CV | cross-seed CV | update/weight | risk |
|---|---:|---:|---:|---:|---:|---:|
| `base_model.model.model.layers.28.mlp.gate_proj` | 28 | 0.1343 | 0.455 | 0.177 | 3.77e-02 | 0.928 |
| `base_model.model.model.layers.24.mlp.gate_proj` | 24 | 0.1307 | 0.584 | 0.162 | 3.69e-02 | 0.922 |
| `base_model.model.model.layers.25.mlp.gate_proj` | 25 | 0.0936 | 0.549 | 0.226 | 3.69e-02 | 0.904 |
| `base_model.model.model.layers.35.mlp.gate_proj` | 35 | 0.1275 | 0.438 | 0.146 | 3.91e-02 | 0.900 |
| `base_model.model.model.layers.31.mlp.gate_proj` | 31 | 0.0983 | 0.431 | 0.155 | 3.85e-02 | 0.881 |
| `base_model.model.model.layers.18.mlp.gate_proj` | 18 | 0.1057 | 0.518 | 0.191 | 3.55e-02 | 0.871 |
| `base_model.model.model.layers.16.mlp.gate_proj` | 16 | 0.1160 | 0.550 | 0.146 | 3.58e-02 | 0.870 |
| `base_model.model.model.layers.19.mlp.gate_proj` | 19 | 0.0950 | 0.577 | 0.155 | 3.61e-02 | 0.869 |
| `base_model.model.model.layers.12.mlp.gate_proj` | 12 | 0.1432 | 0.535 | 0.131 | 3.53e-02 | 0.858 |
| `base_model.model.model.layers.21.mlp.up_proj` | 21 | 0.0992 | 0.505 | 0.247 | 3.39e-02 | 0.858 |
| `base_model.model.model.layers.21.mlp.gate_proj` | 21 | 0.0967 | 0.566 | 0.136 | 3.64e-02 | 0.857 |
| `base_model.model.model.layers.23.mlp.gate_proj` | 23 | 0.0844 | 0.542 | 0.206 | 3.53e-02 | 0.852 |
| `base_model.model.model.layers.33.mlp.gate_proj` | 33 | 0.0947 | 0.439 | 0.127 | 3.88e-02 | 0.851 |
| `base_model.model.model.layers.11.mlp.gate_proj` | 11 | 0.0848 | 0.581 | 0.146 | 3.61e-02 | 0.849 |
| `base_model.model.model.layers.14.mlp.gate_proj` | 14 | 0.1339 | 0.508 | 0.125 | 3.55e-02 | 0.843 |

Module-kind aggregate:

| Kind | Modules | |g| mean | risk mean |
|---|---:|---:|---:|
| down_proj | 48 | 0.0530 | 0.546 |
| gate_proj | 48 | 0.1137 | 0.760 |
| k_proj | 48 | 0.0230 | 0.335 |
| o_proj | 48 | 0.0535 | 0.424 |
| q_proj | 48 | 0.0221 | 0.364 |
| up_proj | 48 | 0.1119 | 0.665 |
| v_proj | 48 | 0.0678 | 0.407 |

Depth buckets:

| Layers | Modules | |g| mean | risk mean |
|---|---:|---:|---:|
| 0–11 | 84 | 0.0535 | 0.543 |
| 12–23 | 84 | 0.0654 | 0.592 |
| 24–35 | 84 | 0.0688 | 0.485 |
| 36–47 | 84 | 0.0667 | 0.381 |

`risk_score` is the equal-weight mean of the within-model percentile ranks of module gradient level, within-run gradient volatility, cross-seed gradient spread and update-to-weight ratio. It orders modules for follow-up; it is not a calibrated failure probability.
