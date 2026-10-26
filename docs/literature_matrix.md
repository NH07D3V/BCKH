# Literature matrix — QLoRA stability, quantization and adaptive LoRA

Ngày rà soát: 2026-10-26. Bảng này dùng để kiểm soát claim; không xem là exhaustive until backward citation search before submission.

| Paper | Core contribution | Overlap with this project | Gap left for this project |
|---|---|---|---|
| [QLoRA, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html) | 4-bit NF4, double quantization and paged optimizers | Shared QLoRA setting | Does not center multi-seed predictor validity for Vietnamese SFT |
| [QLoRA repeatability study, IEEE Access 2024](https://ieeexplore.ieee.org/document/10700744/) | Documents run-to-run variability across repeated QLoRA fine-tuning | Direct stability motivation | Does not provide our Vietnamese grouped-holdout telemetry protocol |
| [LoRA-GA](https://arxiv.org/abs/2407.05000) | Gradient alignment for LoRA initialization | Gradient alignment overlap | Our candidate question is early trajectory prediction and transfer, not initialization alone |
| [QuAILoRA, PMLR 2024](https://proceedings.mlr.press/v262/g-lawton24a.html) | Quantization-aware LoRA initialization | Quantization discrepancy overlap | Does not establish out-of-sample seed/model stability prediction in our setting |
| [QDyLoRA, EMNLP 2024](https://aclanthology.org/2024.emnlp-industry.53/) | Dynamic rank choices during training | Adaptive rank overlap | Does not test combined residual-gradient risk features as a predictor |
| [QERA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/21718991f6acf19a42376b5c7a8668c5-Abstract-Conference.html) | Output-error/activation-aware quantization | Residual/output geometry overlap | Different objective; stability predictor and Vietnamese SFT remain open candidate |
| [CLoQ](https://arxiv.org/abs/2501.18475) | Calibrated LoRA initialization for quantized LLMs | Quantization error compensation overlap | Calibration transfer and run variance are not the same question |
| [LoRA-One, ICML 2025](https://proceedings.mlr.press/v267/zhang25ax.html) | Singular-subspace alignment and LoRA theory | Alignment/rank overlap | Does not make our matched-budget, multi-seed predictor claim |
| [Q-GaLore, PMLR 2025](https://proceedings.mlr.press/v280/zhang25a.html) | Quantized low-rank gradient adaptation | Quantized gradient overlap | Different optimizer/compression objective |
| [GoRA](https://arxiv.org/abs/2502.12171) | Gradient-driven adaptive rank/initialization | Gradient rank overlap | Generic adaptive allocation is crowded; transfer/failure boundary remains candidate |
| [QR-Adaptor](https://arxiv.org/abs/2505.03802) | Joint per-layer bitwidth/rank allocation through calibration | Rank/bit allocation overlap | Policy cannot be claimed novel without predictor validity and equal overhead |
| [IGU-LoRA](https://arxiv.org/abs/2603.13792) | Uncertainty-aware adaptive rank | Uncertainty/rank overlap | Does not settle our specific QLoRA SFT cross-seed/model audit |
| [GradientStabilizer, ICML 2026](https://proceedings.mlr.press/v306/huang26bb.html) | Stabilizes gradient norm spikes across settings | Gradient stability overlap | QGA must be quantization-specific and measured as predictive validity |
| [When Gradient Importance Lies](https://arxiv.org/abs/2605.07366) | Shows gradient rank allocation can fail under GRPO | Supports failure-boundary framing | Our study tests SFT, quantization and Vietnamese QA with controlled seeds |

## Candidate gap statement

Sau rà soát hiện tại, không được tuyên bố rằng chưa ai kết hợp gradient, quantization hoặc adaptive rank. Gap ứng viên hẹp hơn là: chưa thấy một protocol matched-budget kiểm định out-of-sample xem tổ hợp quantization residual và stochastic gradient geometry có dự đoán được run-to-run variance, loss spike hoặc rank sufficiency của 4-bit QLoRA SFT trên tiếng Việt hay không, đồng thời tính overhead telemetry.

## Search status

- Đã có paper gần cho QLoRA repeatability, gradient alignment, residual-aware initialization, adaptive rank, quantized gradient và gradient stabilization.
- Cần backward search từ các paper trên và forward search ngay trước submission.
- Cần đưa mọi baseline tương thích vào thử nghiệm hoặc ghi rõ lý do không thể chạy công bằng.
- Vietnamese QA là testbed thực nghiệm; không dùng nó một mình để claim method novelty.
