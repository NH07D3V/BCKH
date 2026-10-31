# Đo lường và dự đoán độ ổn định QLoRA 4-bit trên mô hình ngôn ngữ nhỏ tiếng Việt

## Tóm tắt

QLoRA giúp fine-tune mô hình ngôn ngữ trong giới hạn bộ nhớ thấp, nhưng chất lượng cuối và độ ổn định giữa các lần chạy có thể phụ thuộc vào seed, lượng tử hóa và hình học của cập nhật LoRA. Bài báo này xây dựng một protocol tái lập để kiểm định liệu quantization residual, gradient noise và gradient–residual alignment trong giai đoạn đầu có dự đoán được độ biến động và chất lượng cuối của QLoRA supervised fine-tuning trên tiếng Việt hay không. Thí nghiệm dùng Sailor2-1B-Chat làm model chính, Qwen2.5-0.5B-Instruct làm control cùng lineage và UIT-ViQuAD2.0 với grouped holdout theo context. Chúng tôi giữ cố định token budget, số bước, rank, precision và parser, đồng thời báo cáo chất lượng, variance giữa seed, thời gian, bộ nhớ và overhead telemetry. Kết quả chính sẽ được điền sau khi hoàn tất baseline và kiểm định ngoài mẫu: `[TBD]`. Nếu predictor không chuyển giao hoặc policy không thắng uniform dưới matched budget, bài báo báo cáo stability audit và failure boundary thay vì claim optimizer mới.

**Từ khóa:** QLoRA, LoRA, quantization, gradient geometry, stability, Vietnamese NLP, parameter-efficient fine-tuning.

## 1. Giới thiệu

Fine-tuning mô hình ngôn ngữ nhỏ trong môi trường GPU giới hạn thường dùng parameter-efficient fine-tuning kết hợp lượng tử hóa. QLoRA là một lựa chọn thực dụng, nhưng một run tốt không đảm bảo các run khác với cùng cấu hình cũng có kết quả tương tự. Sự biến động này làm khó việc chọn rank, learning rate và precision khi ngân sách thử nghiệm thấp.

Các nghiên cứu gần đây đã đề xuất khởi tạo LoRA theo gradient hoặc quantization discrepancy, phân bổ rank thích nghi, joint bit–rank allocation và các biện pháp ổn định gradient. Vì vậy, câu hỏi của chúng tôi không phải là tạo thêm một heuristic adaptive rank chung chung. Chúng tôi tập trung vào khả năng kiểm chứng của các tín hiệu sớm: chúng có dự đoán được run nào sẽ có loss spike, variance cao hoặc rank không đủ khi ngân sách được giữ cố định hay không?

Đóng góp của bài báo gồm:

1. Một protocol tái lập, grouped theo context, cho stability audit của 4-bit QLoRA trên QA tiếng Việt.
2. Phân tích multi-seed về quan hệ giữa quantization residual, gradient geometry và kết quả cuối.
3. Kiểm định ngoài mẫu theo seed/model, cùng đo overhead của telemetry.
4. Một policy phân bổ rank chỉ được đề xuất khi predictor vượt qua các cổng kiểm định định trước; nếu không, báo cáo failure boundary có thể tái lập.

## 2. Câu hỏi nghiên cứu và giả thuyết

**RQ1.** Cùng một recipe QLoRA biến động bao nhiêu giữa các seed trên QA tiếng Việt?

**RQ2.** Tín hiệu nào dự đoán tốt nhất final EM/F1, validation loss variance và loss spike: loss-only, gradient-only, residual-only hay feature kết hợp?

**RQ3.** Predictor có chuyển giao sang seed/model chưa thấy không?

**RQ4.** Nếu dùng predictor để phân bổ rank, policy có cải thiện quality–stability–memory–time Pareto dưới cùng ngân sách không?

**H1.** Feature kết hợp residual và gradient geometry có predictive gain ngoài loss-only.

**H2.** Transfer trong cùng lineage có thể tồn tại nhưng suy giảm khi đổi model.

**H3.** Risk-aware rank policy chỉ được chấp nhận nếu thắng uniform ở cùng tổng rank, token, step, memory và telemetry overhead.

## 3. Công trình liên quan

QLoRA đưa NF4, double quantization và paged optimizer vào fine-tuning 4-bit. Các hướng LoRA-GA, LoRA-One và Q-GaLore đã nghiên cứu gradient alignment hoặc low-rank gradient adaptation. QuAILoRA, CLoQ, QERA và SRR xử lý quantization discrepancy hoặc output/spectral preservation. AdaLoRA, QDyLoRA, GoRA, IGU-LoRA và QR-Adaptor cho thấy adaptive rank hoặc joint bit–rank allocation là một lĩnh vực đông. Một nghiên cứu repeatability đã ghi nhận kết quả QLoRA không hoàn toàn lặp lại giữa các lần chạy.

Khoảng trống ứng viên của chúng tôi nằm ở predictive validity và failure boundary: liệu tổ hợp residual lượng tử và stochastic gradient geometry có dự đoán được variance, loss spike hoặc rank sufficiency ngoài seed/model trong QLoRA SFT tiếng Việt hay không, khi tính cả chi phí đo feature. Bảng nguồn và tình trạng rà soát được ghi tại [`docs/literature_matrix.md`](../docs/literature_matrix.md).

## 4. Dữ liệu và protocol

### 4.1. Model

Model chính là `sail/Sailor2-1B-Chat`, revision `51b48ecd7c0629e4c79dc927a0445e6b671d8692`. Model control là `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`. Hai model cùng lineage Qwen/Sailor, nên control này kiểm tra controlled adaptation chứ chưa chứng minh transfer giữa hai họ độc lập.

### 4.2. Dataset và split

Dataset là `taidng/UIT-ViQuAD2.0`, revision `406f09a45cc106a8f7b7fd0c25078883fe58cb1`. Split nhóm được tạo từ dữ liệu có nhãn theo khóa `(title, context)` với seed 13: train 22.691 mẫu, dev 2.866 mẫu và held-out test 2.897 mẫu. Split test gốc của bản phát hành không có nhãn và không được dùng để chọn phương pháp.

### 4.3. Fine-tuning

Mọi run dùng QLoRA NF4 4-bit, double quantization, LoRA rank 8, alpha 16, dropout 0,05, sequence length 512, batch hiệu dụng 8, learning rate `2e-4`, warmup ratio 0,03 và một epoch. Seeds chính là 13, 37 và 73. Template và parser được cố định trong mã nguồn; raw output được lưu cùng prediction đã parse.

### 4.4. Metrics

Chất lượng gồm Exact Match và token-level F1. Độ ổn định gồm mean ± SD theo seed, worst-seed gap, validation loss variance và loss-spike rate. Hiệu quả gồm wall-clock, peak VRAM, throughput, số token và overhead telemetry. Mọi con số chưa có artifact ghi `[TBD]`.

## 5. Telemetry và predictor

Telemetry pilot ghi gradient norm **trước clip**, update norm, parameter norm và update-to-weight ratio của từng module LoRA tại các mốc định kỳ bằng [`code/gradient_telemetry.py`](../code/gradient_telemetry.py). Sampling chạy trong `on_optimizer_step` (vì Trainer gọi `model.zero_grad()` trước `on_step_end`) và norm thô được thu bằng `register_post_accumulate_grad_hook`, nên không bị `max_grad_norm` ép phẳng.

Vòng này chỉ dùng các đại lượng đo được: mức gradient, độ biến thiên trong run, độ dốc, update-to-weight ratio và hình học module (depth, loại module). Các feature giàu thông tin hơn như residual energy, gradient noise proxy, signed/absolute alignment và effective rank **chưa được thu**, nên không xuất hiện trong kết quả. Predictor được fit trên seed 13+37 và đánh giá ngoài mẫu trên seed 73, sau đó chuyển Sailor2↔Qwen; loss-only và shape-only là control bắt buộc.

QGA-P chỉ được gọi là policy khi có định nghĩa rõ, pseudocode, ablation và kết quả vượt uniform dưới matched budget. Nếu điều kiện không đạt, phần này báo cáo predictor thất bại hoặc stability map — đó là trường hợp của vòng này.

## 6. Kết quả

### 6.1. Kiểm tra pipeline

Gate 1 đã đạt: asset, split, parser, tokenization và calibration chạy được. Calibration Sailor2 seed 13 trong 100 steps có train loss `1.798`, eval loss `1.985`, trên 32 prediction kiểm tra đạt EM `0.34375`, F1 `0.41422`. Đây là kiểm tra pipeline với cỡ mẫu nhỏ, không phải kết quả chính.

### 6.2. Baseline multi-seed

| Model | Seed | EM | F1 | Val loss | Runtime | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Sailor2-1B-Chat | 13 | 0.3343 | 0.4257 | 2.332 | 1h56m | 5.58 GiB |
| Sailor2-1B-Chat | 37 | 0.3416 | 0.4388 | 2.342 | 1h49m | 5.58 GiB |
| Sailor2-1B-Chat | 73 | 0.3402 | 0.4404 | 2.348 | 2h23m | 5.58 GiB |
| Qwen2.5-0.5B-Instruct | 13 | 0.3193 | 0.3722 | 2.427 | 58m | 4.67 GiB |
| Qwen2.5-0.5B-Instruct | 37 | 0.3217 | 0.3701 | 2.428 | 1h00m | 4.67 GiB |
| Qwen2.5-0.5B-Instruct | 73 | 0.3196 | 0.3700 | 2.429 | 1h01m | 4.67 GiB |

Bảng đầy đủ kèm train loss, wall-clock giây và bản gốc metric nằm ở [`paper/baseline_results.md`](baseline_results.md) và `results/baseline_summary.json`.

Theo nhóm, Sailor2 đạt EM `0.3387 ± 0.0039` và F1 `0.4350 ± 0.0081`, Qwen đạt EM `0.3202 ± 0.0013` và F1 `0.3708 ± 0.0013` (mean ± SD trên ba seed). Sailor2 hơn Qwen khoảng `+1,8` điểm EM và `+6,4` điểm F1 với độ lệch chuẩn giữa seed nhỏ hơn một điểm phần trăm, nên khác biệt model lớn hơn nhiễu seed. Vì spread giữa seed nhỏ, phần predictor phải giải thích rủi ro **trong từng model** (layer/module nào bất ổn) chứ không thể dựa vào dao động chất lượng cuối.

Sailor2 seed 13 được resume từ `checkpoint-250`, nên runtime và train loss của seed này chỉ tính cho bước 250–2.837; EM/F1 và val loss vẫn tính trên toàn bộ grouped dev. Toàn bộ sáu run dùng chung split, seed, ngân sách một epoch và recipe đã khóa trong [`docs/protocol_frozen.md`](../docs/protocol_frozen.md).

### 6.3. Bản đồ ổn định module

Sáu pilot telemetry chạy 200 optimizer step trên 2.048 mẫu train, ghi 20 mốc mỗi run; mỗi record chứa gradient norm trước clip, update norm, parameter norm và update-to-weight ratio cho từng module LoRA (336 module ở Sailor2, 168 ở Qwen). Không record nào rỗng.

Thứ hạng gradient norm theo module gần như trùng nhau giữa ba seed: Spearman `0.981–0.986` (Sailor2) và `0.983–0.986` (Qwen). Nói cách khác, "module nào nhận gradient lớn" là thuộc tính ổn định của kiến trúc, không phải dao động ngẫu nhiên của seed.

| Module kind | Sailor2 \|g\| mean | Sailor2 risk | Qwen \|g\| mean | Qwen risk |
|---|---:|---:|---:|---:|
| `gate_proj` | 0.1137 | 0.760 | 0.1795 | 0.796 |
| `up_proj` | 0.1119 | 0.665 | 0.1588 | 0.678 |
| `down_proj` | 0.0530 | 0.546 | 0.0693 | 0.514 |
| `o_proj` | 0.0535 | 0.424 | 0.0665 | 0.383 |
| `v_proj` | 0.0678 | 0.407 | 0.0779 | 0.360 |
| `q_proj` | 0.0221 | 0.364 | 0.0323 | 0.390 |
| `k_proj` | 0.0230 | 0.335 | 0.0356 | 0.379 |

MLP `gate/up` nhận gradient lớn hơn attention `q/k` khoảng 3–5 lần trên cả hai kiến trúc, và thứ tự này giữ nguyên khi chuyển model. Trong suốt pilot, train loss giảm đơn điệu (Sailor2 `2.12 → 1.18`, Qwen `2.78 → 1.94`) và không có loss spike. Sự kiện bất thường duy nhất là một cú tăng gradient `1.77 → 4.65 → 1.80` ở Sailor2 seed 73 tại step 180, lan ra 195/336 module, tức là do một batch khó chứ không phải một module mất ổn định.

Bảng đầy đủ theo 336/168 module, gồm volatility trong run, spread giữa seed và risk score, ở [`docs/telemetry_analysis.md`](../docs/telemetry_analysis.md).

### 6.4. Predictor QGA-P

Protocol và ngưỡng được khóa trước khi đọc dữ liệu ([`docs/gate2_protocol.md`](../docs/gate2_protocol.md)). Feature lấy từ step 10–50, nhãn lấy từ step 160–200; ridge regression, Spearman ρ out-of-sample, bootstrap 2.000 lần và permutation 2.000 lần; train seed 13+37, hold out seed 73.

| Split | Nhãn | F_grad | F_update | F_shape | F_full |
|---|---|---:|---:|---:|---:|
| Sailor2 seed 73 | `late_update_ratio` | 0.545 | 0.891 | 0.917 | 0.939 |
| Sailor2 seed 73 | `late_grad_cv` | −0.043 | 0.196 | 0.623 | 0.561 |
| Qwen seed 73 | `late_update_ratio` | 0.390 | 0.887 | 0.930 | 0.958 |
| Qwen seed 73 | `late_grad_cv` | 0.316 | 0.635 | 0.617 | 0.640 |
| Qwen → Sailor2 | `late_update_ratio` | 0.447 | 0.890 | 0.919 | 0.908 |
| Qwen → Sailor2 | `late_grad_cv` | 0.128 | 0.236 | 0.399 | 0.225 |
| Sailor2 → Qwen | `late_update_ratio` | 0.389 | 0.878 | 0.935 | 0.922 |
| Sailor2 → Qwen | `late_grad_cv` | 0.075 | 0.466 | 0.298 | 0.270 |

Kết quả trông tốt ở nhãn chính, nhưng hai kiểm tra cho thấy đó là hiệu ứng cấu trúc:

1. **Phân rã phương sai.** `late_update_ratio` có 99.5% (Sailor2) và 99.6% (Qwen) phương sai nằm giữa các module, chỉ 0.4–0.5% nằm giữa các seed. Nhãn gần như là hằng số hình học, nên `F_shape` (depth + loại module) đã đạt ρ ≈ 0.92–0.94 mà không dùng bất kỳ tín hiệu động học nào. Điều này khớp với Adam: độ lớn cập nhật tỉ lệ với `lr·√n/‖w‖`, một đại lượng hình học.
2. **Nhãn phần lệch riêng của seed** (exploratory): lấy `late − trung bình hai seed còn lại` cho từng module rồi dự đoán phần lệch đó. ρ nằm trong `[−0.10; 0.13]` cho mọi feature set trên cả hai model — không có tín hiệu.

Theo điều khoản giá trị bổ sung của protocol (Δρ ≥ 0.05 so với control tốt nhất), `F_full` không đạt: Δρ lần lượt `+0.028, +0.021, −0.011, −0.013`. Vì vậy kết quả này **không** được gọi là predictor rủi ro dựa trên gradient; nó là **stability audit** kèm **negative result** cho giả thuyết telemetry warm-up dự đoán phần bất ổn riêng của seed ở ngân sách 200 step / 2.048 mẫu. Chi tiết và artifact ở [`docs/gate2_results.md`](../docs/gate2_results.md).

## 7. Phân tích lỗi và giới hạn

Các giới hạn cần báo cáo gồm: dataset QA trích xuất đơn, kích thước model nhỏ, hai model cùng lineage, tỷ lệ prompt+kèm answer vượt 512 token khoảng 3,9% trên grouped train, chi phí đo telemetry, chất lượng parser và khả năng kết quả thay đổi khi dùng model family độc lập. Không dùng calibration metric hoặc smoke test để claim SOTA.

Riêng kết quả Gate 2 có bốn giới hạn quyết định phạm vi claim:

- Pilot chỉ chạy 200 step trên 2.048 mẫu, tức khoảng 7% của một epoch; vùng huấn luyện có loss giảm đơn điệu nên **không có sự kiện bất ổn định** để dự đoán. Kết quả âm chỉ đúng cho ngân sách này.
- Nhãn `late_update_ratio` gần như là hằng số hình học (99,5% phương sai giữa module). Một phần "thành công" của protocol là do chọn nhãn chưa đủ động học; điều này đã được ghi nhận thay vì che đi.
- Telemetry chưa có residual energy, gradient cosine hay effective rank, nên chưa loại trừ được khả năng các feature đó mang tín hiệu mà norm không có.
- Chuyển model chỉ kiểm tra hai kiến trúc cùng lineage (Qwen2.5 và Sailor2 đều dựa trên Qwen2), nên "transfer" ở đây chưa phải bằng chứng cross-family.

Held-out test vẫn chưa mở; mọi kết quả trong §6 là trên grouped dev.

## 8. Kết luận

Baseline QLoRA 1 epoch đã hoàn tất trên hai model và ba seed với spread giữa seed nhỏ (SD EM ≤ 0,004), nên khác biệt giữa Sailor2 và Qwen lớn hơn nhiễu seed. Telemetry cho thấy thứ hạng rủi ro theo module là thuộc tính kiến trúc, tái lập gần như hoàn hảo giữa seed (ρ ≥ 0,98), với MLP `gate/up` cao hơn attention `q/k` 3–5 lần.

Predictor QGA-P **không** vượt được control hình học: khi tách phần lệch riêng của seed, telemetry warm-up không dự đoán được gì (|ρ| ≤ 0,13). Vì vậy bài không mở nhánh QGA-R/QGA-S và không gọi QGA-P là method; đóng góp được báo cáo là stability audit có kiểm soát cùng negative result cho giả thuyết predictor gradient ở ngân sách nhỏ. Claim cuối cùng vẫn chờ held-out test và sẽ trả lời trực tiếp RQ1–RQ4.

## Phụ lục tái lập

- Protocol: [`docs/protocol_frozen.md`](../docs/protocol_frozen.md)
- Gate audit: [`docs/gate1_audit.md`](../docs/gate1_audit.md)
- Experiment log: [`docs/experiment_log.md`](../docs/experiment_log.md)
- Literature matrix: [`docs/literature_matrix.md`](../docs/literature_matrix.md)
- Configs: [`configs/sailor2_qlora_vi.yaml`](../configs/sailor2_qlora_vi.yaml), [`configs/baseline_qlora_vi.yaml`](../configs/baseline_qlora_vi.yaml)
