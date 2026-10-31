# Gate 2b — Giao thức pilot dài (khóa trước khi chạy)

Ngày khóa: 2026-10-31, ghi **trước** khi pilot dài bắt đầu.

Lý do: pilot 200 step ([`docs/gate2_results.md`](gate2_results.md)) không có sự kiện bất ổn định nào, nên không thể kiểm tra giả thuyết "telemetry warm-up dự đoán module sẽ mất ổn định". Vòng này chạy lâu hơn với toàn bộ train set, thêm tín hiệu hướng gradient, và định nghĩa trước sự kiện bất ổn định.

## 1. Cấu hình pilot

- 6 run: Sailor2-1B-Chat và Qwen2.5-0.5B-Instruct, seed 13/37/73.
- 1.000 optimizer step, **toàn bộ grouped train** (22.691 mẫu, không giới hạn `--max-train-samples`), batch 4 × accumulation 2.
- Recipe giữ nguyên như baseline: NF4, LoRA, TF32, không gradient checkpointing, `max_grad_norm=1.0`.
- Telemetry mỗi 10 step ⇒ 100 record/run. Eval mỗi 250 step trên 256 mẫu dev để theo dõi loss.
- Collector bổ sung `gradient_cosine`: cosine giữa vector gradient của module tại record hiện tại và record trước (10 step), lấy trung bình có trọng số theo gradient norm của `lora_A`/`lora_B`. Clip gradient là phép nhân vô hướng nên không đổi hướng, do đó cosine vẫn là cosine của gradient trước clip.

## 2. Cửa sổ

- Warm-up (feature): record ở step 10–100 (10 mốc).
- Theo dõi (nhãn): record ở step 110–1000 (90 mốc).

## 3. Nhãn sự kiện bất ổn định (khóa)

Với mỗi module `m`:

- `med_m`, `mad_m` = median và MAD của `gradient_norm` trong warm-up; `scale_m = 1.4826 · mad_m` (nếu `mad_m = 0`, dùng độ lệch chuẩn warm-up; nếu vẫn 0, module bị loại).
- Tại record `t` trong cửa sổ theo dõi: **spike** khi `(g_t − med_m) / scale_m ≥ 4.0` **và** `g_t ≥ 2 · med_m`.
- Module **bất ổn định** khi có spike ở **≥ 3 record khác nhau** trong cửa sổ theo dõi. Yêu cầu ≥ 3 để loại cú tăng do một batch khó (như sự kiện step 180 của pilot trước, chỉ 1 record).

Nhãn phụ (chỉ báo cáo): `spike_records` = số record có spike; `severity` = max z.

Nếu một model có tỉ lệ module bất ổn định < 2% hoặc > 98%, AUROC được báo cáo kèm cảnh báo mất cân bằng và kết luận dựa thêm vào tương quan hạng với `spike_records`.

## 4. Feature (chỉ từ warm-up)

| Tên | Nội dung |
|---|---|
| `F_shape` | depth chuẩn hóa + one-hot loại module |
| `F_grad` | log gradient norm mean, CV, độ dốc chuẩn hóa |
| `F_update` | update/weight mean, max, độ dốc chuẩn hóa |
| `F_cosine` | cosine trung bình (bỏ record đầu), độ lệch chuẩn cosine, tỉ lệ cosine âm |
| `F_full` | hợp của bốn nhóm trên |

Mô hình: **ridge regression** trên nhãn nhị phân (linear probability), feature chuẩn hóa trong từng run, hệ số L2 chọn bằng leave-one-seed-out **AUROC** trên hai seed train (lưới `0.01, 0.1, 1, 10, 100`). Không dùng logistic để tránh phải chọn thêm optimizer; AUROC bất biến với phép biến đổi đơn điệu nên điểm số thô là đủ. Nếu nhãn mất cân bằng nghiêm trọng, vẫn giữ ridge và chỉ cảnh báo, không đổi mô hình sau khi xem nhãn.

## 5. Chia tập và metric

- Chính: train seed 13+37, test seed 73, cho từng model.
- Chuyển model: Sailor2 → Qwen và ngược lại (dùng cả 3 seed).
- Metric chính: **AUROC** trên module của tập test (Mann–Whitney U).
- Bootstrap 2.000 lần trên module để lấy CI 95%; permutation 2.000 lần (hoán vị nhãn train rồi fit lại) cho p-value.
- Báo cáo thêm: prevalence của nhãn, AUPRC, và Spearman giữa dự đoán với `spike_records`.

## 6. Ngưỡng quyết định (khóa)

Predictor được coi là có tín hiệu khi, trên **cả hai model** ở split chính:

1. AUROC ≥ **0.65**;
2. cận dưới bootstrap 95% > **0.50**;
3. p permutation < **0.01**;
4. `F_full` hơn `F_shape` ≥ **0.05** AUROC.

Transfer được coi là giữ tín hiệu khi AUROC ≥ **0.60** ở cả hai chiều.

## 7. Quy tắc

- Đạt mục 6 và transfer: mở nhánh B (QGA-R) với risk score từ predictor, matched budget.
- Đạt trong model nhưng không transfer: dừng ở stability audit theo model.
- Không đạt: negative result được xác nhận ở ngân sách 1.000 step; chỉ mở rộng lên full epoch nếu cần cho báo cáo, và ghi rõ là exploratory.
- Nếu **không có module nào** đạt nhãn bất ổn định: kết luận "không có sự kiện để dự đoán ở 1.000 step", không suy ra predictor vô dụng ngoài ngân sách này.

## 8. Artifact

- `results/telemetry_long_<model>_seed<seed>.jsonl` (gitignored, 100 record/run).
- `results/instability_analysis.json`, `docs/instability_analysis.md`.
- `results/gate2b_predictor.json`.
- `code/analyze_instability.py`, `code/fit_gate2b_predictor.py`.
