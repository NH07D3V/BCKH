# Gate 2 — Giao thức QGA-P (khóa trước khi đọc telemetry)

Ngày khóa: 2026-10-31, ghi **trước** khi bất kỳ file `results/telemetry_*.jsonl` hợp lệ nào được phân tích.

Mục tiêu: kiểm tra xem telemetry gradient/update ở đoạn warm-up có dự đoán được mức bất ổn định của từng module LoRA về cuối pilot hay không. Đây là bài kiểm tra **predictor**, chưa phải policy rank.

## 1. Đơn vị quan sát và nhãn

Một dòng dữ liệu là một cặp `(model, module)`, với `module` là khóa 336 module LoRA (`layers.<n>.<q|k|v|o|gate|up|down>_proj`) trên Sailor2 và Qwen.

Pilot: 200 optimizer step, telemetry mỗi 10 step ⇒ 20 mốc thời gian mỗi run.

- Cửa sổ warm-up (feature): 5 mốc đầu, step 10–50.
- Cửa sổ cuối (nhãn): 5 mốc cuối, step 160–200.

Hai nhãn chính, định nghĩa trước:

| Nhãn | Định nghĩa | Ý nghĩa |
|---|---|---|
| `late_update_ratio` | trung bình `update_to_weight_ratio` của module trong cửa sổ cuối | module còn dịch chuyển mạnh về cuối ⇒ rủi ro cao |
| `late_grad_cv` | hệ số biến thiên `gradient_norm` của module trong cửa sổ cuối | module có gradient dao động mạnh ⇒ rủi ro cao |

Nhãn phụ (chỉ báo cáo, không dùng để quyết định): `late_grad_mean`, `update_ratio_growth` = log(tỉ lệ cửa sổ cuối / cửa sổ warm-up).

## 2. Feature set

Tất cả feature chỉ dùng cửa sổ warm-up, chuẩn hóa trong từng run (z-score theo module) để không rò rỉ quy mô giữa model.

| Tên | Nội dung |
|---|---|
| `F_const` | chỉ intercept (control tối thiểu) |
| `F_loss` | loss trung bình của run trong cửa sổ warm-up (hằng số theo module, control) |
| `F_grad` | log gradient norm trung bình, CV, độ dốc chuẩn hóa của gradient norm |
| `F_update` | update/weight trung bình, max, độ dốc chuẩn hóa |
| `F_shape` | depth chuẩn hóa, one-hot module kind (q/k/v/o/gate/up/down) |
| `F_full` | hợp của `F_grad`, `F_update`, `F_shape` |

Mô hình: ridge regression (hệ số L2 chọn bằng leave-one-seed-out bên trong tập train), dự đoán xếp hạng. Không dùng mô hình phi tuyến ở bước này.

## 3. Chia tập

- **Trong model (chính):** train = seed 13 + 37, test = seed 73. Không dùng seed 73 cho bất kỳ bước chọn feature/hyperparameter nào.
- **Chuyển model:** train = Sailor2 (3 seed), test = Qwen (3 seed) và chiều ngược lại, cùng task, cùng nhãn.
- Module không có đủ 20 mốc telemetry bị loại khỏi cả train và test; tỉ lệ loại được báo cáo.

## 4. Metric

- Spearman ρ giữa dự đoán và nhãn trên tập test (out-of-sample), cho từng nhãn.
- Bootstrap 2.000 lần trên module (seed cố định 20261031) để lấy khoảng tin cậy 95%.
- Kiểm định hoán vị nhãn 2.000 lần trong tập train để có null distribution.
- Với `F_full` so với từng control trên cùng tập test: Δρ.

## 5. Ngưỡng quyết định (khóa)

Một nhãn được coi là **dự đoán được** khi đồng thời:

1. ρ out-of-sample trên seed 73 ≥ **0.40**;
2. cận dưới bootstrap 95% > **0**;
3. p-value hoán vị < **0.01**.

`F_full` chỉ được coi là có giá trị bổ sung khi Δρ so với control tốt nhất ≥ **0.05** trên cùng nhãn.

Chuyển model được coi là **giữ được tín hiệu** khi ρ ≥ **0.20** và cận dưới bootstrap > 0 ở cả hai chiều.

## 6. Quy tắc ra quyết định

- Đạt mục 5 cho ít nhất một nhãn **và** giữ được transfer: tiếp tục nhánh B (QGA-R) với risk score lấy từ predictor.
- Đạt trong model nhưng không transfer: dừng ở **stability audit** theo model; không tuyên bố method tổng quát.
- Không đạt: báo cáo **negative result** kèm bản đồ ổn định module, không mở nhánh B/C.

Mọi thay đổi sau thời điểm này phải ghi thành "exploratory" trong `docs/experiment_log.md` và không được dùng để thay ngưỡng chính.

## 7. Artifact bắt buộc

- `results/telemetry_analysis.json`, `docs/telemetry_analysis.md` (mô tả dữ liệu).
- `results/gate2_predictor.json`: ρ, CI, p-value, Δρ cho từng feature set × nhãn × split.
- `code/fit_gate2_predictor.py`: tái lập toàn bộ bảng trên từ JSONL gốc.
