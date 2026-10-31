# Gate 2 — Kết quả QGA-P trên telemetry pilot

Ngày phân tích: 2026-10-31. Protocol khóa trước khi đọc dữ liệu: [`docs/gate2_protocol.md`](gate2_protocol.md).

## 1. Kết luận ngắn

**Không mở nhánh QGA-R trên bằng chứng này.** Nhãn chính của protocol bị chi phối bởi cấu trúc module (99,5% phương sai nằm *giữa* các module, không phải giữa các seed), nên ρ cao không phải bằng chứng cho predictor dựa trên động học gradient. Khi loại thành phần cấu trúc, telemetry warm-up **không dự đoán** được phần lệch riêng của từng seed (|ρ| ≤ 0,13).

Kết quả được báo cáo dưới dạng **stability audit + negative result** theo [`docs/roadmap_qga_lora_vi.md`](roadmap_qga_lora_vi.md) §7.

## 2. Chất lượng pilot

6 run × 200 optimizer step, telemetry mỗi 10 step ⇒ 20 record/run, không record rỗng.

| Run | Module | Records | |g| mean | |g| max | Đỉnh bất thường |
|---|---:|---:|---:|---:|---|
| sailor2_seed13 | 336 | 20 | 1.34 | 2.02 | không |
| sailor2_seed37 | 336 | 20 | 1.32 | 1.90 | không |
| sailor2_seed73 | 336 | 20 | 1.58 | 4.65 | step 180, tăng vọt toàn cục rồi hồi phục |
| qwen_seed13 | 168 | 20 | 1.46 | 1.93 | không |
| qwen_seed37 | 168 | 20 | 1.42 | 1.89 | không |
| qwen_seed73 | 168 | 20 | 1.57 | 2.18 | không |

Train loss giảm đơn điệu trong cả 6 run (Sailor2 `2,12 → 1,18`; Qwen `2,78 → 1,94`), không có loss spike (robust z < 3,5). Tổng gradient norm tăng nhẹ từ ~1,0 ở step 10 lên ~1,7 ở step 200. Sự kiện duy nhất là một cú tăng vọt ở Sailor2 seed 73: `1,77 → 4,65 → 1,80` tại step 180, lan ra **195/336 module** (cùng một batch khó), không phải một module đơn lẻ. Norm ghi được là norm **trước clip**; ngưỡng `max_grad_norm=1,0` chỉ áp sau khi hook đã ghi xong. Nghĩa là pilot nằm trong vùng huấn luyện ổn định: chưa có sự kiện bất ổn định kéo dài nào để dự đoán, và cú spike theo batch thì predictor theo module về nguyên tắc không bắt được.

## 3. Bản đồ ổn định cấu trúc

Xếp hạng gradient norm theo module gần như trùng nhau giữa các seed: Spearman `0,981–0,986` (Sailor2) và `0,983–0,986` (Qwen). Đây là quan sát mạnh nhất của pilot.

| Kind | Sailor2 |g| mean | Sailor2 risk | Qwen |g| mean | Qwen risk |
|---|---:|---:|---:|---:|
| `gate_proj` | 0,1137 | 0,760 | 0,1795 | 0,796 |
| `up_proj` | 0,1119 | 0,665 | 0,1588 | 0,678 |
| `v_proj` | 0,0678 | 0,407 | 0,0779 | 0,360 |
| `down_proj` | 0,0530 | 0,546 | 0,0693 | 0,514 |
| `o_proj` | 0,0535 | 0,424 | 0,0665 | 0,383 |
| `q_proj` | 0,0221 | 0,364 | 0,0323 | 0,390 |
| `k_proj` | 0,0230 | 0,335 | 0,0356 | 0,379 |

Thứ tự này giữ nguyên khi chuyển model: MLP `gate/up` nhận gradient lớn hơn attention `q/k` khoảng 3–5 lần trên cả hai kiến trúc. Bảng đầy đủ theo module nằm ở [`docs/telemetry_analysis.md`](telemetry_analysis.md) và `results/telemetry_analysis.json`.

## 4. Kết quả predictor đúng protocol

`ρ` là Spearman out-of-sample; CI là bootstrap 2.000 lần trên module; p là permutation 2.000 lần. Train seed 13+37, test seed 73.

#### sailor2_seed73 (train 672, test 336)

| Nhãn | Feature set | ρ | CI 95% | p (perm) |
|---|---|---:|---|---:|
| `late_update_ratio` | F_grad | 0,545 | [0,467; 0,615] | 0,0125 |
| `late_update_ratio` | F_update | 0,891 | [0,869; 0,907] | 0,0025 |
| `late_update_ratio` | F_shape | 0,917 | [0,899; 0,930] | 0,0005 |
| `late_update_ratio` | F_full | 0,939 | [0,924; 0,949] | 0,0005 |
| `late_grad_cv` | F_grad | −0,043 | [−0,151; 0,067] | 0,5457 |
| `late_grad_cv` | F_update | 0,196 | [0,086; 0,293] | 0,0330 |
| `late_grad_cv` | F_shape | 0,623 | [0,533; 0,702] | 0,0120 |
| `late_grad_cv` | F_full | 0,561 | [0,471; 0,641] | 0,0045 |

#### qwen_seed73 (train 336, test 168)

| Nhãn | Feature set | ρ | CI 95% | p (perm) |
|---|---|---:|---|---:|
| `late_update_ratio` | F_grad | 0,390 | [0,235; 0,519] | 0,0860 |
| `late_update_ratio` | F_update | 0,887 | [0,848; 0,912] | 0,0010 |
| `late_update_ratio` | F_shape | 0,930 | [0,906; 0,943] | 0,0005 |
| `late_update_ratio` | F_full | 0,958 | [0,940; 0,968] | 0,0005 |
| `late_grad_cv` | F_grad | 0,316 | [0,164; 0,445] | 0,0495 |
| `late_grad_cv` | F_update | 0,635 | [0,522; 0,728] | 0,0120 |
| `late_grad_cv` | F_shape | 0,617 | [0,496; 0,718] | 0,0025 |
| `late_grad_cv` | F_full | 0,640 | [0,520; 0,731] | 0,0020 |

#### Chuyển model

| Split | Nhãn | F_grad | F_update | F_shape | F_full |
|---|---|---:|---:|---:|---:|
| qwen → sailor2 | `late_update_ratio` | 0,447 | 0,890 | 0,919 | 0,908 |
| qwen → sailor2 | `late_grad_cv` | 0,128 | 0,236 | 0,399 | 0,225 |
| sailor2 → qwen | `late_update_ratio` | 0,389 | 0,878 | 0,935 | 0,922 |
| sailor2 → qwen | `late_grad_cv` | 0,075 | 0,466 | 0,298 | 0,270 |

`F_const` và `F_loss` đều cho ρ = 0 (loss là một số vô hướng cho cả run nên không thể xếp hạng module).

## 5. Vì sao "đạt ngưỡng" không đủ để gọi là predictor

Hai kiểm tra bổ sung, ghi rõ là **exploratory** (sau khi đã khóa protocol):

1. **Phân rã phương sai của nhãn.** `late_update_ratio` có 99,5% (Sailor2) và 99,6% (Qwen) phương sai nằm giữa các module, chỉ 0,4–0,5% nằm giữa các seed. Nhãn gần như là một hằng số cấu trúc, nên `F_shape` (depth + loại module) đã đạt ρ ≈ 0,92–0,94 mà không cần bất kỳ tín hiệu động học nào. Điều này khớp với việc Adam cập nhật tỉ lệ với `lr·√n/‖w‖`, một đại lượng hình học.
2. **Nhãn "phần lệch riêng của seed".** Với mỗi module, lấy `late − trung bình hai seed còn lại` rồi dự đoán phần lệch đó từ warm-up (train 13/37, test 73). Kết quả: ρ ∈ [−0,10; 0,13] cho cả `F_grad`, `F_update` và `F_full` trên cả hai model. Không có tín hiệu.

`late_grad_cv` là nhãn động học hơn (21% phương sai giữa module ở Sailor2, 66% ở Qwen), nhưng `F_full` không vượt `F_shape` và không transfer (p = 0,064 và 0,075 > 0,01).

Theo điều khoản "giá trị bổ sung" của protocol (Δρ ≥ 0,05 so với control tốt nhất), `F_full` **không đạt**: Δρ = +0,028, +0,021, −0,011, −0,013.

## 6. Quyết định

- Không mở nhánh B (QGA-R) từ kết quả này: tín hiệu dự đoán được là **thuộc tính kiến trúc**, không phải rủi ro động học của từng run, nên không có cơ sở để phân bổ lại rank theo "risk" đo từ gradient.
- Bài giữ nhánh A ở dạng **stability audit + negative result**: bản đồ ổn định cấu trúc (MLP gate/up > attention) tái lập trên hai model và ba seed, còn giả thuyết "telemetry gradient warm-up dự đoán rủi ro riêng của seed" bị bác bỏ ở ngân sách 200 step / 2.048 mẫu.
- Điều kiện để mở lại: pilot phải **có sự kiện bất ổn định thật** (loss spike, gradient spike lan rộng, phân kỳ cục bộ) và telemetry phải phủ vùng đó. Ở pilot hiện tại loss giảm đơn điệu nên không có gì để dự đoán.

## 7. Artifact

- `results/telemetry_*.jsonl` (6 file, 20 record/run)
- `results/telemetry_analysis.json`, `docs/telemetry_analysis.md`
- `results/gate2_predictor.json`
- `code/analyze_telemetry.py`, `code/fit_gate2_predictor.py`
