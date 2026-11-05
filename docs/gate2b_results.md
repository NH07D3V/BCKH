# Gate 2b — Kết quả pilot dài 1.000 step

Ngày phân tích: 2026-11-05. Protocol khóa trước khi có dữ liệu: [`docs/gate2b_protocol.md`](gate2b_protocol.md).

## 1. Kết luận ngắn

Lần này **có sự kiện bất ổn định thật**: 10,7–23,8% module đạt nhãn spike lặp lại. Predictor đạt AUROC `0,94–0,97` và transfer `0,88–0,94`, nhưng **không đạt tiêu chí giá trị bổ sung** đã khóa: `F_full` chỉ hơn `F_shape` `+0,005` (Sailor2) và `+0,022` (Qwen), dưới ngưỡng `0,05`.

Nguyên nhân: bất ổn định là **thuộc tính kiến trúc**, không phải tín hiệu động học riêng của từng run. `gate_proj` chiếm 95,8% module bất ổn định ở Sailor2 và 70,8% ở Qwen; chỉ cần biết depth + loại module là đủ xếp hạng rủi ro.

Theo §7 của protocol: **negative result được xác nhận ở ngân sách 1.000 step**, không mở QGA-R.

## 2. Chất lượng pilot

6 run × 100 record (step 10–1000), không record rỗng, không OOM. Sailor2 336 module, Qwen 168 module.

| Run | Unstable | Prevalence | Max z | Loss spike |
|---|---:|---:|---:|---:|
| sailor2_seed13 | 63/336 | 18,8% | 73,2 | 0 |
| sailor2_seed37 | 80/336 | 23,8% | 63,3 | 0 |
| sailor2_seed73 | 74/336 | 22,0% | 123,8 | 0 |
| qwen_seed13 | 19/168 | 11,3% | 37,3 | 0* |
| qwen_seed37 | 19/168 | 11,3% | 51,0 | 0* |
| qwen_seed73 | 18/168 | 10,7% | 37,1 | 0* |

\* Bộ đếm ban đầu báo 1 “loss spike” cho mỗi run Qwen, nhưng kiểm tra thủ công cho thấy đó là **loss đầu tiên** (điểm khởi tạo cao hơn median cả run), không phải sự kiện giữa chừng. Không run nào có loss spike thật.

## 3. Bản đồ bất ổn định

Tỉ lệ module bất ổn định theo loại:

| Kind | Sailor2 | Qwen |
|---|---:|---:|
| `gate_proj` | **95,8%** (46/48) | **70,8%** (17/24) |
| `down_proj` | 47,9% (23/48) | 0,0% |
| `k_proj` | 31,2% (15/48) | 8,3% (2/24) |
| `q_proj` | 14,6% (7/48) | 12,5% (3/24) |
| `up_proj` | 12,5% (6/48) | 8,3% (2/24) |
| `o_proj` | 0,0% | 0,0% |
| `v_proj` | 0,0% | 0,0% |

Theo depth (4 dải):

| Layers | Sailor2 | Qwen |
|---|---:|---:|
| đầu | 38,1% (0–11) | 21,4% (0–5) |
| giữa-1 | 42,9% (12–23) | 16,7% (6–11) |
| giữa-2 | 21,4% (24–35) | 11,9% (12–17) |
| cuối | 13,1% (36–47) | 7,1% (18–23) |

Đồng thuận giữa seed (Jaccard tập module bất ổn định):

- Sailor2: `0,59` (13↔37), `0,61` (13↔73), `0,71` (37↔73).
- Qwen: `0,73`, `0,76`, `0,54`.

Phân loại module theo 3 seed: Sailor2 có **49 luôn bất ổn định**, 48 lúc có lúc không, 239 không bao giờ; Qwen 13 / 11 / 144. Tức là phần lớn rủi ro là hằng số kiến trúc, chỉ khoảng 11–14% module là “mixed”.

## 4. Predictor (đúng protocol)

Train seed 13+37, test seed 73; AUROC trên module, bootstrap 2.000, permutation 2.000.

#### sailor2_seed73 (train 672, test 336)

| Feature set | AUROC | CI 95% | AUPRC | p (perm) |
|---|---:|---|---:|---:|
| `F_shape` | 0,952 | [0,928; 0,973] | 0,840 | 0,0005 |
| `F_grad` | 0,772 | [0,716; 0,825] | 0,399 | 0,0255 |
| `F_update` | 0,815 | [0,761; 0,869] | 0,459 | 0,0150 |
| `F_cosine` | 0,696 | [0,624; 0,764] | 0,357 | 0,0285 |
| `F_full` | 0,958 | [0,935; 0,975] | 0,875 | 0,0005 |

#### qwen_seed73 (train 336, test 168)

| Feature set | AUROC | CI 95% | AUPRC | p (perm) |
|---|---:|---|---:|---:|
| `F_shape` | 0,944 | [0,849; 0,997] | 0,837 | 0,0100 |
| `F_grad` | 0,814 | [0,732; 0,888] | 0,312 | 0,0840 |
| `F_update` | 0,895 | [0,831; 0,952] | 0,554 | 0,0035 |
| `F_cosine` | 0,789 | [0,696; 0,871] | 0,306 | 0,0140 |
| `F_full` | 0,966 | [0,905; 1,000] | 0,909 | 0,0010 |

#### Chuyển model

| Split | `F_shape` | `F_grad` | `F_update` | `F_cosine` | `F_full` | AUPRC (full) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen → Sailor2 | 0,851 | 0,687 | 0,788 | 0,597 | 0,881 | 0,765 |
| Sailor2 → Qwen | 0,915 | 0,834 | 0,863 | 0,661 | 0,940 | 0,867 |

## 5. Đối chiếu ngưỡng đã khóa

| Tiêu chí | Sailor2 | Qwen | Kết quả |
|---|---:|---:|---|
| AUROC ≥ 0,65 | 0,958 | 0,966 | ✅ |
| CI dưới > 0,50 | 0,935 | 0,905 | ✅ |
| p < 0,01 | 0,0005 | 0,0010 | ✅ |
| `F_full − F_shape` ≥ 0,05 | **+0,005** | **+0,022** | ❌ |
| Transfer ≥ 0,60 | 0,940 | 0,881 | ✅ |

Tiêu chí 4 trượt ở cả bốn split (`+0,005 / +0,022 / +0,030 / +0,025`). Theo protocol, không được gọi đây là predictor rủi ro dựa trên gradient.

Ghi nhận phụ (không dùng để quyết định): AUPRC của `F_full` cao hơn `F_shape` ở mọi split (`+0,035`, `+0,072`, `+0,053`, `+0,132`), tức mô hình đầy đủ xếp các module bất ổn định lên đầu tốt hơn một chút — nhưng mức tăng AUROC vẫn dưới ngưỡng đã đặt trước.

## 6. Kiểm tra exploratory: bất ổn riêng của seed

Định nghĩa nhãn mới: module bất ổn định ở seed 73 **và** ổn định ở cả seed 13 lẫn 37. Số dương tính rất ít: **7/336** (Sailor2) và **2/168** (Qwen).

| Feature set | Sailor2 AUROC | Qwen AUROC |
|---|---:|---:|
| `F_shape` | 0,637 | 0,483 |
| `F_grad` | 0,624 | 0,587 |
| `F_update` | 0,772 | 0,515 |
| `F_cosine` | 0,490 | 0,090 |
| `F_full` | 0,663 | 0,322 |

Với 7 và 2 dương tính, CI rộng đến mức không thể kết luận; đây chỉ là ghi nhận để tham khảo, không phải bằng chứng. Điều đáng nói là `F_shape` mất gần hết khả năng dự đoán khi nhãn được tách khỏi thành phần cấu trúc, đúng như dự đoán nếu tín hiệu chủ yếu là kiến trúc.

## 7. Quyết định

- Không mở nhánh B (QGA-R): policy phân bổ rank theo “risk” sẽ chỉ tái tạo một prior kiến trúc cố định (gate_proj cao, o/v thấp), không phải điều chỉnh theo run.
- Kết quả được báo cáo là **stability benchmark + negative result** cho predictor động học, với phát hiện dương tính là **bản đồ rủi ro kiến trúc tái lập và transfer được** giữa hai model.
- Điều kiện mở lại: cần tín hiệu dự đoán được phần **seed-specific** (hiện chỉ 2–7 module mỗi model) hoặc một recipe có bất ổn định do động học (LR cao, batch nhỏ, data khó) thay vì do hình học module.

## 8. Artifact

- `results/telemetry_long_*.jsonl` (6 file × 100 record, gitignored)
- `results/instability_analysis.json`, `docs/instability_analysis.md`
- `results/gate2b_predictor.json`
- `code/analyze_instability.py`, `code/fit_gate2b_predictor.py`
