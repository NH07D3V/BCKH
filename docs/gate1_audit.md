# Gate 1 — Audit pipeline và dữ liệu

Ngày chạy: 2026-10-26 (UTC+7)

## Trạng thái

**PASS cho pipeline và calibration; đây chưa phải kết quả chính của nghiên cứu.**

## Bằng chứng đã kiểm tra

- Python 3.12.0, PyTorch 2.11.0+cu128, Transformers 5.20.2, PEFT 0.21.2, bitsandbytes 0.50.2.
- GPU: NVIDIA GeForce RTX 3070 Ti, 8 GiB; CUDA khả dụng; BF16 khả dụng.
- Model chính đã khóa revision `51b48ecd7c0629e4c79dc927a0445e6b671d8692`.
- Model control đã khóa revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Dataset đã khóa revision `406f09a45cc106a8f7b7fd0c25078883fe58cb1f`.
- Grouped holdout seed 13: train 22.691 mẫu/3.281 context groups, dev 2.866/410 groups, test 2.897/410 groups.
- Grouped holdout không chia cùng `(title, context)` giữa train/dev/test theo manifest.

## Phát hiện dữ liệu

Split `test` gốc của dataset có 7.301 dòng nhưng không có gold answer. Ngoài ra, audit raw split ghi nhận context overlap giữa train và test (449 groups) và validation và test (133 groups). Vì vậy split gốc `test` không dùng để chọn hoặc đánh giá EM/F1.

Grouped holdout được tạo từ split train có nhãn, theo hash `sha256(title + U+001F + context)`, và có manifest/hash riêng.

## Truncation audit ở max_length=512

| Model | Split | Prompt vượt 512 | Full prompt + answer vượt 512 | Mean full tokens |
|---|---:|---:|---:|---:|
| Sailor2-1B-Chat | train | 701/22.691 (3,09%) | 883/22.691 (3,89%) | 308,31 |
| Sailor2-1B-Chat | dev | 60/2.866 (2,09%) | 83/2.866 (2,90%) | 299,16 |
| Qwen2.5-0.5B | train | 701/22.691 (3,09%) | 883/22.691 (3,89%) | 308,31 |
| Qwen2.5-0.5B | dev | 60/2.866 (2,09%) | 83/2.866 (2,90%) | 299,16 |

Hai tokenizer cho cùng số token trên các split đã kiểm tra. Tỷ lệ cắt thấp nhưng phải báo cáo trong bài; không được âm thầm loại mẫu.

## Sửa pipeline trước calibration

Smoke prediction cho thấy chat model có thể sinh phần giải thích hoặc marker như `The answer is:`. Đã thêm parser dùng chung tại `code/prediction_utils.py` và sửa `code/generate_predictions.py` để lưu đồng thời:

- `raw_prediction`: continuation nguyên bản;
- `prediction`: bản đã parse dùng cho EM/F1.

Parser phải được giữ nguyên sau khi bắt đầu main run.

Calibration Sailor2 seed 13 đã hoàn tất 100 steps trên tối đa 2.048 mẫu train và 256 mẫu dev. Run không báo OOM hoặc NaN; train loss cuối là `1.798`, eval loss là `1.985`, runtime train khoảng `671.9s`. Đây chỉ là kiểm tra khả năng chạy.

Generation 32 mẫu dev sau calibration cũng chạy thành công. Với parser đã khóa, calibration đạt EM `0.34375` và F1 `0.41422`; cỡ mẫu quá nhỏ để diễn giải chất lượng.

## Artifact

- `data/splits/viquad_quality_audit.json`
- `data/splits/grouped_holdout/manifest.json`
- `results/tokenization_sailor2_train.json`
- `results/tokenization_sailor2_dev.json`
- `results/tokenization_qwen_train.json`
- `results/tokenization_qwen_dev.json`
- `environment/runtime_manifest.json`

## Bước kế tiếp

Chuyển sang baseline chính: Sailor2 với ba seed `13, 37, 73`, sau đó chạy Qwen control cùng protocol. Chỉ sau baseline mới instrument telemetry và quyết định có đủ bằng chứng để phát triển QGA-P hay không.
