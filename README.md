# QGA-LoRA-VI: workspace nghiên cứu

Workspace này chứa toàn bộ mã, dữ liệu, cấu hình và kết quả cho đề tài:

> Tối ưu fine-tuning QLoRA cho mô hình ngôn ngữ nhỏ đã thích nghi tiếng Việt trên UIT-ViQuAD2.0, với Sailor2-1B-Chat là model chính và Qwen2.5-0.5B-Instruct là model đối chứng.

## Mục tiêu giai đoạn đầu

1. Tạo baseline QLoRA có thể tái lập.
2. Đo chất lượng tiếng Việt bằng Exact Match và token-level F1, đồng thời ghi nhận thời gian, VRAM/RAM và số bước.
3. Chỉ sau khi baseline ổn định mới thử QGA-P (dự đoán độ ổn định từ telemetry quantization/gradient), rồi mới đánh giá các chính sách rank hoặc learning-rate.

Đây là nghiên cứu thuật toán, không phải chỉ xây một chatbot. Mọi thay đổi tối ưu phải được so sánh với baseline trên cùng split, seed và ngân sách bước.

Quyết định model và ma trận so sánh được ghi chi tiết tại [docs/model_selection_sailor2_vi.md](docs/model_selection_sailor2_vi.md).

Trạng thái kiểm soát nghiên cứu hiện tại: [docs/gate1_audit.md](docs/gate1_audit.md), [docs/protocol_frozen.md](docs/protocol_frozen.md), [docs/literature_matrix.md](docs/literature_matrix.md) và [docs/experiment_log.md](docs/experiment_log.md).

## Asset chính thức

- Model chính: [sail/Sailor2-1B-Chat](https://huggingface.co/sail/Sailor2-1B-Chat), revision `51b48ecd7c0629e4c79dc927a0445e6b671d8692`
- Model đối chứng: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Dataset: [taidng/UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)
- Dataset revision đã tải: `406f09a45cc106a8f7b7fd0c25078883fe58cb1f`.

Sailor2 là model chính vì đã được continual-pretrain cho tiếng Việt và các ngôn ngữ Đông Nam Á, nhưng vẫn là causal LM theo stack Qwen2/Transformers nên phù hợp với pipeline QLoRA hiện tại. Lưu ý Sailor2 được mở rộng từ Qwen2.5-0.5B, nên Qwen là **control cùng lineage**, không phải một family độc lập. Với Sailor2, pilot dùng `max_length=512` (model config có giới hạn vị trí 4096); mọi model dùng cùng token/step budget khi so sánh. Claim chuyển giao sang family khác chỉ được đưa ra sau một vòng transfer riêng.

Script tải asset là `code/download_assets.py`. Script tải cả hai model, ghi rõ vai trò/revision và chính xác các split mà Hugging Face cung cấp vào `environment/download_manifest.json`.

## Phạm vi lưu trong Git

Repository chỉ lưu mã nguồn, cấu hình, tài liệu, manifest và các metric nhỏ để dễ kiểm soát phiên bản. Model gốc, dataset đã tải, checkpoint LoRA, log và prediction lớn được loại khỏi Git bằng `.gitignore`; có thể tái tạo chúng bằng các script trong `code/` và `scripts/`.

Lưu ý quan trọng về dữ liệu ở revision đã khóa: `train` (28.454) và `validation` (3.814) có nhãn đầy đủ, lần lượt có 9.216 và 1.161 câu không trả lời được. Split `test` có 7.301 dòng nhưng `answers=None` toàn bộ, tức là bản phát hành này là **unlabeled inference snapshot**, không thể tính EM/F1. Vì vậy code đánh dấu `has_gold_label=false`, không dùng test để chọn thuật toán và không báo cáo test như benchmark có nhãn. Chạy `code/audit_viquad.py` để xem lại overlap và chất lượng offset.

## Cấu trúc thư mục

```text
BaoCaoKhoaHoc/
├─ docs/                 # Các bản phân tích và roadmap Markdown
├─ data/
│  ├─ raw/               # Dataset lưu từ Hugging Face, không chỉnh sửa
│  ├─ processed/         # JSONL đã chuẩn hóa để train/eval
│  └─ splits/            # Metadata về split và seed
├─ models/base/          # Checkpoint gốc, không sửa
├─ code/                 # Downloader, chuẩn bị dữ liệu, train, generate, eval
├─ configs/              # Cấu hình thực nghiệm
├─ scripts/              # Lệnh chạy nhanh trên Windows PowerShell
├─ checkpoints/          # Adapter LoRA sau mỗi run
├─ results/              # Prediction, metric và bảng so sánh
├─ logs/                 # Log của từng run
├─ notebooks/            # Phân tích/visualization sau này
└─ environment/          # Requirements và manifest tải xuống
```

## Chạy theo thứ tự

Từ PowerShell:

```powershell
cd <repo-root>
python -m pip install -r environment\requirements-download.txt
python code\download_assets.py
python code\prepare_viquad.py --create-dev
python code\audit_viquad.py
python code\make_grouped_holdout.py
```

Sau khi kiểm tra GPU và cài đủ stack train:

```powershell
python -m pip install -r environment\requirements-train.txt
python code\train_qlora_baseline.py --config configs\sailor2_qlora_vi.yaml --max-steps 20 --output-dir checkpoints\smoke_sailor2_qlora_vi

# Chạy control Qwen với đúng protocol
python code\train_qlora_baseline.py --config configs\baseline_qlora_vi.yaml --max-steps 20 --output-dir checkpoints\smoke_qwen_qlora_vi

# Khi chạy protocol grouped holdout sau khi đã khóa phương pháp:
python code\train_qlora_baseline.py --config configs\sailor2_qlora_vi.yaml --train-file data\splits\grouped_holdout\train.jsonl --eval-file data\splits\grouped_holdout\dev.jsonl --output-dir checkpoints\baseline_sailor2_seed13 --seed 13
python code\generate_predictions.py --adapter checkpoints\baseline_sailor2_seed13 --data-file data\splits\grouped_holdout\dev.jsonl --split validation --output results\baseline_sailor2_seed13_dev.jsonl
python code\train_qlora_baseline.py --config configs\baseline_qlora_vi.yaml --train-file data\splits\grouped_holdout\train.jsonl --eval-file data\splits\grouped_holdout\dev.jsonl --output-dir checkpoints\baseline_qwen_seed13 --seed 13

# Chạy tuần tự toàn bộ sáu baseline, có checkpoint và log riêng
powershell -ExecutionPolicy Bypass -File scripts\run_baseline_sequence.ps1 -WaitForPid <PID_dang_chay>

# Chờ baseline hoàn tất rồi tự sinh prediction và metric trên grouped dev
powershell -ExecutionPolicy Bypass -File scripts\run_evaluation_sequence.ps1
```

`--max-steps 20` chỉ là smoke test. Không dùng kết quả smoke test để kết luận nghiên cứu. Các run chính phải cố định seed, lưu config và đánh giá trên split chưa dùng để chọn siêu tham số.

## Lưu ý tái lập

- Không đưa test (unlabeled snapshot) vào quá trình chọn prompt, rank, threshold hoặc learning rate. Nếu cần một đánh giá cuối có nhãn, phải tạo một held-out labeled split từ train/validation theo nhóm `(title, context)`, khóa hash trước khi xem metric.
- Không gọi split `test` hiện tại là “test held-out có ground truth”; nó chỉ phù hợp để xuất dự đoán/inference sau khi freeze.
- `code/make_grouped_holdout.py` tạo split có nhãn 80/10/10 theo nhóm `(title, context)` từ train, lưu manifest/hash để dùng làm đánh giá cuối mà không rò rỉ cùng một context.
- Bộ audit hiện ghi nhận một số answer offset lệch; preprocessing lưu cả offset gốc, offset đã căn chỉnh và cờ sửa, không âm thầm xóa mẫu.
- Giữ nguyên model tokenizer và commit/revision trong manifest.
- Với Windows, `bitsandbytes` có thể phụ thuộc CUDA/WSL2. Nếu 4-bit không khởi động, ghi nhận lỗi môi trường và chạy baseline fp16/bf16 trên máy Linux/GPU thay vì âm thầm đổi thiết lập.
- Unsloth chỉ là backend tăng tốc tùy chọn; kết quả chính của luận văn phải có một backend chuẩn (Transformers + PEFT + bitsandbytes) để telemetry và so sánh được rõ ràng.

