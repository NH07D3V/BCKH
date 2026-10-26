# Quyết định model: Sailor2-1B-Chat là model chính

**Ngày cập nhật:** 26/10/2026  
**Bài toán:** hỏi–đáp tiếng Việt trên UIT-ViQuAD2.0  
**Mục đích:** nghiên cứu độ ổn định và tối ưu fine-tuning QLoRA, không phải xây chatbot thương mại.

## 1. Quyết định

- **Model chính:** `sail/Sailor2-1B-Chat`
  - revision: `51b48ecd7c0629e4c79dc927a0445e6b671d8692`
  - local path: `models/base/Sailor2-1B-Chat`
  - đã continual-pretrain cho tiếng Việt và các ngôn ngữ Đông Nam Á;
  - kiến trúc `Qwen2ForCausalLM`, chạy với `Transformers + PEFT + bitsandbytes`;
  - license Apache-2.0 theo model card;
  - `max_position_embeddings=4096`, vì vậy pilot khóa `max_length=512`.
- **Model đối chứng:** `Qwen/Qwen2.5-0.5B-Instruct`
  - revision: `7ae557604adf67be50417f59c2c2f167def9a775`
  - local path: `models/base/Qwen2.5-0.5B-Instruct`
  - giữ nguyên để phân biệt tác động của thuật toán với lợi thế do pretraining tiếng Việt.

Nguồn: [Sailor2 model card](https://huggingface.co/sail/Sailor2-1B-Chat), [Sailor2 technical report](https://arxiv.org/abs/2502.12982), [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

## 1.1. Model tree có nghĩa gì?

Model tree trên Hugging Face cho thấy lineage của checkpoint:

```text
Qwen2.5-0.5B
      ↓ model expansion + continual pre-training
Sailor2-1B
      ↓ chat/instruction post-training
Sailor2-1B-Chat
```

Vì vậy, nhãn “Finetuned” trên trang model không có nghĩa Sailor2 chỉ được
fine-tune vài bước trên dữ liệu QA. Nhóm Sailor2 nói rõ họ mở rộng model từ
Qwen2.5-0.5B rồi continual-pretrain trên khoảng 500B token trước khi làm
post-training. Các mục “Adapters”, “Finetunes” và “Quantizations” bên dưới
chỉ là những checkpoint phái sinh do cộng đồng tạo từ model này; chúng không
phải các bước mà ta bắt buộc phải dùng.

Điều này làm Qwen trở thành **control cùng lineage**, chứ chưa phải control
khác họ model. So sánh raw score Sailor2 với Qwen không được dùng để tuyên bố
thuật toán tốt hơn, vì Sailor2 vừa lớn hơn vừa đã được thích nghi tiếng Việt.
Để claim QGA-P chuyển được sang family khác, sau khi freeze method ta cần một
vòng transfer riêng với PhoGPT, VinaLLaMA hoặc một model không dựa trên Qwen.

## 2. Vì sao không chỉ dùng PhoGPT?

[PhoGPT-4B-Chat](https://huggingface.co/vinai/PhoGPT-4B-Chat) là ứng viên thuần Việt đáng chú ý, nhưng lớn hơn đáng kể và dùng MPT/custom code với `trust_remote_code`. Nếu đưa vào pilot ngay, số lần chạy/seed giảm và có thêm biến nhiễu về kiến trúc, tooling và bộ nhớ. PhoGPT được dành cho một vòng transfer sau khi protocol đã freeze.

Sailor2 là thỏa hiệp phù hợp hơn: có thích nghi tiếng Việt, vẫn đủ nhỏ cho QLoRA 4-bit trên GPU 8 GB (phải xác nhận bằng smoke test), và không buộc đổi pipeline causal-LM.

## 3. Ma trận thí nghiệm tối thiểu

| Model | QLoRA chuẩn | QGA-P/QGA-R của đề tài |
|---|---|---|
| Sailor2-1B-Chat | cell A | cell B |
| Qwen2.5-0.5B-Instruct | cell C | cell D |

Mọi cell phải giữ cố định:

- UIT-ViQuAD2.0 và cùng grouped split;
- NF4, LoRA target modules, rank budget, learning rate và optimizer;
- max sequence length 512, số token/step và gradient accumulation;
- seed list, prompt template và quy tắc hậu xử lý.

Chỉ sau khi các cell train/dev được khóa mới mở held-out test hoặc model transfer.

## 4.1. Việc phải khóa trước baseline chính

- Smoke test 2 bước chỉ xác nhận pipeline, không dùng để so sánh chất lượng.
- Chạy benchmark khoảng 100 bước trên Sailor2 trước để ước tính thời gian, peak VRAM và kiểm tra OOM.
- Ghi tỷ lệ ví dụ bị cắt ở `max_length=512`; vì target được nối ở cuối chuỗi, phải biết có bao nhiêu answer bị truncation.
- Chốt một parser đầu ra duy nhất trước khi tính EM/F1. Chat model đôi khi sinh thêm phần giải thích; raw continuation và parsed answer cần được lưu riêng, nhưng parser phải giống nhau cho cả Sailor2 và Qwen.
- Dùng grouped `dev` để chọn; chỉ dùng grouped `test` sau khi đã freeze prompt, parser, hyperparameter và thuật toán.

## 5. Cách đọc kết quả

- B so với A: tác động của thuật toán trên model tiếng Việt chính.
- D so với C: kiểm tra thuật toán trên model control.
- B so với D: không được dùng để kết luận thuật toán tốt hơn; đây chủ yếu là khác biệt nền model.
- Nếu B và D cùng cải thiện: bằng chứng tốt hơn về tính chuyển giao.
- Nếu chỉ B cải thiện: kết luận bị giới hạn ở model đã thích nghi tiếng Việt, vẫn là kết quả hợp lệ nhưng không claim tổng quát.

## 6. Lệnh chạy

```powershell
cd <repo-root>
python code\verify_assets.py
python code\train_qlora_baseline.py --config configs\sailor2_qlora_vi.yaml --max-steps 20 --output-dir checkpoints\sailor2_smoke
python code\train_qlora_baseline.py --config configs\baseline_qlora_vi.yaml --max-steps 20 --output-dir checkpoints\qwen_smoke
```

`code/generate_predictions.py` tự đọc `model.local_dir` trong `run_config.json` của adapter. Có thể truyền `--base-model` khi dùng adapter cũ hoặc thư mục tùy chỉnh.

