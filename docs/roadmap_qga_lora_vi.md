# Roadmap nghiên cứu QGA-LoRA và phát triển thuật toán

**Ngày cập nhật:** 26/10/2026  
**Mục tiêu:** biến câu hỏi Quantization–Gradient Alignment thành một nghiên cứu có thể tái lập, sau đó chỉ phát triển thuật toán nếu dữ liệu thực nghiệm cho thấy một tín hiệu ổn định.  
**Mốc 2028:** checkpoint để đánh giá relevance và khoảng trống còn lại, không phải lời hứa về một roadmap chắc chắn hoặc claim novelty cố định.

**Phân biệt nghiên cứu/sản phẩm:** Vietnamese QA là bài toán benchmark để kiểm tra thuật toán. Mục tiêu là tạo tri thức, protocol và mã tái lập; không phải xây chatbot hay dịch vụ QA.

**Model chính đã cập nhật:** `sail/Sailor2-1B-Chat`. Đây là model đã thích nghi tiếng Việt/Đông Nam Á và là model chính cho baseline, QGA-P và thí nghiệm train/dev vòng đầu. `Qwen/Qwen2.5-0.5B-Instruct` được chạy song song như control cùng lineage với cùng protocol; không đổi model hoặc budget giữa một cell thí nghiệm. Model khác family chỉ được mở sau khi freeze thuật toán.

**Trạng thái thuật toán:** QGA-P là predictor/audit bắt buộc; QGA-R hoặc QGA-S chỉ trở thành thuật toán đề xuất nếu qua gate transfer và matched-budget. Không đạt gate thì không gọi là optimizer mới, mà báo cáo benchmark hoặc negative result.

## 1. Kết luận sau vòng research mới

Tên an toàn hơn của câu hỏi là:

**Quantization–Gradient Alignment as a Predictor of QLoRA Run Stability**

Tiếng Việt:

**Dùng độ lệch lượng tử–gradient để dự đoán độ ổn định của QLoRA**

Không nên gọi ngay là “một thuật toán tối ưu QLoRA”. Contribution ban đầu nên là predictor/audit; algorithm là nhánh thứ hai chỉ được mở khi predictor có khả năng chuyển giao.

### Những gì literature mới đã làm rõ

| Công trình | Hệ quả cho đề tài |
|---|---|
| [AutoQRA, ICML 2026](https://proceedings.mlr.press/v306/zhou26bm.html) | Đã joint-search bitwidth và LoRA rank dưới memory budget bằng multi-fidelity search + trust-region BO. Vì vậy không claim novelty ở “joint bit–rank allocation”. |
| [QR-Adaptor, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.779/) | Đã chạm mixed-precision + rank với calibration và Bayesian/evolutionary refinement. Đây là baseline/related work, không phải khoảng trống trực tiếp. |
| [BALANCER, IJCAI 2026](https://ijcai.org/proceedings/2026/475) | Đã dùng activation, gradient magnitude và curvature cho allocation post-training. QGA phải khác ở training-time stability và multi-seed transfer. |
| [QuantFT-VL, Algorithms 2026](https://www.mdpi.com/1999-4893/19/5/364) | Residual-aware LoRA initialization đã được nghiên cứu trong vision-language. Không claim residual-based initialization là mới. |
| [GuidedQuant, ICML 2025](https://proceedings.mlr.press/v267/kim25d.html) | Đã đưa end-loss gradient vào mục tiêu post-training quantization. QGA phải khác ở training-time QLoRA run-risk prediction, không phải gradient-aware PTQ. |
| [CLoQ, TMLR 2025](https://research.ibm.com/publications/cloq-enhancing-fine-tuning-of-quantized-llms-via-calibrated-lora-initialization) | Calibration nhỏ để khởi tạo LoRA theo quantization discrepancy; có code tham khảo. Phải dùng làm baseline nếu tương thích, vì QGA không được claim calibrated initialization là mới. |
| [Stable-LoRA, arXiv:2603.05204](https://arxiv.org/abs/2603.05204) | Đã có shrinkage để ổn định feature learning của LoRA nói chung. QGA phải chứng minh phần quantization-specific. |
| [AnLR-LoRA, arXiv:2609.05885](https://arxiv.org/abs/2609.05885) | Đã có learning rate riêng cho từng rank-one component dựa trên velocity và Adam SNR. Đây là nền tốt để mở rộng quantization-conditioned, không phải baseline có thể bỏ qua. |
| [GradientStabilizer, ICML 2026](https://proceedings.mlr.press/v306/huang26bb.html) | Đã nghiên cứu norm stabilization trên nhiều objective, kể cả FP4. QGA cần so sánh với norm control và chứng minh tín hiệu residual thêm thông tin. |
| [GradCodes, arXiv:2608.30908](https://arxiv.org/abs/2608.30908) | Đã tối ưu trực tiếp trong quantized code space. QGA không cạnh tranh ở code-space update; scope là frozen-base QLoRA và run-risk prediction. |
| [Compute-Optimal QAT, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6e1adf459d1901edf173252b6bd40cb9-Abstract-Conference.html) | Tỷ lệ FP/QAT và bitwidth ảnh hưởng kết quả dưới cùng compute. Mọi so sánh phải cố định hoặc ghi rõ lịch FP/QAT. |
| [WinQ, ICML 2026](https://proceedings.mlr.press/v306/li26dr.html) | Hessian/saddle-point dynamics là một covariate ổn định QAT, nhưng WinQ là full-weight QAT, không phải QLoRA frozen-base. |
| [TokenSeek, ICLR 2026](https://arxiv.org/abs/2601.19739) | Token dropping có thể giảm activation memory; đây là hướng khác, không được trộn vào contribution layer-risk/rank. |
| [QLoRA repeatability study, IEEE Access 2024](https://ieeexplore.ieee.org/document/10700744/) | Có cơ sở để xem seed-to-seed variance là vấn đề thực nghiệm nghiêm túc, không chỉ phụ lục. |

### Khoảng trống còn có thể bảo vệ

Đây là **candidate gap**, cần rà lại ngay trước khi nộp:

1. **Quantization-specific risk predictor:** residual, activation covariance và stochastic LoRA-gradient geometry có dự đoán được run instability ngoài mẫu không?
2. **Transfer boundary:** predictor có chuyển giữa seed, model scale/family và bitwidth hay chỉ hoạt động trong một setting? Task OOD là extension, không phải điều kiện core.
3. **Training-time control:** nếu risk dự đoán được, có thể dùng nó để điều chỉnh per-rank learning rate hoặc gradient damping dưới cùng global LR budget không?
4. **Fair protocol:** nhiều paper tối ưu quality/memory nhưng chưa cùng lúc báo uncertainty, measurement overhead, wall-clock và fixed-token fairness trong QLoRA.
5. **Negative result có giá trị:** nếu residual/gradient alignment không chuyển, ta có thể công bố điều kiện thất bại và benchmark stability thay vì ép thành method.

Không nên tuyên bố “chưa ai làm gradient alignment cho quantization”; [LoRA-GA](https://arxiv.org/abs/2407.05000) và các hướng quantization khác đã dùng ý tưởng liên quan. Claim hẹp hơn là: **đánh giá và điều khiển risk của training-time QLoRA bằng feature quantization–gradient, với cross-seed/cross-model validation trên cùng bài toán QA**. Cross-task chỉ là extension.

## 2. Câu hỏi, biến số và target

### Research questions

- **RQ1:** Feature nào dự đoán tốt nhất loss spike, gradient mismatch và validation variance?
- **RQ2:** Dự đoán có giữ được trên seed/model-scale/model-family hoặc bitwidth hold-out của cùng bài toán QA không? Cross-task là câu hỏi phụ.
- **RQ3:** Một controller nhẹ có giảm variance hoặc wall-clock regret ở cùng quality/memory không?
- **RQ4:** Lợi ích đến từ quantization-specific feature hay chỉ từ một dạng gradient norm clipping đã có?

### Biến chính

- Quantizer: NF4 4-bit là main; INT8 là robustness; 3-bit chỉ khi pilot không sập.
- Rank: total rank cố định; layer-wise map là action.
- Seed: 3 pilot, 5 main, 7–10 cho stability claim trên model nhỏ.
- Objective: SFT trước; không trộn RL/GRPO vào core paper.
- Target risk: validation-loss variance, worst-seed gap, loss-spike rate, gradient cosine mismatch.

## 3. Roadmap có cổng quyết định

| Giai đoạn | Thời lượng tương đối | Việc phải làm | Đầu ra | Cổng quyết định |
|---|---:|---|---|---|
| 0. Khóa câu hỏi | 1 tuần | Chốt model, quantizer, rank, metric, split, seed và budget; cập nhật citation matrix | Protocol 2–3 trang, config cố định, claim map | Không thêm biến mới sau khi xem test |
| 1. Dựng môi trường | 1 tuần | Pin CUDA/PyTorch/Transformers/PEFT/bitsandbytes; kiểm tra checkpoint/tokenizer; viết logging | Repo chạy được một lệnh, hash môi trường, smoke test | Nếu chưa reproducible thì chưa thu thập feature |
| 2. Replication baseline | 2–3 tuần | QLoRA uniform, BF16 LoRA, LoftQ/CLoQ; thêm Stable-LoRA/AnLR-LoRA nếu chạy được | Bảng quality/memory/time cho 3 seed | Sai lệch lớn phải giải thích trước khi nghiên cứu method |
| 3. Stability audit | 3–4 tuần | 5 seed trên model nhỏ; log residual, activation, gradient mean/covariance ở warm-up checkpoints | Stability map theo layer/rank/bit | Nếu variance gần 0, đổi câu hỏi sang transfer/efficiency; không phát minh controller |
| 4. Predictor | 3–4 tuần | residual-only, gradient-only, alignment-only, full; train/hold-out theo seed/model | Feature report, out-of-sample correlation, calibration plot | Chỉ mở algorithm nếu full feature vượt baseline và không leakage |
| 5. Algorithm branch | 4–6 tuần | QGA-RiskRank hoặc QGA-S tùy failure mechanism; ablation và overhead accounting | Prototype + controlled comparison | Nếu không đạt gate, giữ predictor/audit làm paper chính |
| 6. Transfer và stress test | 3–5 tuần | Đổi model scale/family, sequence length, seed set; task OOD và INT8/3-bit là optional | Bản đồ điều kiện hoạt động/thất bại | Cần ít nhất một hold-out model thành công để claim method |
| 7. Paper domestic | 2–4 tuần | Viết 6–8 trang, release code/config/logs, kiểm tra citation mới | Manuscript + supplementary + artifact | Chỉ claim mức mà CI và protocol hỗ trợ |

Tổng thời lượng thực tế cho một vòng paper sinh viên: khoảng **17–26 tuần**, phụ thuộc GPU và số baseline. Đây là ước lượng kế hoạch, không phải cam kết publication.

## 4. Thuật toán nào có thể phát triển?

### Nhánh A — QGA-P: predictor/audit (bắt buộc làm trước)

Đây là contribution an toàn nhất. Thu telemetry ở 1–5–10% warm-up, fit predictor đơn giản và hold out theo seed/model. So sánh loss-only, residual-only, gradient-only, alignment-only và full feature set. Chỉ được gọi là method/empirical novelty nếu predictor có calibration và transfer; nếu không, bài vẫn hợp lệ dưới dạng stability audit hoặc negative result.

### Nhánh B — QGA-R: uncertainty-aware fixed-budget rank policy

Nếu QGA-P dự đoán được layer risk nhưng tín hiệu per-rank còn nhiễu:

- giữ NF4 và tổng rank cố định;
- dùng risk cùng khoảng tin cậy hoặc ước lượng marginal gain để greedy allocate (r_l \in \{0,2,4,8,16,32\});
- chỉ cho phép một hoặc hai lần reallocation sau warm-up;
- so với uniform, AdaLoRA, TLoRA, LoRA-GA² và các baseline tương thích.

Định vị phải là **early, matched-budget, seed-aware rank allocation**, không phải joint bit–rank black-box search như AutoQRA/QR-Adaptor.

### Nhánh C — QGA-S: risk-gated stability control

Nếu audit cho thấy failure chính là gradient-norm spike, dùng predictor để chỉ kích hoạt clipping/damping hoặc [GradientStabilizer](https://proceedings.mlr.press/v306/huang26bb.html) trên layer có residual–gradient misalignment cao:

\[
g'_l = \frac{g_l}{1+\lambda \hat v_l}.
\]

So sánh no-control, always-on control, Stable-LoRA, AnLR-LoRA và selective gate. Nếu always-on GradientStabilizer thắng, QGA-S chỉ là negative composition result, không gọi là optimizer mới.

### Vai trò của AnLR-LoRA

[AnLR-LoRA](https://arxiv.org/abs/2609.05885) nên là baseline và có thể làm composition ở appendix. Không được tái đặt velocity/SNR của AnLR thành novelty riêng; phần QGA phải nằm ở quantization-specific risk và kiểm định transfer.

### Thứ tự ưu tiên

1. **QGA-P** — bắt buộc.
2. **QGA-R** — algorithm chính nếu predictor có tín hiệu.
3. **QGA-S** — chỉ mở nếu loss spike là failure mechanism rõ ràng.

Không làm cả ba nhánh cùng lúc.

## 5. Baseline và phép so sánh công bằng

### Baseline tối thiểu

- BF16 LoRA uniform.
- QLoRA-NF4 uniform.
- LoftQ và CLoQ nếu mã/stack tương thích.
- Stable-LoRA hoặc AnLR-LoRA.
- Gradient clipping/GradientStabilizer nếu controller là gradient-based.
- Residual-only và gradient-only ablation.
- QERA/SRR nếu code và setting tương thích.
- AutoQRA/QR-Adaptor/BALANCER chỉ dùng để định vị joint allocation; không nhất thiết phải tái lập toàn bộ trong pilot.

### Bốn trục phải báo cáo riêng

1. Cùng số trainable parameters.
2. Cùng peak memory.
3. Cùng token/optimizer steps.
4. Cùng wall-clock hoặc cùng quality target.

Phải cộng cả chi phí đo feature, warm-up, search và calibration. Không được gọi là “tiết kiệm” nếu preprocessing đắt hơn phần training được giảm.

## 6. Dataset, model và resource plan

### Pilot nhỏ

Ta khóa **một bài toán chính** để phát triển thuật toán: hỏi–đáp trích xuất tiếng Việt trên [UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0). Đây là benchmark nghiên cứu, không phải yêu cầu xây chatbot.

- **Model pilot chốt:** Sailor2-1B-Chat, NF4, sequence 512; Qwen2.5-0.5B-Instruct là control. Cả hai dùng cùng token/step/rank budget và 3 seed. Không dùng TinyLlama để chọn phương pháp trong pilot.
- Chỉ dùng split train/dev để chọn feature, predictor, threshold, rank budget và hyperparameter.
- Mặc định dùng toàn bộ train/dev; nếu GPU không đủ, khóa trước một subset phân tầng (ví dụ 10k/2k), lưu seed và hash, rồi dùng đúng subset đó cho mọi baseline.
- 3 seeds trong pilot; ghi hash model/tokenizer/dataset và toàn bộ config.
- 1 GPU 16–24 GB; ưu tiên chạy baseline trước khi thu telemetry.

### Main study

Main study không được mở thêm task tùy ý trước khi hoàn thành core QA:

1. **Freeze:** khóa QGA-P và, nếu có gate, QGA-R/QGA-S sau khi chỉ nhìn train/dev của model pilot.
2. **Held-out test:** chạy đúng recipe đã khóa trên test UIT-ViQuAD2.0; không chỉnh threshold/rank/seed theo test.
3. **Model transfer:** sau khi freeze, chạy cùng code và ngân sách trên Qwen2.5-1.5B hoặc PhoGPT-4B nếu đủ tài nguyên. Dùng 5 seeds nếu tài nguyên cho phép.
4. **Optional OOD:** chỉ sau hai bước trên mới thêm ViANLI, VLUE hoặc vi-Alpaca để kiểm tra domain-shift.

24 GB GPU hoặc cloud tương đương là đủ cho vòng chính; model thứ hai là kiểm chứng transfer chứ không phải một sản phẩm triển khai.

### Phân tách dữ liệu và chống leakage

| Giai đoạn | Dữ liệu được phép nhìn | Quyết định được phép đưa ra |
|---|---|---|
| Phát triển | train + dev | feature, predictor, threshold, rank map, hyperparameter, số bước, QA template |
| Freeze | không mở test | chốt commit, config, seed list và báo cáo phân tích |
| Đánh giá | test held-out | chỉ đo EM/F1, loss, variance, memory/time; không sửa thuật toán |
| Transfer | cùng recipe trên model thứ hai | kiểm tra tính chuyển giao; nếu sửa phải ghi là thí nghiệm mới |

Nếu test chính thức không tải được ở thời điểm chạy, tạo một held-out split xác định bằng seed và hash trước khi xem kết quả; không được đổi split sau khi thấy metric.

### Chỉ mở rộng sau khi có tín hiệu

- 3-bit hoặc INT8.
- Model 3–7B.
- Vietnamese domain shift, energy logging hoặc RL/GRPO.

## 7. Tiêu chí thành công và thất bại

### Gate predictor

Một ngưỡng định trước có thể là:

- correlation out-of-sample ở mức có ý nghĩa và không chỉ do một seed;
- full feature set tốt hơn residual-only/gradient-only;
- hiệu ứng giữ được trên model thứ hai với cùng task; task/OOD thứ hai là kiểm tra cộng thêm nếu còn tài nguyên.

Ngưỡng chính thức phải khóa trước pilot cuối; không chọn ngưỡng sau khi xem test.

### Gate algorithm

QGA-R hoặc QGA-S chỉ được giữ nếu:

- quality không giảm đáng kể ở cùng token/memory;
- giảm validation SD hoặc worst-seed gap;
- không tăng wall-clock/preprocessing quá mức;
- hiệu ứng lặp lại trên model thứ hai ở cùng task và matched budget.

Nếu không đạt, bài chuyển thành:

- stability benchmark;
- negative result về predictor;
- hoặc replication/transfer study.

Đó vẫn là kết quả nghiên cứu hợp lệ.

## 8. Những khoảng trống có thể phát triển tiếp đến 2028

- **Cross-quantizer transfer:** NF4, INT8, FP4/NVFP4 có cùng risk signature không?
- **Objective dependence:** SFT và RL/GRPO có đảo dấu tác động của quantization noise không?
- **Early prediction:** có thể dự đoán run cuối từ 5–10% trajectory không?
- **Uncertainty-aware controller:** controller có biết khi nào không nên can thiệp không?
- **Deployment stability:** adapter merge/dequantize/re-quantize có làm thay đổi rank-risk relationship không?
- **Domain shift:** predictor học trên QA tiếng Việt có chuyển sang Vietnamese instruction/OOD không?
- **End-to-end compute optimality:** lợi ích còn tồn tại sau khi tính warm-up, feature extraction và kernel overhead không?

## 9. Bước phải làm ngay trong 14 ngày đầu

### Ngày 1–3

- Đọc kỹ QLoRA, LoftQ, CLoQ, QERA, SRR, AnLR-LoRA, Stable-LoRA, AutoQRA.
- Tạo literature matrix gồm: objective, quantizer, rank, seeds, budget, code, limitation.
- Khóa câu hỏi RQ1 và metric stability.

### Ngày 4–7

- Cài môi trường và chạy QLoRA uniform trên Sailor2-1B-Chat cùng control Qwen2.5-0.5B với UIT-ViQuAD2.0 train/dev.
- Lưu config, seed, tokenizer, commit, peak VRAM và wall-clock.
- Chạy cùng một config hai lần để kiểm tra reproducibility.

### Ngày 8–14

- Chạy BF16 LoRA, QLoRA, LoftQ (và CLoQ nếu tương thích) với 3 seeds.
- Log \(E_l\), gradient norm, gradient cosine, loss curve.
- Vẽ bảng đầu tiên: layer risk versus final variance.
- Chưa mở test và chưa viết algorithm nếu bảng train/dev này chưa có tín hiệu.
- Nếu có tín hiệu, khóa predictor/rank policy rồi mới lập kế hoạch test held-out và model thứ hai.

## 10. Quyết định sau pilot

- **Có variance + predictor có tín hiệu:** tiếp tục QGA-R; chỉ mở QGA-S nếu loss spike là failure mechanism chính.
- **Có variance nhưng predictor yếu:** làm stability audit/negative result; thử early telemetry.
- **Không có variance:** bỏ claim stability, chuyển sang compute/wall-clock hoặc cross-model transfer.
- **Predictor tốt nhưng controller không tốt:** giữ empirical predictor, không ép thành algorithm.
- **Controller thắng một setting nhưng không transfer:** chỉ claim regime-specific, không claim tổng quát.

## 11. Cổng kiểm tra mốc 2028

Mỗi lần cập nhật literature, hỏi:

1. Có paper mới giải đúng training-time QLoRA stability predictor chưa?
2. Có baseline nào làm QGA-R/QGA-S trùng trực tiếp chưa?
3. Gap còn là stability/transfer hay đã chuyển thành deployment/merge/domain shift?
4. Code và model mở còn đủ để tái lập không?
5. Nếu gap bị lấp, câu hỏi hẹp kế tiếp là gì?

Nếu câu trả lời thay đổi, chuyển contribution sang replication, stress test hoặc failure-boundary analysis; không giữ claim novelty cũ.

### Tài liệu cập nhật chính

- [AutoQRA](https://proceedings.mlr.press/v306/zhou26bm.html)
- [QR-Adaptor](https://aclanthology.org/2026.findings-acl.779/)
- [BALANCER](https://ijcai.org/proceedings/2026/475)
- [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html)
- [FOEM](https://arxiv.org/abs/2507.11017)
- [CLoQ](https://research.ibm.com/publications/cloq-enhancing-fine-tuning-of-quantized-llms-via-calibrated-lora-initialization)
- [Stable-LoRA](https://arxiv.org/abs/2603.05204)
- [AnLR-LoRA](https://arxiv.org/abs/2609.05885)
- [GradientStabilizer](https://proceedings.mlr.press/v306/huang26bb.html)
- [GradCodes](https://arxiv.org/abs/2608.30908)
- [Compute-Optimal QAT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6e1adf459d1901edf173252b6bd40cb9-Abstract-Conference.html)
- [WinQ](https://proceedings.mlr.press/v306/li26dr.html)
- [TokenSeek](https://arxiv.org/abs/2601.19739)

