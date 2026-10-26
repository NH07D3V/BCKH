# Đề tài tập trung: Quantization–Gradient Alignment làm predictor cho độ ổn định QLoRA

**Ngày cập nhật:** 26/10/2026 (Asia/Bangkok)  
**Trạng thái:** đề xuất nghiên cứu đang được thu hẹp; chưa phải tuyên bố novelty đã được xác nhận.  
**Cách hiểu mốc 2028:** đây là mốc kiểm tra relevance và research gap. Ta cần hỏi “đến cuối 2028, bài toán còn hữu dụng và còn câu hỏi chưa được giải quyết đủ không?”, không phải lập một roadmap dự báo chắc chắn.

Roadmap follow theo giai đoạn, cổng quyết định và các nhánh thuật toán nằm ở [roadmap_qga_lora_vi.md](roadmap_qga_lora_vi.md).

**Định vị:** đây là đề tài nghiên cứu thuật toán/learning dynamics. Vietnamese QA chỉ là benchmark để đo; không xây chatbot hay sản phẩm triển khai.

**Model chính đã cập nhật:** `sail/Sailor2-1B-Chat`. Đây là model đã continual-pretrain cho tiếng Việt và các ngôn ngữ Đông Nam Á, vẫn dùng causal-LM/Qwen2 stack chuẩn nên phù hợp với QLoRA trên GPU 8 GB khi giới hạn sequence length. `Qwen/Qwen2.5-0.5B-Instruct` được giữ làm **control cùng lineage** để kiểm tra độ nhạy của phương pháp khi thay đổi quy mô và mức thích nghi ngôn ngữ; đây chưa phải kiểm chứng khác family. Không dùng PhoGPT/VinaLLaMA làm pilot vì khác kiến trúc hoặc nặng hơn; chỉ mở rộng sau khi freeze protocol.

**Lưu ý về “thuật toán mới”:** hiện chưa thể tuyên bố đã phát minh optimizer mới. Ta sẽ kiểm tra QGA-P trước; chỉ khi predictor chuyển được ngoài mẫu mới mở QGA-R (rank policy) hoặc QGA-S (stability control) như một đề xuất thuật toán có điều kiện. Nếu không đạt, kết quả đúng vẫn là stability audit/negative result.

## 1. Quyết định chốt

### Tên đề tài

**English:** *Quantization–Gradient Alignment as a Predictor of QLoRA Run Stability*  
**Tiếng Việt:** *Dùng độ lệch lượng tử–gradient để dự đoán độ ổn định của QLoRA*  
**Tên làm việc:** **QGA-LoRA** (chỉ là nhãn nội bộ, chưa phải tên phương pháp đã tồn tại).

### Bài toán một câu

Với cùng một ngân sách bộ nhớ, số token, số bước và tổng rank LoRA, liệu thông tin về **quantization residual**, **gradient noise** và **gradient–residual alignment** có dự đoán được layer nào gây bất ổn cho 4-bit QLoRA, rồi dùng dự đoán đó để phân bổ rank theo layer tốt hơn uniform mà không làm tăng chi phí đáng kể không?

Đây là bài toán về **cơ chế và điều kiện hoạt động**, không phải chỉ thêm một learning rate hay trình bày một demo LLM.

Phạm vi cần giữ hẹp: QGA-LoRA là **training-time layer-risk/stability predictor cho QLoRA**. Nó không tuyên bố là phương pháp đầu tiên dùng gradient alignment, không phải post-training joint bit–rank allocation, không phải token dropping và không phải calibration cho diffusion model.

## 2. Vì sao chọn hướng này

1. **Câu hỏi thực dụng còn dài hạn:** fine-tuning mô hình dưới giới hạn VRAM vẫn cần quantization; nếu stability và chất lượng phụ thuộc vào tương tác giữa quantization với low-rank update, câu hỏi này còn đáng kiểm tra ở mốc 2028 dù các thư viện hay model cụ thể thay đổi.
2. **Khoảng trống hẹp hơn adaptive-rank chung chung:** adaptive rank đã có AdaLoRA, DyLoRA, SalientLoRA, RankAdaptor, AROMA, IGU-LoRA, TLoRA và nhiều hướng mới. Claim “một heuristic chia rank mới” sẽ yếu.
3. **Có mâu thuẫn cần giải thích:** QERA/SRR/L4Q/QA-BLoRA sửa các dạng quantization discrepancy khác nhau; QeRL lại cho thấy quantization noise có thể hữu ích trong RL; một preprint gần đây cho thấy gradient-based rank allocation có thể thất bại dưới GRPO. Điều đó gợi ý phải xác định *khi nào* tín hiệu có giá trị, thay vì giả định một proxy luôn đúng.
4. **Có đầu ra hợp lệ kể cả khi phương pháp không thắng:** ta có thể công bố một stability audit, predictor không chuyển được giữa model/task, hoặc điều kiện bác bỏ một heuristic. Đây là rủi ro khoa học có thể quản lý.
5. **Phù hợp sinh viên hơn hướng data-selection rộng:** GIST, QLESS, OPUS, BLISS, DualSFT và các phương pháp selection mới đã làm giao điểm data–optimizer khá đông. Data selection có thể là phần mở rộng sau, không phải contribution chính đầu tiên.

## 3. Literature gần nhất và điều đã biết

| Công trình | Đã làm được gì | Khoảng chưa giải quyết mà đề tài nhắm tới |
|---|---|---|
| [QLoRA, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html) | NF4, double quantization và paged optimizer cho phép fine-tune 4-bit ở quy mô lớn. | Không đặt câu hỏi predictor layer-wise cho seed variance/rank sufficiency làm trọng tâm. |
| [QERA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/21718991f6acf19a42376b5c7a8668c5-Abstract-Conference.html) | Tối ưu quantization theo output error và activation covariance, không chỉ weight error. | Chưa nối có hệ thống output residual với stochastic LoRA-gradient geometry và multi-seed stability. |
| [GuidedQuant, ICML 2025](https://proceedings.mlr.press/v267/kim25d.html) | Dùng end-loss gradients và cross-weight dependencies trong mục tiêu post-training quantization. | Làm raw gradient–quantization alignment không còn là claim mới; QGA phải khác ở predictor động trong QLoRA, seed variance và transfer. |
| [FOEM, AAAI 2026 / arXiv](https://arxiv.org/abs/2507.11017) | Đưa thành phần first-order gradient vào compensation error của post-training quantization. | Củng cố rằng gradient-aware quantization đã đông ở PTQ; khoảng QGA chỉ còn hợp lý nếu gắn với trajectory, multi-seed và frozen-base QLoRA. |
| [CLoQ, TMLR 2025](https://research.ibm.com/publications/cloq-enhancing-fine-tuning-of-quantized-llms-via-calibrated-lora-initialization) | Dùng tập calibration nhỏ để khởi tạo thành phần LoRA theo discrepancy giữa trọng số gốc và lượng tử hóa; có mã tham khảo [CLoQ](https://github.com/AozhongZhang/CLoQ). | Chưa hỏi calibration example nào chuyển được giữa seed/task, chưa đo run-to-run stability và chưa dùng predictor training-time. Phải đưa vào nhóm baseline nếu stack tương thích. |
| [L4Q, ACL 2025](https://aclanthology.org/2025.acl-long.99/) | Coupled weight–gradient quantization-aware training cho LoRA-aware QAT. | Chưa cho biết alignment/Fisher nào dự báo khi nào coupling giúp, đặc biệt trong SFT nhỏ. |
| [QA-BLoRA, TACL 2025](https://aclanthology.org/2025.tacl-1.40/) | Phân tích gradient quantization và hiện tượng adapter vừa học task vừa bù quantization error. | Chưa có protocol rộng về seed variance, rank allocation và transfer giữa bitwidth/model. |
| [SRR / Preserve-Then-Quantize, ICML 2026](https://proceedings.mlr.press/v306/cho26e.html) | Bảo toàn một subspace trước khi quantize; dùng spectrum/gradient scaling. | Các giả định spectral/proxy cần kiểm tra ngoài setting và số seed hẹp; chưa có predictor kết hợp gradient covariance. |
| [QeRL, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a15a2ece7f0663d1ba7db91103ac61c9-Abstract-Conference.html) | Quantization noise có thể tăng exploration/entropy trong RL. | Khác objective với SFT; chưa có quy luật chung cho SFT vs RL hoặc cho low-data PEFT. |
| [AQLoRA, arXiv:2608.23816](https://arxiv.org/abs/2608.23816) | Chọn layer bảo vệ theo NF4 reconstruction error để tối ưu speed/quality. | Negative controls trong paper/repo cho thấy identity layer không luôn quyết định kết quả; residual-only phải được xem là baseline, không phải giả thuyết đúng. |
| [When Gradient Importance Lies, arXiv:2605.07366](https://arxiv.org/abs/2605.07366) | Một setting GRPO cho thấy proportional gradient-rank allocation kém uniform. | Preprint chỉ có một model/task/seed; cần kiểm tra SFT, nhiều seed và quantization. |
| [μA, ICML 2026](https://proceedings.mlr.press/v306/chen26ei.html) | Cho thấy learning-rate scaling phụ thuộc rank và initialization. | Không nghiên cứu quantization residual/gradient stability; dùng làm control khi đổi rank. |
| [StableQAT, arXiv:2601.19320](https://arxiv.org/abs/2601.19320) | RDFS surrogate để giảm forward/backward mismatch trong QAT 2–4 bit. | Full-weight QAT, không phải LoRA rank allocation; dùng như optional stability control, không phải đối thủ trực tiếp. |
| [WinQ, ICML 2026](https://proceedings.mlr.press/v306/li26dr.html) | Dùng Hessian spectrum và interpolation/noise để ổn định QAT sub-4-bit. | Không nghiên cứu QLoRA rank hay multi-seed gradient predictor; scope khác cần phân biệt rõ. |
| [BALANCER, IJCAI 2026](https://ijcai.org/proceedings/2026/475) | Joint bit/rank allocation theo activation, gradient magnitude và curvature dưới budget post-training. | Không đo training-time QLoRA seed stability; QGA-LoRA phải khác ở predictor động và matched multi-seed evaluation. |
| [TokenSeek, ICLR 2026](https://arxiv.org/abs/2601.19739) | Gradient/attention-based token dropping, có hỗ trợ QLoRA. | Chọn token chứ không dự đoán layer risk hay rank allocation; cần phân biệt rõ scope. |
| [Repeatability of Fine-Tuning LLMs Illustrated Using QLoRA, IEEE Access 2024](https://ieeexplore.ieee.org/document/10700744/) | Báo cáo biến thiên giữa các lần chạy QLoRA cùng hardware/software. | Là bằng chứng nền cho stability audit, không phải predictor hay allocation method. |

### Literature bổ sung cần tính vào claim novelty

Rà soát bổ sung cho thấy các claim kiểu “gradient alignment mới”, “residual-driven rank allocation mới” hoặc “QLoRA instability chưa được biết” đều phải thu hẹp hơn:

| Công trình | Ý nghĩa đối với đề tài |
|---|---|
| [LoRA-GA](https://arxiv.org/abs/2407.05000) | Đã dùng gradient của full fine-tuning để khởi tạo LoRA; vì vậy gradient alignment không thể được tuyên bố là ý tưởng đầu tiên. |
| [LoRA-One, ICML 2025](https://proceedings.mlr.press/v267/zhang25ax.html) | Có lập luận về singular-subspace alignment và rank allocation; cần phân biệt với predictor stability ngoài mẫu. |
| [QuAILoRA, PMLR 2024](https://proceedings.mlr.press/v262/g-lawton24a.html) và [CLoQ](https://arxiv.org/abs/2501.18475) | Đã dùng quantization discrepancy/residual cho khởi tạo adapter; residual-only phải là baseline/negative control. |
| [QDyLoRA, EMNLP 2024](https://aclanthology.org/2024.emnlp-industry.53/) và [GoRA](https://arxiv.org/abs/2502.12171) | Adaptive/dynamic rank và gradient-driven allocation đã là hướng đông; không đặt novelty vào việc “chia rank theo gradient” đơn thuần. |
| [QR-Adaptor](https://arxiv.org/abs/2505.03802) | Joint per-layer bitwidth/rank allocation qua calibration; policy của chúng ta chỉ đáng gọi là mới nếu có predictive validity và transfer được chứng minh. |
| [Q-GaLore, PMLR 2025](https://proceedings.mlr.press/v280/zhang25a.html) | Đã kết hợp quantization với low-rank gradient adaptation; cần tách QGA khỏi gradient compression/optimizer. |
| [IGU-LoRA](https://arxiv.org/abs/2603.13792) | Uncertainty-aware adaptive rank đã xuất hiện; uncertainty không phải contribution riêng của đề tài. |
| [GradientStabilizer, ICML 2026](https://proceedings.mlr.press/v306/huang26bb.html) | Gradient spike stabilization đã có ở phạm vi rộng; QGA phải chứng minh giá trị đặc thù cho 4-bit QLoRA và run-to-run prediction. |

Vì vậy, gap được bảo vệ ở mức hẹp hơn là **predictive validity và failure boundary**: liệu vector feature kết hợp quantization residual, stochastic gradient geometry và alignment có dự đoán được variance/loss spike/rank sufficiency ngoài seed và ngoài model trong QLoRA SFT tiếng Việt hay không, với matched budget và accounting đầy đủ cho telemetry overhead. Đây là candidate gap cần tiếp tục backward citation search trước khi viết claim “novel”.

### Kết luận literature hiện tại

Có **bằng chứng trực tiếp** rằng quantization discrepancy, activation geometry, gradient dynamics và rank/initialization đều ảnh hưởng đến fine-tuning. Nhưng nhận định sau vẫn là **gap ứng viên cần kiểm chứng**, không phải fact “chưa ai làm”:

> Chưa thấy một protocol được kiểm tra đủ rộng để dùng vector (residual, activation covariance, stochastic gradient geometry) làm predictor ngoài mẫu cho seed variance và rank sufficiency của QLoRA dưới cùng memory/steps.

Các công trình mới có thể lấp gap trước khi nộp; vì vậy phải chạy lại backward/forward citation search ngay trước submission.

## 4. Research gap theo chuỗi lập luận

1. **Vấn đề:** 4-bit QLoRA tiết kiệm memory nhưng có thể làm thay đổi hướng update, tốc độ hội tụ và độ ổn định giữa random seeds.
2. **Cách hiện tại:**
   - QLoRA thay đổi biểu diễn/training state để chạy được low-bit.
   - LoftQ/QERA/SRR thay đổi initialization hoặc cách bảo toàn output/spectrum.
   - QA-BLoRA/L4Q đưa quantization vào objective hoặc gradient.
   - AdaLoRA/SalientLoRA/RankAdaptor phân bổ rank theo importance/spectrum.
3. **Hạn chế đã có bằng chứng:** mỗi dòng dùng một proxy khác nhau; kết quả phụ thuộc objective, bit format, calibration và model. AQLoRA còn cho thấy ranking theo reconstruction error đơn lẻ không nhất quán; GRPO preprint cho thấy gradient-based allocation có thể tự gây feedback.
4. **Phần còn thiếu hợp lý:** một đánh giá factorized, multi-seed, equal-budget xem residual, gradient noise và alignment có dự đoán *risk* hay chỉ tương quan giả.
5. **Ý nghĩa:** nếu predictor hợp lệ, ta biết lúc nào nên tăng rank/bảo vệ precision; nếu không, ta có bằng chứng chống lại một lớp heuristic và một benchmark reproducible cho cộng đồng.

## 5. Câu hỏi và giả thuyết

### RQ1 — Chẩn đoán

Trong QLoRA 4-bit, feature nào dự đoán tốt nhất layer-wise loss spike, gradient mismatch và seed-to-seed validation variance: quantization residual, gradient SNR, alignment, effective rank hay tổ hợp của chúng?

### RQ2 — Chuyển giao

Predictor học trên model pilot có giữ thứ hạng rủi ro khi đổi seed, model scale/family hoặc bitwidth trên cùng bài toán QA không? Đổi task chỉ là kiểm tra phụ.

### RQ3 — Chính sách

Với tổng rank cố định, phân bổ rank theo predicted risk có cải thiện quality–stability Pareto so với uniform và các baseline adaptive-rank ở cùng token, steps, memory và wall-clock không?

### Hypotheses

- **H1 (diagnostic):** tổ hợp residual + gradient-noise/alignment dự đoán tốt hơn residual-only; phải đánh giá out-of-sample.
- **H2 (transfer):** transfer trong cùng model family có thể tồn tại nhưng suy giảm khi đổi family/bitwidth; không giả định transfer phổ quát.
- **H3 (policy):** lợi ích của risk-aware rank allocation chỉ xuất hiện ở low-memory/low-rank regime; ở budget dư dả, uniform có thể ngang bằng.
- **H0/falsifier:** nếu feature không chuyển, hoặc policy không thắng uniform sau khi matched budget, bác bỏ claim method và giữ lại kết quả audit/negative result.

## 6. Phương pháp đề xuất (tên làm việc QGA-LoRA)

### 6.1. Giai đoạn A — predictor trước, policy sau

Không bắt đầu bằng việc tuyên bố một thuật toán mới. Trước hết xây dựng một bộ đo:

1. Quantize base model về NF4 4-bit; giữ một bản BF16 nhỏ để đo residual trên calibration layers.
2. Ở warm-up checkpoints và nhiều seeds, ghi:
   - \(E_l = W_l - Q(W_l)\), residual của layer \(l\);
   - \(R_{xx,l}\), activation covariance hoặc low-rank/diagonal sketch;
   - \(\bar G_l\) và \(\widehat{\Sigma}_{G_l}\), mean gradient và covariance/noise proxy của LoRA;
   - effective rank/spectral energy của gradient và update-to-weight ratio.
3. Tính các feature đã đăng ký:
   - residual energy \(e_l = \|E_l\|_F / (\|W_l\|_F+\epsilon)\);
   - gradient-noise scale \(n_l = \operatorname{tr}(\widehat{\Sigma}_{G_l})/(\|\bar G_l\|_F^2+\epsilon)\);
   - output-error direction \(u_l = E_lR_{xx,l}\);
   - alignment \(a_l = \cos(\operatorname{vec}(\bar G_l),\operatorname{vec}(u_l))\);
   - effective gradient rank và độ biến thiên qua seeds.
4. Không giả định alignment dương là tốt hay xấu: đăng ký trước cả signed alignment và absolute alignment, rồi chọn chỉ trên train split; không được đổi dấu sau khi xem test.
5. Dùng train/dev để fit một predictor đơn giản (linear/elastic-net hoặc small tree), rồi đánh giá trên seed/model đã hold out. Calibration data phải tách khỏi tập đánh giá cuối và không được dùng để nhìn test benchmark.

### 6.2. Giai đoạn B — policy tối giản

Chỉ khi RQ1–RQ2 cho tín hiệu ngoài mẫu, dùng predicted risk để cấp rank layer-wise dưới ràng buộc:

\[
\sum_l r_l = R_{\mathrm{total}}, \qquad r_l \in \{0,2,4,8,16,32\}.
\]

Policy đầu tiên nên là greedy allocation với risk score đã chuẩn hóa; không thêm meta-learning hay bilevel optimizer. Mọi policy phải so với uniform ở **cùng tổng rank, tokens, steps, optimizer và memory budget**.

Bitwidth 3/8-bit chỉ là stress test thứ hai. Contribution chính không phải “jointly optimize mọi thứ”, mà là kiểm tra xem geometry có thật sự giúp quyết định rank trong QLoRA hay không.

### 6.3. Optional theory

Nếu đủ thời gian, chứng minh một bound nhiễu đơn giản cho cosine/alignment khi gradient và residual bị sketch/quantize. Đây là phần tùy chọn; không hứa theoretical novelty trước khi biết bound có không-vacuous hay không.

## 7. Protocol thực nghiệm tối thiểu

### Models và data

Để tránh đề tài bị trôi thành “thử mọi model, mọi task”, ta khóa một bài toán chính trước:

- **Bài toán benchmark chính:** hỏi–đáp trích xuất tiếng Việt trên [UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0). Dataset có câu hỏi có thể trả lời và không thể trả lời, nên đo được cả chất lượng câu trả lời lẫn khả năng nhận biết không có đáp án.
- **Pilot để phát triển:** chốt **Sailor2-1B-Chat** làm model chính, NF4, sequence length 512, 3 seeds; chạy **Qwen2.5-0.5B-Instruct** như control với đúng token/step/rank budget. Chỉ dùng train/dev của UIT-ViQuAD2.0 để chọn feature, predictor, rank map và hyperparameter.
- Mặc định chạy toàn bộ train/dev; nếu tài nguyên hạn chế, dùng một subset phân tầng đã khóa trước (ví dụ 10k/2k), ghi seed/hash và không thay subset giữa các baseline.
- **Kiểm chứng chính:** sau khi khóa code và thuật toán, chạy đúng một lần trên **test held-out** của UIT-ViQuAD2.0; không dùng test để chọn feature, threshold, rank hay seed.
- **Kiểm chứng chuyển model:** sau khi khóa thuật toán trên Sailor2 và control Qwen, giữ nguyên recipe rồi chạy trên một model scale/family khác (ưu tiên Qwen2.5-1.5B hoặc PhoGPT-4B nếu đủ tài nguyên). Đây là điều kiện để nói predictor không chỉ khớp một model nhỏ.
- **Mở rộng tùy chọn sau core:** ViANLI, VLUE hoặc một tập instruction tiếng Việt chỉ dùng như domain-shift/OOD validation. Không đưa chúng vào claim chính nếu chưa hoàn thành pipeline QA.
- **Seeds:** pilot 3; kết luận/transfer 5; chỉ dùng 7–10 seed trên model nhỏ nếu cần CI ổn định hơn. Tên split chính xác, hash dữ liệu và quy tắc tiền xử lý phải được ghi trong config trước lần chạy test.

### Nhiệm vụ fine-tuning được khóa

Fine-tuning ở đây là **phương tiện tạo ra các trajectory để nghiên cứu thuật toán**, không phải bước xây chatbot. Pipeline cố định là:

```text
Sailor2-1B-Chat (primary) + Qwen control
        + UIT-ViQuAD2.0(train/dev)
        ↓
phát triển QGA-P và (nếu đủ bằng chứng) QGA-R/QGA-S
        ↓  khóa predictor, threshold, rank policy và hyperparameter
UIT-ViQuAD2.0(test held-out)
        ↓
model thứ hai để kiểm tra transfer
```

Ở mỗi ví dụ, input được chuẩn hóa thành `context + question`; target của câu trả lời được chuẩn hóa thành **text span** hoặc token đặc biệt `<NO_ANSWER>` cho câu không thể trả lời. Template prompt, tokenizer, max length và quy tắc hậu xử lý phải cố định trước pilot. Báo cáo riêng answerable EM/F1, no-answer precision/recall/F1 và overall SQuAD-style EM/F1 (hoặc script chính thức tương đương), cùng loss, seed variance và chi phí đo feature. Không biến benchmark thành demo hội thoại và không dùng câu trả lời test để chỉnh prompt.

Nếu bản HF không cho truy cập test theo đúng tên split ở thời điểm chạy, phải tạo một split held-out cố định từ dữ liệu được phép tải, ghi rõ seed/split hash và khóa nó trước khi xem kết quả; tuyệt đối không đổi split sau khi thấy số liệu.

### Baselines

1. BF16 LoRA uniform (reference dynamics).
2. QLoRA-NF4 uniform rank.
3. QLoRA + LoftQ initialization.
4. CLoQ, QERA-approx hoặc SRR tùy khả năng chạy và độ tương thích của code.
5. Residual-only allocation (để kiểm tra trực tiếp giả thuyết AQLoRA-style).
6. Gradient/spectrum-only allocation (AdaLoRA/SalientLoRA hoặc implementation tương thích).
7. QGA-LoRA ablations: residual-only, gradient-only, alignment-only, full feature set.

Không cần đưa tất cả baseline vào pilot; nhưng paper cuối phải ghi rõ baseline nào không chạy được và vì sao, không thay bằng số liệu không tương đương.

### Metrics và công bằng

- Task quality: Exact Match/F1 (hoặc script chính thức của UIT-ViQuAD2.0), mean, median, SD, bootstrap 95% CI; test held-out chỉ mở một lần sau khi freeze.
- Stability: validation-loss variance, worst-seed gap, loss-spike rate, gradient norm CV, cosine giữa gradient quantized và BF16 reference.
- Efficiency: peak GPU memory, wall-clock, throughput, tokens, optimizer steps, approximate FLOPs, preprocessing/measurement overhead; energy nếu đo được.
- Allocation: rank map, effective rank, correlation giữa risk score và per-layer loss impact; transfer Spearman/Kendall.
- Báo cáo riêng: cùng trainable parameters, cùng memory, cùng wall-clock và cùng quality target. Không gộp thành một “điểm thắng” duy nhất.

### Ablation bắt buộc

- residual-only / gradient-only / alignment-only / full;
- 3, 5, 7 seeds;
- rank tổng cố định nhưng rank map khác nhau;
- calibration/train-dev split khác evaluation/test split;
- NF4 vs INT8 (3-bit chỉ khi pilot ổn định);
- sequence length và batch/gradient accumulation;
- có/không warm-up checkpoint;
- model-family holdout.

## 8. Tiêu chí kết luận và bác bỏ

### Có thể nói “có tín hiệu” nếu

- predictor đạt tương quan ngoài mẫu và CI có ý nghĩa trên ít nhất hai điều kiện, không chỉ một seed;
- policy giữ hoặc cải thiện quality trong khi giảm SD/worst-seed gap ở cùng ngân sách;
- overhead đo feature và allocation không xoá lợi ích;
- kết quả lặp lại với model thứ hai; task OOD chỉ là kiểm tra cộng thêm, không phải điều kiện bắt buộc của vòng đầu.

### Phải bác bỏ claim method nếu

- correlation out-of-sample gần 0 hoặc đổi dấu;
- residual-only ngang hoặc tốt hơn full feature set;
- policy chỉ thắng vì dùng nhiều token/steps hoặc tuning budget hơn;
- hiệu ứng biến mất khi đổi seed/model family;
- calibration/measurement cost lớn hơn training savings;
- 4-bit QLoRA không có variance đáng kể sau CI.

Khi đó bài vẫn có thể chuyển thành **replication + negative-result/stability benchmark**, không cố giữ tên “QGA-LoRA method”.

## 9. Tài nguyên và phạm vi sinh viên

| Kịch bản | Phạm vi an toàn | Ước lượng |
|---|---|---|
| A — 8–16 GB VRAM | Sailor2-1B-Chat + Qwen2.5-0.5B-Instruct, NF4, UIT-ViQuAD2.0 train/dev, rank 4–16, 3 seeds | Pilot và audit cơ chế |
| B — thuê/cloud 24 GB | Qwen2.5-1.5B hoặc PhoGPT-4B, test held-out + transfer, 5 seeds, feature ablation | Paper domestic khả thi; khoảng 2–6 GPU-days cho mỗi vòng chính tùy cache |
| C — lab mạnh | Thêm OOD tiếng Việt, 3-bit/8-bit, model 3–7B và energy trace | Validation ngoài mẫu, không bắt buộc cho contribution đầu tiên |

Không cần bắt đầu bằng 7B. Nếu QGA-LoRA không giữ được ở 0.5–1.5B thì mở rộng model sẽ không cứu được hypothesis.

## 10. Kế hoạch nộp trong nước

- **FJCAI 2027:** trang chính thức hiện ghi hạn 15/11/2026; bài 6–8 trang IEEE, tiếng Việt hoặc tiếng Anh, phản biện kín ít nhất hai chuyên gia: [submission guide](https://fjcai.org.vn/submission/index.html). Vì deadline rất gần, chỉ nộp nếu pilot đã có bảng CI và baseline công bằng; không nên vội biến một ý tưởng chưa chạy thành claim.
- **Journal of Computer Science and Cybernetics (VAST):** bài nghiên cứu tiếng Anh, tối thiểu 8 và tối đa 15 trang theo hướng dẫn hiện tại: [author guidelines](https://vjs.ac.vn/index.php/jcc/about/submissions). Phù hợp hơn cho bản đầy đủ có predictor, ablation và negative result.
- Có thể chuẩn bị bản tiếng Việt trước cho seminar/đề tài sinh viên, sau đó viết bản English manuscript. Cần kiểm tra lại call và trạng thái venue ngay trước khi gửi.

## 11. Cổng kiểm tra relevance đến 31/12/2028

Giữ đề tài này khi cả năm điều kiện còn hợp lý:

1. Low-bit PEFT vẫn là ràng buộc thực tế trong tài nguyên hạn chế.
2. Literature mới vẫn tách rời output-error, gradient geometry và seed stability, hoặc chưa có benchmark matched-budget đủ rộng.
3. Có thể đo residual/gradient trên model và data công khai mà không cần hạ tầng độc quyền.
4. Predictor/policy còn falsifiable; nếu đã bị giải quyết, chuyển câu hỏi sang stress test, replication hoặc objective/domain shift hẹp hơn.
5. Contribution không phụ thuộc một model/version cụ thể.

Đây là **cổng quyết định**, không phải lời hứa rằng đề tài chắc chắn còn mới vào 2028.

## 12. Bước tiếp theo ngay lập tức

1. Khóa dataset card, split train/dev/test (hoặc held-out hash), template QA, preprocessing và metric của UIT-ViQuAD2.0 trong một file config.
2. Tái lập BF16 LoRA, QLoRA-NF4 và LoftQ trên **Sailor2-1B-Chat** và control **Qwen2.5-0.5B-Instruct** bằng **train/dev**, 3 seeds; chưa mở test.
3. Log \(E_l\), \(R_{xx,l}\), gradient mean/covariance và alignment ở 2–3 warm-up checkpoints; fit predictor chỉ trên train/dev.
4. Chạy residual-only và gradient-only trước khi viết policy full; chốt ngưỡng và rank policy trước khi mở test.
5. Freeze code, predictor, threshold, rank budget và seed list; chạy test held-out một lần trên model chính, sau đó lặp lại cùng recipe trên model đối chứng/transfer.
6. Chỉ mở rộng sang QGA-R/QGA-S nếu tín hiệu giữ được ngoài mẫu; nếu không, chuyển bài thành stability audit/negative result thay vì ép “phát minh optimizer mới”.

### Tài liệu chính đã kiểm tra

- [QLoRA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)
- [QERA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/21718991f6acf19a42376b5c7a8668c5-Abstract-Conference.html)
- [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html)
- [FOEM](https://arxiv.org/abs/2507.11017)
- [CLoQ](https://research.ibm.com/publications/cloq-enhancing-fine-tuning-of-quantized-llms-via-calibrated-lora-initialization)
- [L4Q](https://aclanthology.org/2025.acl-long.99/)
- [QA-BLoRA](https://aclanthology.org/2025.tacl-1.40/)
- [SRR](https://proceedings.mlr.press/v306/cho26e.html)
- [QeRL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a15a2ece7f0663d1ba7db91103ac61c9-Abstract-Conference.html)
- [AQLoRA](https://arxiv.org/abs/2608.23816)
- [When Gradient Importance Lies](https://arxiv.org/abs/2605.07366)
- [μA](https://proceedings.mlr.press/v306/chen26ei.html)

