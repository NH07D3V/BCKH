# Khảo sát định hướng nghiên cứu về tối ưu hoá fine-tuning hiệu quả (2020–25/10/2026)

**Ngày tra cứu:** 25/10/2026 (Asia/Bangkok)  
**Đối tượng:** sinh viên CNTT/DS–AI, ưu tiên thuật toán và phân tích cơ chế học.  
**Mốc đánh giá:** kiểm tra xem câu hỏi còn hữu dụng và còn research gap có thể kiểm chứng đến 31/12/2028. Đây **không** phải là cam kết phải lập một roadmap dự báo chắc chắn đến năm 2028.

> **Giải thích mốc 2028 theo yêu cầu của người dùng.** “Đến 2028” được hiểu là một *relevance/gap check*: đến thời điểm đó, bài toán còn có giá trị khoa học/thực tiễn không, và literature có còn khoảng trống đủ rõ để làm tiếp không. Nếu paper mới lấp gap trước thời điểm đó, hướng phải được chuyển sang replication, stress test hoặc một câu hỏi hẹp hơn; không giữ nguyên claim novelty.

> **Phạm vi và mức độ hoàn thành.** Đây là một khảo sát có hệ thống nhưng có giới hạn (bounded systematic survey), không phải tuyên bố đã rà soát toàn bộ Google Scholar/Scopus/IEEE/ACM. Tôi đã tra cứu nhiều truy vấn trên arXiv, ACL Anthology, PMLR, OpenReview, NeurIPS/ICLR/ICML/EMNLP/AAAI pages, Microsoft/DeepMind pages và GitHub chính thức; một số trang IEEE/ACM/Springer yêu cầu quyền truy cập nên chỉ dùng metadata/abstract khi có thể. Các kết luận mạnh chỉ dựa trên nguồn gốc có thể kiểm tra. Số liệu/claim trong paper được ghi là “tác giả báo cáo”; nhận định về gap là suy luận có độ tin cậy, không phải bằng chứng “chưa ai làm”.

## I. Executive Summary

### Kết luận ngắn

1. **PEFT/LoRA, quantization–LoRA, data selection và federated PEFT là các nhánh đông người làm.** LoRA, AdaLoRA, DyLoRA, QLoRA, DoRA, QA-LoRA, LoftQ, LQ-LoRA, LoRA+, SalientLoRA, AlphaLoRA, RankAdaptor và nhiều biến thể đã lấp phần lớn khoảng trống “thêm một cách chia rank/khởi tạo”. Các survey/benchmark gần đây cho thấy kết quả phụ thuộc mạnh vào model, task, dữ liệu và mức tuning; vì vậy claim “phương pháp mới luôn tốt hơn” thường không bền.
2. **Khoảng trống có bằng chứng đáng theo đuổi** nằm ở *đo lường và điều kiện hoạt động*: (i) liên hệ định lượng giữa quantization error, gradient geometry và run-to-run stability; (ii) early-training telemetry để dự đoán thứ hạng cấu hình PEFT dưới cùng ngân sách; (iii) tương tác giữa data selector và optimizer/rank schedule dưới cùng token/step/compute budget. Đây là các gap về kiểm chứng, khả năng khái quát và protocol, không nên trình bày như “chưa ai làm”.
3. **Ba ứng viên cân bằng nhất cho sinh viên:**
   - *Quantization Error × Gradient Geometry as Predictors of QLoRA Stability* (empirical/theoretical diagnostic; dễ thu hẹp trên 0.5–3B models).
   - *Budget-Calibrated Early Telemetry for PEFT Hyperparameter/Rank Selection* (replication + method nhẹ; có thể bắt đầu trên GLUE/SuperGLUE và SLM).
   - *Joint Data Selection and PEFT Optimization under a Fixed Compute Budget* (factorial study, sau đó mới đề xuất selector/optimizer).
4. **Đề tài nên bắt đầu bằng replication:** đánh giá lại độ bất ổn của QLoRA/LoRA qua ≥5–10 seeds và nhiều model/task, dùng protocol kiểm soát token/step/compute. Đây là đầu ra có giá trị ngay cả khi không tạo thuật toán mới.
5. **Đề tài có khả năng dẫn tới thuật toán mới:** telemetry-calibrated rank/bit allocation (chỉ sau khi chứng minh tín hiệu có khả năng dự đoán ngoài mẫu) hoặc optimizer-aware data selection; không nên bắt đầu bằng việc ghép nhiều heuristic.

### Mức tin cậy

- **Cao:** thông tin venue, abstract, phương pháp chính và code được ghi trên trang chính thức (LoRA, QLoRA, AdaLoRA, LESS, TSDS, QA-LoRA, LoftQ, GaLore, SalientLoRA, instability study).
- **Trung bình:** kết luận về độ bền xuyên model/seed chỉ được báo cáo trong một số benchmark; cần tái lập độc lập.
- **Thấp/giả thuyết:** mọi tuyên bố về lợi ích của một tín hiệu mới (gradient SNR, curvature, uncertainty) cho rank allocation hoặc HPO trước khi thử nghiệm cross-task.

## II. Bản đồ nghiên cứu và xu hướng tài liệu

### 1. Dòng phát triển chính

| Giai đoạn | Dòng phương pháp | Câu hỏi đã thúc đẩy dòng nghiên cứu | Trạng thái hiện nay |
|---|---|---|---|
| 2020–2021 | Adapter, prompt/prefix tuning, BitFit, LoRA | Có thể chỉ cập nhật phần nhỏ mà giữ chất lượng không? | LoRA trở thành baseline mặc định; chi phí thấp nhưng rank/target-module thường cố định. |
| 2022–2023 | DyLoRA, AdaLoRA, QLoRA, low-memory optimizers | Phân bổ ngân sách/rank và bộ nhớ thế nào? | Có nhiều giải pháp hiệu quả trong benchmark riêng; tính ổn định và công bằng khi so sánh còn vấn đề. |
| 2024 | DoRA, LoRA+, QA-LoRA, LoftQ, LQ-LoRA, GaLore, LESS, TSDS, SalientLoRA | Khắc phục mismatch giữa update, quantization, data và optimizer | Nhánh tăng trưởng nhanh; generic method novelty bị bão hoà hơn. |
| 2025–10/2026 | Gradient-rank optimizers, adaptive rank/bit allocation, federated heterogeneous PEFT, stability/benchmark studies | Có thể giải thích/dự đoán khi nào phương pháp hoạt động dưới budget thực tế? | Nhiều preprint/benchmark mới; cần xác minh peer review và replication. |

### 2. Những giả định phổ biến cần kiểm tra

- **Uniform rank là đủ tốt:** bị thách thức bởi AdaLoRA/SalientLoRA/AlphaLoRA/RankAdaptor, nhưng chưa có quy luật ổn định xuyên task/model.
- **Gradient norm/singular value là proxy importance:** có thể bị lệch bởi sequence length, optimizer preconditioning và noise; paper về dynamic data selection đã chỉ ra length bias và suy giảm tương quan một-step theo thời gian.
- **4-bit QLoRA giữ nguyên dynamics:** QLoRA chứng minh khả năng thực nghiệm lớn, nhưng LoftQ/QA-LoRA/QuAILoRA cho thấy quantization discrepancy và initialization ảnh hưởng quỹ đạo.
- **Kết quả một seed đại diện:** các nghiên cứu instability (ACL 2023) và repeatability QLoRA (IEEE Access 2024) cảnh báo ngược lại.
- **Thêm trainable parameters luôn giúp:** scaling study của DeepMind cho thấy PET parameter scaling nói chung có thể kém hiệu quả và phụ thuộc task/data.
- **Loss thấp là quality cao:** NeurIPS 2024 về Bayesian model fusion ghi nhận sai khác giữa loss landscape và metric landscape khi fine-tuning.

### 3. Mật độ nghiên cứu (đánh giá định tính)

- **Rất cao:** generic LoRA variants, prompt tuning, quantization recipe, data-quality filtering, federated LoRA.
- **Cao:** adaptive rank theo singular value/importance, memory-efficient optimizer.
- **Trung bình:** stability/seed variance của PEFT, cross-model transfer của rank schedule, early stopping cho PEFT.
- **Thấp hơn nhưng có rủi ro:** joint data–optimizer–rank under fixed budget; quantization error × gradient geometry × stability; theoretical sufficient-rank criteria trên model thực.

## III. Literature Review theo 11 nhánh

### A. Efficient fine-tuning và PEFT

LoRA đóng băng backbone và tham số hoá update bằng tích hai ma trận hạng thấp; paper gốc báo cáo giảm mạnh số tham số trainable và memory trên nhiều backbone ([LoRA, ICLR 2022/arXiv](https://arxiv.org/abs/2106.09685)). Survey Nature Machine Intelligence 2023 và survey 2024–2025 mô tả hàng chục phương pháp delta-tuning ([Ding et al.](https://www.nature.com/articles/s42256-023-00626-4), [Han et al.](https://arxiv.org/abs/2403.14608)).

Đã tương đối rõ: PEFT thường giảm optimizer-state memory và storage; LoRA có thể đạt chất lượng gần full fine-tuning trong nhiều task. Chưa rõ: kết quả nào còn giữ khi data nhỏ, seed thay đổi, model family đổi, hoặc so sánh ở cùng *wall-clock/FLOPs* thay vì cùng số trainable parameters. Benchmark head-to-head cho thấy nhiều phương pháp vượt LoRA trong paper gốc không còn vượt khi tuning bị hạn chế ([Lialin et al.](https://arxiv.org/abs/2303.15647)). Gap có giá trị là xây dựng protocol công bằng và tìm điều kiện thất bại, không phải thêm một adapter tuỳ tiện.

### B. Optimization algorithms cho neural networks và LLMs

Adam/AdamW là baseline thực dụng nhưng giữ momentum/variance tốn memory. LOMO hợp nhất gradient computation và parameter update để giảm memory ([ACL 2024](https://aclanthology.org/2024.acl-long.445.pdf)); AdaLomo bổ sung learning-rate thích nghi ([arXiv:2310.10195](https://arxiv.org/abs/2310.10195)). GaLore chiếu gradient hạng thấp nhưng vẫn cập nhật full parameters ([arXiv:2403.03507](https://arxiv.org/abs/2403.03507)); APOLLO, AdaRankGrad và các optimizer low-rank tiếp tục giảm state.

Đã biết: memory và training dynamics là trade-off; chiếu/ nén gradient có thể thay đổi hướng update. Còn tranh luận: khi nào giảm rank/state không làm mất generalization, rank projection nên đổi theo step hay theo layer, và các guarantee non-convex có phản ánh LLM fine-tuning thực tế không. Một optimizer mới chỉ có ý nghĩa nếu so sánh cùng tokens, steps, FLOPs, memory và tuning budget.

### C. Adaptive LoRA, rank allocation và low-rank optimization

AdaLoRA phân bổ ngân sách theo importance của singular triplets ([OpenReview ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY)); DyLoRA huấn luyện nested ranks ([EACL 2023](https://aclanthology.org/2023.eacl-main.239.pdf)); ALoRA, SalientLoRA, AlphaLoRA và RankAdaptor dùng các tín hiệu/heuristic khác nhau ([ALoRA](https://arxiv.org/abs/2403.16187), [SalientLoRA](https://proceedings.neurips.cc/paper_files/paper/2024/file/ed9f00cb7dd5fbdc2175d55e2fdf1b05-Paper-Conference.pdf), [AlphaLoRA](https://aclanthology.org/2024.emnlp-main.1141/), [RankAdaptor](https://aclanthology.org/2025.findings-naacl.321.pdf)).

Đã tương đối rõ: uniform allocation không luôn tối ưu; layer/task heterogeneity tồn tại. Chưa rõ: tín hiệu nào ổn định nhất khi *đồng thời* thay đổi seed, batch, data scale, quantization và model family; rank cần thiết có thể dự đoán trước bằng curvature/Fisher/gradient SNR hay không. Vì generic adaptive rank đã đông, novelty hợp lý hơn là causal/diagnostic study và calibration dưới fixed budget.

### D. Memory-, compute- và energy-efficient training

QLoRA dùng NF4, double quantization và paged optimizers; GaLore/LOMO giảm optimizer states; GreenTrainer và các selective-backprop methods giảm FLOPs ([QLoRA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html), [GreenTrainer](https://proceedings.iclr.cc/paper_files/paper/2024/file/b054fadf1ccd80b37d465f6082629934-Paper-Conference.pdf)).

Điểm yếu của literature là metric không đồng nhất: có paper chỉ báo trainable parameters, paper khác chỉ báo peak memory, hoặc số bước; kernel/sequence length/hardware làm thay đổi wall time. Một contribution thực tế có thể là *Pareto-front protocol* đo quality–memory–time–FLOPs–energy, nhưng novelty khoa học cần thêm phân tích cơ chế hoặc điều kiện hoạt động.

### E. Data selection, data quality và data-efficient learning

Deita chọn theo complexity/quality/diversity ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6091f2bb355e960600f62566ac0e2862-Abstract-Conference.html)); LESS dùng optimizer-aware low-rank gradient similarity và thường chọn 5% dữ liệu ([ICML 2024](https://proceedings.mlr.press/v235/xia24c.html)); Get More for Less chọn dữ liệu để dịch chuyển phân phối pretraining về target ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b36dc39b319ba6ba2a0fd7601951efb4-Paper-Conference.pdf)); TSDS dùng optimal transport + diversity/KDE ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13848b5893119ff772b69812c95914fa-Abstract-Conference.html)).

Generic selection đã đông. Gap đáng khảo sát là *joint* selector–optimizer–rank với kiểm soát token/step, và sự suy giảm utility theo training time/sequence length. Không được suy ra rằng subset nhỏ luôn tốt hơn: selection cost, target-set leakage và mismatch giữa selector model và target model có thể đảo kết quả.

### F. HPO, bilevel optimization và meta-learning

Hyperband/BOHB là nền tảng cho multi-fidelity early stopping ([Hyperband](https://www.jmlr.org/beta/papers/v18/16-558.html), [BOHB](https://proceedings.mlr.press/v80/falkner18a.html)). NeurIPS 2024 về Bayesian optimization trong fine-tuning ghi nhận loss/metric landscape discrepancy và dùng multi-objective BO ([paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/34d3cf97696022b179171e5abda42c0b-Abstract-Conference.html)).

Câu hỏi mở không phải “có thể dùng BO không”, mà là early signal nào dự đoán *Pareto ranking* của LoRA rank/LR/bitwidth dưới model/task shift, và confidence của dự đoán là bao nhiêu. Bilevel/meta-learning có chi phí outer-loop lớn; cần so sánh regret và compute với random/Hyperband/ASHA, không chỉ final score.

### G. Stability, convergence, generalization và catastrophic forgetting

Fine-tuning có thể bất ổn theo seed, nhất là data nhỏ; ACL 2023 đề xuất đo nhiều dạng instability thay vì chỉ SD ([Du & Nguyen](https://aclanthology.org/2023.acl-long.342/)). Các nghiên cứu continual PEFT dùng replay, orthogonality, Bayesian/Laplace hoặc interpolation ([I-LoRA](https://arxiv.org/abs/2402.18865), [Bayesian PEFT](https://arxiv.org/abs/2402.12220)). Sharpness không phải proxy tổng quát đáng tin cậy trong mọi setting; ICML 2023 cho thấy tương quan có thể đổi dấu và phụ thuộc measure/data ([Andriushchenko et al.](https://proceedings.mlr.press/v202/andriushchenko23a.html)).

Gap có bằng chứng: cần phân biệt quality gain với giảm variance, và đánh giá forgetting trên held-out old tasks/OOD chứ không chỉ in-domain. Một thuật toán được coi là ổn định chỉ khi có seed distribution, CI/effect size và kiểm tra robustness.

### H. Quantization, pruning, compression và fine-tuning kết hợp

QLoRA là mốc lớn; QA-LoRA dùng group-wise operators ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e6c2e85db1f1039177c4495ccd399ac4-Abstract-Conference.html)); LoftQ tối ưu khởi tạo (Q+AB^T) gần weight gốc ([ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/39ec972afab01e0d8ddc6834a9d12ac1-Paper-Conference.pdf)); LQ-LoRA kết hợp low-rank và quantized decomposition ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d97a16cb83d74195b76e0bf1e85bf072-Abstract-Conference.html)); QuAILoRA xử lý initialization ([PMLR 262](https://proceedings.mlr.press/v262/g-lawton24a.html)).

Nhánh recipe/initialization đã dày. Khoảng trống tốt hơn là đo *quantization residual, gradient variance/alignment, rank sufficiency và forgetting* trong cùng protocol; xem layer nào thực sự nhạy và liệu predictor có chuyển được giữa NF4/INT4/8-bit và model family hay không.

### I. Gradient sparsification, compression và structured updates

Gradient-mask/parameter selection (GPS, GMT), structured channel selection và gradient low-rank projection cho thấy có thể giảm update cost ([GPS](https://arxiv.org/abs/2312.10136), [GMT/AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34621), [GaLore](https://arxiv.org/abs/2403.03507)).

Vấn đề còn lại là bias do top-k norm, temporal correlation và communication/compute overhead của việc tạo mask. Đề tài sinh viên nên đo khi nào mask ổn định và khi nào gây mất coverage; tránh tuyên bố “sparse luôn tốt”.

### J. Continual learning, federated learning và resource-limited settings

Federated LoRA/heterogeneous rank và continual PEFT đang tăng nhanh (FLoRA NeurIPS 2024, [HetLoRA EMNLP 2024](https://aclanthology.org/2024.emnlp-main.717/), [LoRA-A2 ACL 2025](https://aclanthology.org/2025.acl-long.19.pdf)). Các phương pháp đã chạm tới client heterogeneity, aggregation mismatch, non-IID và communication budget.

Gap khả thi: replication dưới non-IID có kiểm soát với straggler/energy budget, hoặc liên hệ rank allocation với client gradient geometry. Không nên chọn “adaptive federated LoRA” chung chung vì đã có nhiều preprint và paper gần đây.

### K. Giao điểm optimization theory × statistical learning × efficient AI

Các giao điểm có giá trị nhất là: (i) low-rank update và gradient covariance/Fisher; (ii) quantization noise và stochastic optimization; (iii) data influence và optimizer preconditioning; (iv) multi-fidelity HPO và generalization/seed uncertainty; (v) continual adaptation và stability–plasticity. Đóng góp lý thuyết nên nhắm tới mệnh đề hẹp, có giả định rõ (ví dụ bound cho sai số projection hoặc criterion dự đoán rank trong linearized model), rồi kiểm tra xem có giải thích được mô hình thật không.

**Bằng chứng cần đặc biệt lưu ý.** GoLore (ICML 2025) cho thấy low-rank gradient compression có thể không hội tụ khi noise dị hướng nếu không có điều kiện batch/noise phù hợp; điều này bác bỏ cách diễn giải đơn giản “gradient luôn low-rank nên chiếu là an toàn”. AdaRankGrad có theorem riêng cho rank của gradient/moment, nhưng không suy ra trực tiếp rank tối ưu của LoRA weight. Do đó các đề tài optimizer phải đo covariance/noise theo layer và báo cả failure cases. Tương tự, SpIEL/SparseGrad/S²FT cho thấy sparse/structured updates có thể thắng low-rank trong một số task, nhưng index/mask overhead và kernel support làm thay đổi kết luận. So sánh sparse–low-rank chỉ có ý nghĩa khi cùng tổng memory, FLOPs, tokens và wall-clock.

### III-A. Ma trận “đã khá rõ / còn tranh luận / mật độ 2–3 năm gần đây / giao điểm”

| Nhánh | Câu hỏi đã giải quyết tương đối | Còn tranh luận/chưa rõ | Mật độ 2024–2026 | Giao điểm có giá trị |
|---|---|---|---|---|
| A PEFT | Có thể giảm tham số/state và giữ chất lượng trong nhiều task | Khi nào PEFT thua FT/OOD; seed và tuning | Rất cao | A×G stability; A×F HPO |
| B Optimizer | AdamW mạnh, optimizer memory là bottleneck; nhiều low-memory alternatives | Convergence finite-batch, transfer, equal-time | Cao | B×K theory; B×D budget |
| C Rank/LoRA | Layer heterogeneity và fixed-rank limitation là có thật | Criterion sufficient-rank, online drift, cross-family | Cao–rất cao | C×K geometry; C×H quantization |
| D Memory/compute | Quantization/projection/sharding giảm memory | Wall-clock/energy/kernel overhead | Cao | D×I sparse/low-rank; D×F multi-fidelity |
| E Data | Chất lượng/diversity/influence có thể giảm data | Long-horizon utility, length bias, selector cost | Rất cao | E×B optimizer; E×C rank |
| F HPO/bilevel | Early stopping/BO tiết kiệm trials trong nhiều setting | Curve inversion, heteroscedastic seed noise, neural assumptions | Cao | F×G uncertainty; F×E data mixture |
| G Stability/generalization | Seed variation/forgetting là hiện tượng đo được | Proxy sharpness/geometry và causal mechanism | Trung–cao | G×H quant noise; G×J continual |
| H Quant/compression | 4-bit/low-rank recipes khả thi | QAT dynamics, mixed precision, optimizer interaction | Cao | H×C rank-bit; H×B clipping |
| I Sparse/structured | Sparse/low-rank gradient có thể tiết kiệm update | Bias, error feedback, hardware realization | Trung–cao | I×D systems; I×K convergence |
| J Continual/federated | PEFT giảm communication/adapter memory; orthogonal/replay hữu ích | Unknown boundaries, privacy, stragglers, rank/data co-allocation | Cao và tăng nhanh | J×C client rank; J×E selection |
| K Theory/statistics | Có theorem trong toy/assumption-specific regimes | Finite-width Transformer, quantized stochastic dynamics | Trung | K×mọi nhánh; ưu tiên theorem hẹp + test |

## IV. Bảng các paper quan trọng

| Công trình | Năm/venue | Đóng góp trực tiếp | Hạn chế/điểm cần kiểm tra | Trạng thái & link |
|---|---:|---|---|---|
| LoRA (Hu et al.) | 2021 preprint; ICLR 2022 | Frozen base + low-rank update; rank-deficiency study | Rank/module thường cố định; benchmark cũ hơn LLM hiện nay | Published; [arXiv](https://arxiv.org/abs/2106.09685) |
| DyLoRA | 2022 preprint; EACL 2023 | Nested dynamic ranks, search-free | Cùng rank giữa layers; task/model scale hạn chế | Published; [ACL](https://aclanthology.org/2023.eacl-main.239.pdf) |
| AdaLoRA | ICLR 2023 | Importance-aware SVD rank budget | Pruning/scheduling có thể gây instability; SVD/overhead | Published; [OpenReview](https://openreview.net/pdf?id=lq62uWRJjiY) |
| QLoRA | NeurIPS 2023 | NF4, double quantization, paged optimizer + LoRA | Tác giả cảnh báo benchmark chatbot; 4-bit dynamics phụ thuộc setup | Published; [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html), DOI 10.52202/075280-0441 |
| LOMO | ACL 2024 | Fused low-memory full-parameter optimizer | SGD-like dynamics, tuning nhạy; không phải PEFT | Published; [ACL](https://aclanthology.org/2024.acl-long.445.pdf) |
| GaLore | ICML 2024 | Low-rank gradient projection, full-parameter learning | Projection rank/schedule và extra SVD cost; pretraining vs FT khác nhau | Published; [PMLR](https://proceedings.mlr.press/v235/zhao24s.html), [arXiv](https://arxiv.org/abs/2403.03507) |
| DoRA | ICML 2024 | Tách magnitude/direction để gần full FT | Extra parameters/overhead; không giải quyết rank budget tổng quát | Published; [PMLR](https://proceedings.mlr.press/v235/liu24bn.html) |
| LoRA+ | ICML 2024 | Learning rates khác nhau cho A/B, scaling argument | Ratio và transfer across tasks/model chưa universal | Published; [PMLR](https://proceedings.mlr.press/v235/hayou24a.html) |
| QA-LoRA | ICLR 2024 | Group-wise quantization/adaptation | Nhiều lựa chọn group/bit; chưa phân tích seed stability sâu | Published; [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e6c2e85db1f1039177c4495ccd399ac4-Abstract-Conference.html) |
| LoftQ | ICLR 2024 | Alternating quantization–SVD initialization | Chi phí/hiệu quả phụ thuộc target rank và quantizer | Published; [PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/39ec972afab01e0d8ddc6834a9d12ac1-Paper-Conference.pdf) |
| LQ-LoRA | ICLR 2024 | Low-rank + quantized matrix decomposition | Calibration/Fisher objective và model scale cần replication | Published; [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d97a16cb83d74195b76e0bf1e85bf072-Abstract-Conference.html) |
| LESS | ICML 2024 | Optimizer-aware low-rank gradient similarity data selection | Selector cost, target examples, transfer under optimizer/rank shift | Published; [PMLR](https://proceedings.mlr.press/v235/xia24c.html), [code](https://github.com/princeton-nlp/LESS) |
| Deita | ICLR 2024 | Quality/complexity/diversity selection | LLM scoring/corpus bias; quality may not equal long-term utility | Published; [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6091f2bb355e960600f62566ac0e2862-Abstract-Conference.html) |
| Get More for Less | ICLR 2024 | Distribution-shift-aware pre-fine-tuning selection | Conditions for optimality; mainly pre-fine-tuning setup | Published; [PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/b36dc39b319ba6ba2a0fd7601951efb4-Paper-Conference.pdf) |
| TSDS | NeurIPS 2024 | OT alignment + diversity/KDE, 1% selection experiments | Target set and representation dependence | Published; [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13848b5893119ff772b69812c95914fa-Abstract-Conference.html) |
| SalientLoRA | NeurIPS 2024 | Time-series salience for rank pruning | Extra temporal window heuristic; cross-seed/model evidence limited | Published; [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/ed9f00cb7dd5fbdc2175d55e2fdf1b05-Paper-Conference.pdf) |
| AlphaLoRA | EMNLP 2024 | HT-SR theory for layer-wise expert allocation | MoE-LoRA setting; not direct rank/optimizer guarantee | Published; DOI 10.18653/v1/2024.emnlp-main.1141, [ACL](https://aclanthology.org/2024.emnlp-main.1141/) |
| LoRA-GA | NeurIPS 2024 | Gradient-aligned LoRA initialization | First-step alignment may not predict long-run generalization | Published; [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/62c4718cc334f6a0a62fb81c4a2095a1-Paper-Conference.pdf) |
| Instability study (Du & Nguyen) | ACL 2023 | Multiple measures of seed instability | Mostly PLM tasks; needs PEFT/quantized replication | Published; DOI 10.18653/v1/2023.acl-long.342, [ACL](https://aclanthology.org/2023.acl-long.342/) |
| Scaling study (Zhang et al.) | ICLR 2024 | Joint scaling of model/data/PET params | Bilingual MT/summarization scope; not all SFT | Published; [DeepMind](https://deepmind.google/research/publications/49667/) |
| Bayesian model fusion for FT | NeurIPS 2024 | Multi-objective BO; loss vs metric discrepancy | Model-fusion-specific; outer-loop cost | Published; [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/34d3cf97696022b179171e5abda42c0b-Abstract-Conference.html) |
| FLoRA | NeurIPS 2024 | Federated heterogeneous low-rank adaptation | Communication/client assumptions; not student-low-budget by default | Published; [NeurIPS](https://papers.nips.cc/paper_files/paper/2024/hash/28312c9491d60ed0c77f7fff4ad86dd1-Abstract-Conference.html) |
| I-LoRA | 2024 preprint | Interpolation/replay for continual PEFT | Domain-specific CL and preprint status | PREPRINT; [arXiv](https://arxiv.org/abs/2402.18865) |

### IV-A. Các mốc optimizer/gradient/compression cần đưa vào baseline

| Công trình | Vì sao quan trọng cho đề tài | Hạn chế trực tiếp cần kiểm soát |
|---|---|---|
| Adafactor (ICML 2018) | Factored second moments, baseline memory thấp cho Transformer ([PMLR](https://proceedings.mlr.press/v80/shazeer18a.html)) | Factorization và clipping/decay có thể đổi dynamics; không phải LLM-scale guarantee. |
| Sophia (ICLR 2024) | Diagonal Hessian + clipped update, báo cáo ít bước hơn Adam trên GPT-2 ([OpenReview](https://openreview.net/pdf?id=3xHDeA8Noi)) | Cadence/clip ρ nhạy; bằng chứng chủ yếu pretraining, chưa đủ cho PEFT/quantized FT. |
| Lion (NeurIPS 2023) | Sign-momentum, một state, có thể giảm memory ([PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/9a39b4925e35cf447ccba8757137d84f-Paper-Conference.pdf)) | Cần retune LR/weight decay; sign update có thể làm mất magnitude information. |
| Schedule-Free AdamW (NeurIPS 2024) | Đánh giá optimizer theo self-tuning/wall-clock thay vì lịch biết trước ([PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/136b9a13861308c8948cd308ccd02658-Paper-Conference.pdf)) | Kết quả phụ thuộc protocol; không thay thế fair budget trong LLM FT. |
| AdaRankGrad (ICLR 2025) | Rank adaptive cho gradient/moment với phân tích convergence ([ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fe99ca7c293fe083c8fe7e85dce2f814-Abstract-Conference.html)) | Gradient-rank không đồng nhất với weight-update rank; assumptions cần kiểm tra trên Transformer/Adam thực. |
| GoLore (ICML 2025) | Chỉ ra GaLore không luôn hội tụ dưới noise dị hướng; đưa điều kiện batch/noise ([PMLR](https://proceedings.mlr.press/v267/he25i.html)) | Lý thuyết và counterexample cần nối với layerwise telemetry và quantized FT. |
| APOLLO (MLSys 2025) | Random low-rank projection cho scaling Adam, giảm state ([MLSys](https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf)) | Rank/projection/transfer và equal-FLOP comparison chưa được xác minh đầy đủ. |
| SpIEL/SparseGrad/S²FT | Baseline sparse/structured update để so với low-rank ([SpIEL](https://arxiv.org/abs/2401.16405), [SparseGrad](https://aclanthology.org/2024.emnlp-main.831.pdf), [S²FT PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/6e3b9fb0c0c56cf6e1ee61e6a068fca4-Paper-Conference.pdf)) | Index/mask overhead và hardware sparsity; phải đo total memory/FLOPs chứ không chỉ density. |
| DSIR/GraNd/EL2N/GRAD-MATCH | Baseline data selection trước LESS ([DSIR](https://papers.nips.cc/paper/2023/file/6b9aa8f418bde2840d5f4ab7a02f663b-Abstract.html), [GRAD-MATCH](https://proceedings.mlr.press/v139/killamsetty21a.html)) | Proxy/gradient cost và transfer target–model còn hạn chế. |
| Long Is More for Alignment | Kết quả trái chiều: top-k response dài có thể cạnh tranh selector phức tạp ([ICML 2024 PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhao24b/zhao24b.pdf)) | Length bias; mọi data-selection study phải có length/token-matched baseline. |

### IV-B. Quy tắc gán nhãn bằng chứng

Trong toàn bộ báo cáo, các nhãn sau được dùng nhất quán:

- **[PUBLISHED]**: trang proceedings/journal chính thức đã xác nhận venue và bài full/abstract.
- **[PREPRINT]**: arXiv/OpenReview bản chưa có xác nhận venue tại thời điểm snapshot.
- **[ACCEPTED-NOT-FINAL]**: chỉ dùng khi nguồn chính thức ghi rõ accepted/early-access nhưng bản xuất bản cuối hoặc volume chưa xác minh; trong bảng này không gộp nhãn đó vào [PUBLISHED].
- **[ABSTRACT-ONLY]**: chỉ đọc metadata/abstract; không suy diễn limitation hoặc kết quả ngoài phần đó.
- **[AUTHOR-CLAIM]**: số liệu được paper tác giả báo cáo; chưa kiểm chứng độc lập.
- **[INFERENCE]**: suy luận của khảo sát từ nhiều paper; phải kiểm tra bằng thí nghiệm.
- **[2028-RELEVANCE]**: đánh giá điều kiện để câu hỏi còn hữu dụng/còn gap vào 2028; không phải dự báo chắc chắn và không phải sự thật đã xác nhận.

### IV-C. Cập nhật 2025–25/10/2026 và kết quả trái chiều

Một số công trình mới làm thay đổi cách đọc gap, nhưng cần ghi đúng trạng thái:

- **GoLore (ICML 2025)** đưa ra counterexample/điều kiện cho hội tụ của GaLore dưới gradient noise dị hướng ([PMLR](https://proceedings.mlr.press/v267/he25i.html)). Điều này làm cho câu hỏi “rank projection luôn an toàn?” không còn là giả định hợp lý.
- **AdaRankGrad (ICLR 2025)** có phân tích convergence cho rank gradient giảm theo thời gian ([ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fe99ca7c293fe083c8fe7e85dce2f814-Abstract-Conference.html)); không đồng nhất kết quả này với rank của LoRA weight update.
- **PDS (ICLR 2025)** và **GTP (ICLR 2025)** mở rộng data selection theo optimal control/gradient trajectory ([PDS](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9ad4891facabf17aa11580686bacfe4e-Abstract-Conference.html), [GTP](https://openreview.net/forum?id=che9LCwPQM)). Generic “gradient selection mới” vì thế có novelty thấp hơn.
- **InstructEval (Findings NAACL 2024)** và các nghiên cứu scale cho thấy selector tự động không luôn thắng random/manual; **Long Is More for Alignment** báo top-k response dài có thể cạnh tranh selector phức tạp ([InstructEval](https://aclanthology.org/2024.findings-naacl.270/), [Long Is More](https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhao24b/zhao24b.pdf)). Vì vậy mọi đề tài data selection phải có baseline length/token-matched và tính selector overhead.
- **How to train data-efficient LLMs (ICLR 2026)** so sánh 22 kỹ thuật curation ([official page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4b8bd97555d4060104a6b84b80920bab-Abstract-Conference.html)); đây là bằng chứng nhánh data curation đã đông, không phải bằng chứng mọi phương pháp đã bão hoà.
- **AFLoRA (IEEE TMC early access 2026)** và các paper federated rank mới khiến “adaptive federated LoRA” chung chung trở thành gap yếu ([abstract/DOI](https://ieeexplore.ieee.org/abstract/document/11684902)).
- Các preprint như **LAARA**, **QR-Adaptor**, **LoRA landscape/RIP 2026** chỉ được dùng như tín hiệu xu hướng; phải đánh dấu [PREPRINT] và kiểm tra lại version/venue trước khi nộp manuscript.
- Cụm continual/federated cũng tăng nhanh trong ICLR 2026 (ví dụ [Merge before Forget](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0dedca10992d34035c9a9776254cfa8c-Abstract-Conference.html), [EWC-LoRA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/47197750b407ef48daa1438e0c0af242-Abstract-Conference.html), [LA-LoRA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34b13425b5ba8ad05d97b0043df52ed3-Abstract-Conference.html)). Vì vậy Candidate 20–23 phải được coi là benchmark/replication có điều kiện, không phải khoảng trống tuyệt đối.

## V. Danh sách đề tài sơ bộ (25)

Các đề tài dưới đây là *hypotheses to test*, không phải claim novelty đã xác nhận. Mỗi đề tài có nhánh, câu hỏi, giả thuyết, phương pháp gần nhất, thiết kế sơ bộ, chi phí và rủi ro. Dấu **[R]** là có thể bắt đầu bằng replication/benchmark; **[M]** có thể phát triển method sau khi kiểm chứng.

| # | Tên English / tên Việt | Nhánh | RQ & giả thuyết (H) | Gần nhất và khác biệt dự kiến | Thí nghiệm, baseline, dữ liệu/model, metrics | Tài nguyên/rủi ro/tái lập |
|---:|---|---|---|---|---|---|
| 1 [R] | **Seed-Stable QLoRA Audit / Kiểm toán độ ổn định seed của QLoRA** | A,G,H | RQ: variance thực tế khi giữ nguyên recipe? H: variance tăng ở data nhỏ/4-bit và phụ thuộc model. | QLoRA; IEEE repeatability study. Khác: protocol multi-task, CI, contamination check. | Qwen2.5-0.5B/1.5B, Llama/Phi nhỏ; GLUE+GSM8K subset+held-out; LoRA/QLoRA/FT; mean, SD, CI, worst-case, memory/time. | A/B; 1–3 GPU ngày. Rủi ro kết quả “không có effect”; vẫn là replication có giá trị. Code/config/seed công khai. |
| 2 [M] | **Quantization Error × Gradient Geometry for QLoRA Stability / Sai số lượng tử × hình học gradient** | G,H,K | Q: residual quantization và gradient variance/alignment có dự báo seed variance/rank sufficiency? H: layerwise predictor chuyển được giữa bitwidth/model. | QA-LoRA, LoftQ, QuAILoRA; khác ở diagnostic + predictor, không chỉ init. | 0.5–3B, NF4/INT4/8-bit, r∈{2,4,8,16}; đo residual norm, cosine/alignment, SNR, Fisher proxy, rank energy; regress/CI; compare QLoRA/LoftQ. | A/B; 2–6 GPU ngày. Rủi ro tương quan yếu; falsification vẫn bác bỏ proxy. |
| 3 [M] | **Budget-Calibrated Early Telemetry for PEFT HPO / Tín hiệu sớm hiệu chỉnh ngân sách** | F,K | Q: 5–20% trajectory có dự báo Pareto ranking cuối? H: telemetry + uncertainty giảm regret so với ASHA/Hyperband ở cùng budget. | Hyperband/BOHB, NeurIPS BO; khác: gradient/curvature + calibration cross-task. | 8–32 configs (rank, LR, target modules, bits), 5 seeds; RoBERTa/DeBERTa/SmolLM; GLUE/SuperGLUE; rank correlation, regret, AUC, budget saved. | A/B; 2–5 GPU ngày. Rủi ro early rank inversion. Tái lập bằng fixed search space. |
| 4 [M] | **Joint Data Selection–Optimizer under Fixed Budget / Đồng tối ưu chọn dữ liệu–optimizer** | B,E,F,K | Q: interaction có vượt chọn dữ liệu hoặc optimizer riêng lẻ? H: chỉ có lợi khi data scarce/non-IID. | LESS, Deita, TSDS; LoRA/AdaLoRA/AdamW. Khác: full factorial và fixed tokens/steps. | Candidate instruction corpus; random/length/Deita/LESS/TSDS × AdamW/Adafactor/AdaLoRA; Qwen2.5-0.5B/1.5B; quality, diversity, OOD, tokens/FLOPs. | B; 3–8 GPU ngày. Rủi ro selection overhead xoá lợi ích. Log selector cost. |
| 5 [R/M] | **Cross-Model Rank Allocation Transfer / Chuyển giao rank allocation giữa model** | C,G | Q: rank map học trên model nhỏ có chuyển sang model cùng/khác family? H: normalized geometry transfer tốt hơn fixed uniform. | AdaLoRA, RankAdaptor, AlphaLoRA. Khác: out-of-family transfer và uncertainty. | Pythia/OPT/Qwen small; GLUE/NLU+domain; train map on source, freeze on target; accuracy/parameter/memory. | A/B; 2–4 GPU ngày. Rủi ro không transfer; negative result hữu ích. |
| 6 [M] | **Gradient-SNR Rank Scheduler / Lịch rank theo gradient SNR** | B,C,K | Q: SNR có xác định khi tăng/giảm rank? H: hysteresis schedule giảm params mà giữ quality. | AdaLoRA/SalientLoRA/LAARA; khác: SNR + confidence, không chỉ singular value. | 0.5–3B, 5 seeds, rank budget fixed; compare static/AdaLoRA/SalientLoRA; metrics rank regret, quality, FLOPs. | B; 3–6 GPU ngày. Rủi ro SNR noisy; bootstrap CI. |
| 7 [M] | **Curvature-Aware Module Selection for LoRA / Chọn module theo curvature** | C,G,K | Q: diagonal Fisher/Hutchinson curvature dự báo module hữu ích hơn norm? H: gain per trainable parameter tăng. | GPS/GMT, AdaLoRA, gradient norm selection. | RoBERTa-large/DeBERTa-v3; GLUE; q/k/v/o/MLP subsets; accuracy, params, step time, stability. | A/B; 1–4 GPU ngày. Rủi ro curvature cost; dùng low-rank Hutchinson. |
| 8 [R] | **Uniform vs Adaptive Rank under Data Scaling / Uniform–adaptive rank theo quy mô dữ liệu** | C,E,G | Q: điểm rank crossover khi data tăng? H: adaptive chỉ thắng ở low-data/shift regime. | LoRA/AdaLoRA/DoRA; khác: scaling curve và confidence. | 1%,5%,20%,100% data; 3 models; GLUE/domain; quality vs params/tokens. | A/B; 2–5 GPU ngày. |
| 9 [M] | **Quantization-Conditioned Rank–Bit Allocation / Phân bổ rank–bit có điều kiện** | C,H,K | Q: layer nhạy quantization có cần rank cao hơn? H: joint allocation Pareto tốt hơn tách r/bit. | QA-LoRA/LoftQ/QR-Adaptor (preprint); khác: predictor + ablation. | INT4/8/NF4, r budget; Qwen/Phi; residual + Fisher; quality/memory/time. | B; 4–10 GPU ngày. Rủi ro gần phương pháp 2026; phải kiểm tra literature trước. |
| 10 [R] | **LoRA+ Ratio Robustness / Độ bền tỉ lệ learning-rate LoRA+** | B,C | Q: ratio A/B có phụ thuộc width/task/seed? H: scaling law dự đoán vùng ổn định. | LoRA+; khác: systematic sensitivity. | RoBERTa, T5/SLM; grid ratio×rank×batch; quality/speed/variance. | A; <2 GPU ngày. |
| 11 [M] | **Adaptive Gradient Clipping for Quantized PEFT / Clipping thích nghi cho PEFT lượng tử** | B,G,H | Q: local clip theo quantization noise có giảm outlier/variance? H: ổn định hơn global clipping ở 4-bit. | AdaGC/AGGC (preprint), QLoRA; khác: quantization-aware threshold. | QLoRA 4/8-bit, 5 seeds; loss spikes, grad norms, quality. | B; 2–5 GPU ngày. Rủi ro tuning threshold. |
| 12 [R] | **GaLore vs LoRA at Equal FLOPs / So sánh GaLore–LoRA cùng FLOPs** | B,D,I | Q: kết luận memory/quality đổi thế nào khi equalize FLOPs/tokens? H: Pareto frontier phụ thuộc batch/sequence. | GaLore, LoRA, LOMO, AdamW. | RoBERTa/1B; GLUE/C4 slice; peak memory, FLOPs, throughput, quality. | A/B; 2–6 GPU ngày. |
| 13 [M] | **Projection-Rank Schedule for GaLore / Lịch rank chiếu cho GaLore** | B,I,K | Q: rank projection nên theo gradient spectrum hay fixed? H: adaptive schedule giảm memory không mất convergence. | GaLore/APOLLO/AdaRankGrad; khác: same compute protocol. | 1B LM, C4 small; rank schedule ablation; perplexity/quality/memory. | B/C; 5–15 GPU ngày. Rủi ro gần literature mới. |
| 14 [R] | **Length-Normalized Influence Selection / Chọn dữ liệu influence đã chuẩn hoá độ dài** | E,K | Q: length bias ảnh hưởng selector bao nhiêu? H: normalization cải thiện utility trong long-form SFT. | LESS; dynamic influence paper 2025. | Alpaca/OpenHermes subset; 0.5–1.5B; random/LESS/normalized; quality by length bins. | A/B; 1–3 GPU ngày. |
| 15 [M] | **Online Decaying Influence for Multi-Stage FT / Influence suy giảm theo thời gian** | E,F,G | Q: một-step influence giảm tương quan theo training; schedule refresh có lợi? H: refresh adaptive giảm compute. | Dynamic data selection 2025, DELIFT; khác: refresh policy + optimizer state. | 3 stages, 5 seeds, fixed token budget; performance trajectory/selector overhead. | B; 3–8 GPU ngày. |
| 16 [M] | **Data–Rank Co-Selection / Đồng chọn mẫu và rank** | C,E,K | Q: sample difficulty/gradient diversity có liên hệ rank cần thiết? H: hard diverse subset cần rank cao hơn nhưng tổng budget tối ưu joint. | LESS/TSDS + AdaLoRA; khác: interaction model. | Candidate data × rank grid, 0.5–3B; Pareto quality/params/tokens. | B; 4–10 GPU ngày. |
| 17 [R] | **Early Loss vs Final PEFT Ranking Replication / Replication tương quan loss sớm–xếp hạng cuối** | F,G | Q: one-epoch/20% ranking có đáng tin không? H: calibration phụ thuộc task. | Hyperband, one-epoch HPO preprints; khác: open configs/seeds. | 50–100 short runs trên SLM; Spearman/Kendall/regret. | A/B; 1–4 GPU ngày. |
| 18 [M] | **Uncertainty-Aware Multi-Fidelity HPO / HPO đa fidelity có bất định** | F,G,K | Q: seed variance nên vào surrogate thế nào? H: heteroscedastic BO chọn config tốt hơn mean-only. | BOHB/ASHA/NeurIPS BO; khác: variance-aware acquisition. | 20–40 configs, ≥5 seeds subset; quality–variance Pareto. | B; 4–10 GPU ngày. |
| 19 [R] | **Hyperparameter Transfer across SLM Families / Chuyển HPO giữa họ SLM** | F | Q: rank/LR priors có chuyển được? H: normalized width/depth meta-features giảm trials. | PEFT surveys/scaling study; khác: transfer protocol. | Phi/Qwen/SmolLM/Pythia; GLUE/instruction; regret/trials. | A/B; 2–5 GPU ngày. |
| 20 [R] | **Continual LoRA Stability with Memory Budget / Ổn định continual LoRA dưới memory budget** | G,J | Q: replay/orthogonality/interpolation trade-off thế nào? H: rank-aware replay tốt hơn fixed. | I-LoRA, Bayesian PEFT, InfLoRA. | 4–8 domain sequence, 0.5–1.5B; average accuracy, BWT/FWT, memory. | A/B; 2–6 GPU ngày. |
| 21 [M] | **Rank Allocation for Continual Stability–Plasticity / Rank cho stability–plasticity** | C,G,J | Q: rank theo task uncertainty giảm forgetting? H: allocate capacity ở module ít interference. | I-LoRA/continual PEFT; khác: gradient interference criterion. | DomainNet/text CL, 5 seeds; forgetting/interference/params. | B; 4–10 GPU ngày. |
| 22 [R] | **Federated LoRA under Stragglers / Federated LoRA với client chậm** | J,D | Q: adaptive rank có còn lợi khi straggler/heterogeneity? H: communication Pareto phụ thuộc rank schedule. | FLoRA/HetLoRA/LoRA-A2; khác: reproducible simulator. | FEMNIST/AG News/StackOverflow slice; 10–50 clients; accuracy, rounds, bytes, energy proxy. | A/B; simulator, 2–5 GPU ngày. |
| 23 [M] | **Client-Gradient Geometry Rank Allocation / Rank theo hình học gradient client** | J,C,K | Q: consensus/subspace overlap dự báo rank per client? H: rank allocated theo overlap giảm bytes mà giữ global quality. | FLoRA/heterogeneous LoRA; khác: geometry + theory. | Non-IID partitions, 0.5–1B; cosine/subspace, communication, accuracy. | B/C; 5–15 GPU ngày. |
| 24 [R] | **Selective Backprop Stability / Ổn định selective backprop** | D,I,G | Q: dynamic mask có thay đổi generalization/forgetting? H: resampling masks giảm bias so với static. | GreenTrainer/TraDy/GMT; khác: multi-seed/OOD. | RoBERTa/ViT small; GLUE/VTAB; FLOPs, quality, OOD. | A/B; 2–5 GPU ngày. |
| 25 [M] | **Low-Rank Update Generalization Bound in Linearized Regime / Bound tổng quát cho update hạng thấp** | K,C,G | Q: có bound theo projected gradient/Fisher và rank không? H: bound dự đoán rank crossover trong NTK/linearized model. | LoRA expressivity theory, LoRA-One, COLT dynamics; khác: link empirical predictor. | Toy + small Transformer; derive/prove then test rank/quality correlation. | A/B; compute thấp nhưng rủi ro lý thuyết cao. |

### Ghi chú về novelty

Các mục 2, 3, 4, 9, 15, 16, 18, 21, 23, 25 chỉ nên gọi là *candidate gap*. Trước khi viết paper phải chạy forward/backward citation search và kiểm tra các preprint 2025–2026 (ví dụ LAARA, QR-Adaptor, adaptive rank/bit papers). Một kết quả “tôi không tìm thấy” không đủ để tuyên bố mới.

### V-A. Topic cards (đủ trường thông tin cho 25 đề tài)

Để bảng V không quá rộng, các card dưới đây ghi rõ 18 trường bắt buộc theo mẫu: **RQ/H; gap/evidence; nearest/difference; possible contribution; experiment/baseline/data/metrics; resources; risk/falsifier; reproducibility; 2028 relevance/gap check**. Mục “2028” chỉ là điều kiện cần kiểm tra lại, không phải lời hứa về một hướng phát triển chắc chắn. “Gap” ở đây luôn là gap ứng viên, không phải tuyên bố tuyệt đối.

**T1 — Seed-Stable QLoRA Audit.** RQ/H: cùng recipe QLoRA có variance đáng kể theo seed, tăng khi data nhỏ/4-bit. Gap/evidence: ACL 2023 đo instability; QLoRA paper không đặt stability làm trọng tâm; cần replication. Gần nhất: QLoRA và IEEE Access repeatability; khác ở multi-task, CI và contamination checks. Đóng góp: benchmark/recommendation về số seed tối thiểu. Thí nghiệm: Qwen/Pythia 0.5–1.5B, GLUE + math/instruction holdout; LoRA/QLoRA/FT; mean/SD/CI/worst-case/quality–memory–time. A/B, 1–3 GPU-days. Bác bỏ: CI nhỏ và không có effect. Tái lập: pin seed/model/data hash, 7–10 runs. 2028: stability-aware PEFT benchmark.

**T2 — Quantization Error × Gradient Geometry.** RQ/H: residual quantization kết hợp gradient SNR/alignment/Fisher dự đoán seed variance và sufficient rank; H predictor chuyển được một phần xuyên bit/model. Gap/evidence: LoftQ/QA-LoRA/QuAILoRA chứng minh discrepancy/init quan trọng; chưa có predictor out-of-sample. Gần nhất: các phương pháp trên; khác ở diagnostic và policy rank–bit. Đóng góp: empirical predictor, có thể thêm bound surrogate. Thí nghiệm: NF4/INT4/8, r 2–16, 0.5–3B, 3 task types; compare QLoRA/LoftQ/QA-LoRA; residual, alignment, SNR, effective rank, quality/OOD/memory/FLOPs. B/C, 2–6 GPU-days. Bác bỏ: correlation/transfer gần 0 hoặc policy không thắng uniform. Tái lập: preregister feature/split, bootstrap. 2028: online rank–bit controller.

**T3 — Budget-Calibrated Early Telemetry.** RQ/H: 5–20% trajectory dự báo Pareto ranking cuối; H telemetry + uncertainty giảm regret ở cùng GPU-hour. Gap/evidence: Hyperband/BOHB có early stopping; LC-PFN/one-epoch evidence trái chiều; chưa có PEFT rank/quantization transfer có calibration. Gần nhất: ASHA/BOHB/NeurIPS BO; khác ở gradient/curvature telemetry và held-out task. Đóng góp: benchmark rank inversion + calibrated policy. Thí nghiệm: 32–100 configs, 5 seeds, RoBERTa/SmolLM/Qwen; Random/ASHA/BOHB/BO; Spearman/Kendall, regret, ECE, Pareto area. A/B, 2–5 GPU-days. Bác bỏ: false-stop cao hoặc không tiết kiệm compute. Tái lập: fixed search space/commit. 2028: transfer meta-surrogate.

**T4 — Joint Data Selection–Optimizer.** RQ/H: selector và optimizer/rank có interaction dưới low-data/non-IID; H joint policy vượt additive baselines. Gap/evidence: LESS đã optimizer-aware nhưng thường fixed adapter; Deita/TSDS/DELIFT khác mục tiêu; chưa full-factorial equal-budget. Gần nhất: LESS/Deita/TSDS; khác ở interaction và overhead accounting. Đóng góp: interaction effect + safe operating region, method chỉ sau pilot. Thí nghiệm: random/length/Deita/LESS/TSDS × AdamW/Adafactor/AdaLoRA; 0.5–1.5B, instruction corpus; in/OOD/forgetting/tokens/FLOPs. B/C, 3–8 GPU-days. Bác bỏ: không interaction hoặc selector cost xoá savings. Tái lập: log selection cost và all cells. 2028: bilevel data–rank controller.

**T5 — Cross-Model Rank Allocation Transfer.** RQ/H: rank map từ model nhỏ chuyển được sang cùng/khác family; H normalized geometry tốt hơn uniform. Gap/evidence: AdaLoRA/RankAdaptor báo task-specific allocation, LESS báo selector transfer; rank-map transfer ít được kiểm tra có hệ thống. Gần nhất: AdaLoRA/AlphaLoRA/RankAdaptor; khác ở leave-one-family-out. Đóng góp: transferability map/negative boundary. Thí nghiệm: Pythia/OPT/Qwen/DeBERTa, GLUE/domain; static vs transferred map; quality/params/memory/variance. A/B, 2–4 GPU-days. Bác bỏ: transfer không hơn uniform. Tái lập: source/target split công khai. 2028: meta-learned rank prior.

**T6 — Gradient-SNR Rank Scheduler.** RQ/H: SNR và confidence quyết định grow/prune; H hysteresis schedule giảm params/compute mà giữ quality. Gap/evidence: AdaLoRA/SalientLoRA dùng importance/spectrum; GoRA/AdaRankGrad cho geometry khác; chưa có SNR controller thống nhất. Gần nhất: AdaLoRA/SalientLoRA/AdaRankGrad; khác ở uncertainty/hysteresis. Đóng góp: controller + ablation. Thí nghiệm: 0.5–3B, 5 seeds, fixed rank budget; quality, rank regret, FLOPs, switch cost. B, 3–6 GPU-days. Bác bỏ: SNR noisy/overhead lớn. Tái lập: EMA/window fixed. 2028: continual drift adaptation.

**T7 — Curvature-Aware Module Selection.** RQ/H: Fisher/Hutchinson curvature chọn module tốt hơn gradient norm; H gain/parameter cao hơn. Gap/evidence: GPS/GMT/norm methods; sharpness literature cảnh báo proxy phụ thuộc parametrization. Gần nhất: GPS/GMT/AdaLoRA; khác ở second-order and OOD. Đóng góp: calibrated module selector hoặc negative result. Thí nghiệm: RoBERTa/DeBERTa, GLUE/SuperGLUE; q/k/v/o/MLP subsets; accuracy/OOD/curvature cost. A/B, 1–4 GPU-days. Bác bỏ: no gain after cost. Tái lập: Hutchinson seeds/CI. 2028: low-cost block curvature.

**T8 — Uniform vs Adaptive Rank under Data Scaling.** RQ/H: crossover rank phụ thuộc data quantity; H adaptive chỉ thắng low-data/shift. Gap/evidence: scaling study cho thấy task/data dependence; adaptive-rank papers ít vẽ scaling curves. Gần nhất: LoRA/AdaLoRA/DoRA; khác ở data fractions and matched tokens. Đóng góp: data–rank scaling law. Thí nghiệm: 1/5/20/100% data × 3 models/tasks; score/params/variance. A/B, 2–5 GPU-days. Bác bỏ: uniform luôn tương đương. Tái lập: fixed sampler/seed. 2028: principled budget estimator.

**T9 — Quantization-Conditioned Rank–Bit Allocation.** RQ/H: layer quantization sensitivity cần rank cao; H joint allocation Pareto tốt hơn separated. Gap/evidence: QA-LoRA/LoftQ/QR-Adaptor chạm interaction nhưng protocol/rank-bit theory chưa thống nhất. Gần nhất: QA-LoRA/LoftQ/QR-Adaptor (preprint status phải kiểm tra); khác ở predictor/fair budget. Đóng góp: discrete allocator. Thí nghiệm: NF4/INT4/8 × rank/bit budgets, Qwen/Phi; quality/memory/latency. B, 4–10 GPU-days. Bác bỏ: no Pareto gain. Tái lập: integer budget table. 2028: hardware-aware allocator.

**T10 — LoRA+ Ratio Robustness.** RQ/H: tỉ lệ LR A/B phụ thuộc width/task/seed; H scaling law xác định vùng ổn định. Gap/evidence: LoRA+ báo gain nhưng ratio transfer chưa universal. Gần nhất: LoRA+; khác ở sensitivity/width sweep. Đóng góp: practical ratio prior or null result. Thí nghiệm: RoBERTa/T5/SLM, ratio×rank×batch; score/speed/variance. A, <2 GPU-days. Bác bỏ: ratio effect không đáng kể. Tái lập: grid + 5 seeds. 2028: joint ratio–rank scheduler.

**T11 — Quantization-Aware Adaptive Clipping.** RQ/H: local clipping theo residual/SNR giảm spikes và variance ở 4-bit; H tốt hơn global clipping. Gap/evidence: AdaGC/AGGC và QAT papers tách optimizer/quantization. Gần nhất: AdaGC/AGGC/QLoRA; khác ở threshold quantization-conditioned. Đóng góp: clip rule + stability analysis. Thí nghiệm: QLoRA 4/8-bit, 5 seeds; loss spikes/grad norms/quality. B, 2–5 GPU-days. Bác bỏ: clipping không cải thiện hoặc làm bias. Tái lập: threshold formula fixed. 2028: mixed-precision optimizer.

**T12 — GaLore vs LoRA at Equal FLOPs.** RQ/H: Pareto ranking đổi khi equalize FLOPs/tokens; H memory winner không luôn time winner. Gap/evidence: GaLore/LOMO thường nhấn mạnh memory; projection overhead chưa chuẩn hoá. Gần nhất: GaLore/LOMO/AdamW; khác ở equal-FLOP/throughput. Đóng góp: fair benchmark. Thí nghiệm: RoBERTa/1B, GLUE/C4 slice; peak memory/FLOPs/throughput/quality. A/B, 2–6 GPU-days. Bác bỏ: same ordering mọi budget. Tái lập: kernel/hardware logs. 2028: systems-aware optimizer selection.

**T13 — Projection-Rank Schedule for GaLore.** RQ/H: rank/refresh theo spectrum/noise tốt hơn fixed; H giảm state mà không mất convergence. Gap/evidence: GaLore/APOLLO/AdaRankGrad có heuristics/assumptions; GoLore nêu failure dưới anisotropic noise. Gần nhất: GaLore/GoLore/APOLLO; khác ở measured noise controller. Đóng góp: schedule/theorem on quadratic surrogate. Thí nghiệm: 1B C4/GLUE; PPL/quality/memory/refresh cost. B/C, 5–15 GPU-days. Bác bỏ: no stable schedule. Tái lập: projection seed/refresh logs. 2028: convergence-aware controller.

**T14 — Length-Normalized Influence Selection.** RQ/H: influence bị length bias; normalization cải thiện utility long-form. Gap/evidence: LESS và dynamic-selection paper nêu variable-length issue/one-step decay. Gần nhất: LESS; khác ở token-normalized, length-bin analysis. Đóng góp: safe normalization and bias audit. Thí nghiệm: Alpaca/OpenHermes, 0.5–1.5B; random/length/LESS/normalized; quality by length/OOD. A/B, 1–3 GPU-days. Bác bỏ: normalization không đổi hoặc hại. Tái lập: tokenization/version pinned. 2028: adaptive influence normalization.

**T15 — Online Decaying Influence.** RQ/H: one-step utility giảm theo training; refresh schedule adaptive có lợi; H giảm selector cost và giữ quality. Gap/evidence: dynamic data-selection paper trực tiếp chỉ ra correlation decay; DELIFT multi-stage chưa cùng objective. Gần nhất: dynamic influence/DELIFT; khác ở optimizer-state-aware refresh. Đóng góp: refresh policy + long-horizon benchmark. Thí nghiệm: 3-stage FT, fixed tokens, 5 seeds; trajectory/utility/overhead. B, 3–8 GPU-days. Bác bỏ: no decay/effect. Tái lập: refresh intervals preregistered. 2028: continual data controller.

**T16 — Data–Rank Co-Selection.** RQ/H: subset difficulty/diversity liên hệ rank cần thiết; H joint allocation đạt same quality với fewer tokens/params. Gap/evidence: selectors and rank allocators mostly independent; JI²S/LESS higher-order influence are nearest. Gần nhất: LESS/TSDS + AdaLoRA; khác ở joint budget. Đóng góp: interaction model. Thí nghiệm: data×rank grid, 0.5–3B; Pareto quality/params/tokens. B, 4–10 GPU-days. Bác bỏ: additive/no transfer. Tái lập: full factorial subset. 2028: bilevel joint optimization.

**T17 — Early Loss/Final Ranking Replication.** RQ/H: one epoch/20% ranking có đáng tin không; H task-dependent, false-stop measurable. Gap/evidence: Hyperband, PASHA, LC-PFN và ESANN one-epoch result trái chiều; chưa đủ LoRA rank/quantization. Gần nhất: ASHA/LC-PFN; khác ở PEFT config curves. Đóng góp: curve dataset/false-stop taxonomy. Thí nghiệm: 50–100 short runs SLM; Spearman/regret/false-stop. A/B, 1–4 GPU-days. Bác bỏ: ranking ổn định mọi task. Tái lập: curve archive. 2028: open PEFT learning-curve benchmark.

**T18 — Uncertainty-Aware Multi-Fidelity HPO.** RQ/H: surrogate biết seed variance chọn config tốt hơn mean-only; H giảm simple regret. Gap/evidence: BO/Hyperband methods thường coi noise đơn giản; instability papers cho thấy heteroscedasticity. Gần nhất: BOHB/NeurIPS BO/LC-PFN; khác ở variance-aware acquisition. Đóng góp: calibrated HPO policy. Thí nghiệm: 20–40 configs, 5 seeds subset; quality–variance Pareto. B, 4–10 GPU-days. Bác bỏ: no regret benefit. Tái lập: fixed acquisition/search seeds. 2028: meta-HPO across families.

**T19 — Hyperparameter Transfer across SLM Families.** RQ/H: priors rank/LR chuyển theo depth/width/meta-features; H giảm trials. Gap/evidence: DeepMind scaling study cho thấy task/data dependence; transfer evidence chưa systematic. Gần nhất: scaling study/PEFT surveys; khác ở leave-one-family-out HPO. Đóng góp: transferability map. Thí nghiệm: Phi/Qwen/SmolLM/Pythia × GLUE/instruction; regret/trials. A/B, 2–5 GPU-days. Bác bỏ: transfer worse than random. Tái lập: public search traces. 2028: learned prior library.

**T20 — Continual LoRA Stability under Memory Budget.** RQ/H: replay/orthogonality/interpolation trade-off; H rank-aware replay giữ old tasks với bounded adapter memory. Gap/evidence: I-LoRA, Bayesian PEFT, InfLoRA, CoIN/L2R có assumptions khác nhau. Gần nhất: I-LoRA/InfLoRA; khác ở fixed memory and task-order shift. Đóng góp: memory–forgetting curve. Thí nghiệm: 4–8 domain sequence, 0.5–1.5B; ACC/BWT/FWT/adapter memory. A/B, 2–6 GPU-days. Bác bỏ: no forgetting reduction. Tái lập: task orders/seed table. 2028: unknown-boundary controller.

**T21 — Rank for Continual Stability–Plasticity.** RQ/H: gradient conflict/uncertainty phân bổ rank vào module ít interference; H cải thiện BWT tại cùng params. Gap/evidence: continual PEFT và rank allocation hiếm khi joint. Gần nhất: I-LoRA/orthogonal projection/AdaLoRA; khác ở rank controller. Đóng góp: conflict-aware rank schedule. Thí nghiệm: text CL/CoIN subset, 5 seeds; forgetting/interference/params. B, 4–10 GPU-days. Bác bỏ: rank allocation không ảnh hưởng BWT. Tái lập: conflict metric logs. 2028: continual multi-adapter composition.

**T22 — Federated LoRA with Stragglers.** RQ/H: adaptive rank còn lợi khi client chậm/non-IID; H communication–accuracy Pareto cải thiện. Gap/evidence: FLoRA/FlexLoRA giải aggregation/heterogeneous rank nhưng straggler/energy ít được chuẩn hoá. Gần nhất: FLoRA/FlexLoRA/FedPETuning; khác ở simulator/resource trace. Đóng góp: reproducible system benchmark. Thí nghiệm: 10–50 clients, Dirichlet partitions; accuracy/rounds/bytes/straggler time. A/B simulator, 2–5 GPU-days. Bác bỏ: fixed rank tốt hơn. Tái lập: deterministic client traces. 2028: privacy/energy-aware allocation.

**T23 — Client-Gradient Geometry Rank Allocation.** RQ/H: subspace overlap/cosine dự báo rank per client; H giảm bytes mà giữ global quality/fairness. Gap/evidence: FLoRA/FlexLoRA allocation còn heuristic; gradient geometry evidence từ GaLore/AdaRankGrad chưa đưa vào FL. Gần nhất: FLoRA/FlexLoRA; khác ở client uncertainty/overlap. Đóng góp: allocation rule + non-IID analysis. Thí nghiệm: 0.5–1B, 5–20 clients; overlap/bytes/accuracy/fairness. B/C, 5–15 GPU-days. Bác bỏ: no correlation. Tái lập: fixed partition/aggregation. 2028: secure aggregation compatible rule.

**T24 — Selective Backprop Stability.** RQ/H: dynamic masks có thay đổi OOD/generalization; H resampling mask giảm bias static. Gap/evidence: GreenTrainer/TraDy/GMT báo FLOP savings nhưng stability/seed/OOD chưa đồng nhất. Gần nhất: GreenTrainer/TraDy/GMT; khác ở multi-seed fair comparison. Đóng góp: mask stability/coverage metric. Thí nghiệm: RoBERTa/ViT-Tiny, GLUE/VTAB; FLOPs/quality/OOD. A/B, 2–5 GPU-days. Bác bỏ: masks gây degradation hoặc no savings after overhead. Tái lập: mask seeds/coverage logs. 2028: hardware-aware structured masks.

**T25 — Low-Rank Update Generalization Bound.** RQ/H: projected gradient/Fisher và rank cho bound dự đoán rank crossover trong linearized regime; H bound có tương quan thực nghiệm. Gap/evidence: LoRA expressivity/LoRA-One/COLT dynamics có theory nhưng gap finite-width, Adam, quantization. Gần nhất: LoRA expressivity, LoRA-One, Dayi–Chen; khác ở empirical predictor link. Đóng góp: theorem + calibrated test, hoặc negative theorem boundary. Thí nghiệm: quadratic/NTK toy + small Transformer; test rank/quality correlation. A/B, compute thấp, rủi ro lý thuyết cao. Bác bỏ: bound vacuous/no correlation. Tái lập: proof, synthetic seeds, code. 2028: non-convex/quantized extension.

## VI. Sàng lọc 10 đề tài

Thang 1–5 chỉ là đánh giá ra quyết định trong khảo sát (5 tốt hơn), không phải xác suất được nhận hội nghị. Điểm có **±** thể hiện bất định do corpus chưa exhaustive.

| Đề tài | Novelty evidence | Significance | Feasibility | Compute accessibility | Evaluation clarity | Reproducibility | Publication potential | Tổng/35 | Nhận xét |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1. QLoRA stability audit | 3±1 | 4 | 5 | 4 | 5 | 5 | 3±1 | 29 | Có thể xuất replication/benchmark nếu protocol chặt. |
| 2. Quantization error × gradient geometry | 4±1 | 5 | 4 | 4 | 4 | 4 | 4±1 | 29 | Gap đo lường có bằng chứng; method mới chỉ là bước hai. |
| 3. Early telemetry HPO | 4±1 | 4 | 4 | 4 | 4 | 4 | 4±1 | 28 | Cần kiểm tra rank inversion cross-task. |
| 4. Joint data–optimizer | 4±1 | 5 | 3 | 3 | 4 | 3 | 4±1 | 26 | Selection overhead và leakage là rủi ro lớn. |
| 5. Cross-model rank transfer | 4±1 | 4 | 4 | 4 | 4 | 4 | 4±1 | 28 | Dễ bắt đầu từ replication, có thể thất bại hợp lệ. |
| 6. Gradient-SNR rank scheduler | 3±1 | 4 | 3 | 3 | 4 | 3 | 4±1 | 24 | Nhiều phương pháp adaptive rank gần đây. |
| 7. Curvature-aware module selection | 3±1 | 4 | 3 | 3 | 4 | 3 | 3±1 | 23 | Chi phí Hessian/Fisher và instability. |
| 8. Data–rank co-selection | 4±1 | 4 | 3 | 3 | 3 | 3 | 4±1 | 24 | Giao điểm ít hơn nhưng design factorial nặng. |
| 9. Continual LoRA budget | 3±1 | 5 | 4 | 4 | 4 | 4 | 4±1 | 28 | Ý nghĩa stability–plasticity; benchmark CL phải chọn cẩn thận. |
| 10. Federated rank under stragglers | 3±1 | 4 | 3 | 3 | 4 | 3 | 3±1 | 23 | Cần simulator và kiểm soát non-IID; lĩnh vực đang đông. |

### Quyết định

- **Top 3:** #2, #3, #4.
- **Replication-first:** #1 hoặc #5.
- **Method-first sau pilot:** #2 (predictor/rank-bit policy) hoặc #4 (joint selector–optimizer).
- **Không khuyến nghị làm đầu tiên:** optimizer hoàn toàn mới (#13, #25) nếu chưa có mentor lý thuyết và baseline mạnh; adaptive federated LoRA chung chung (#23) vì mật độ paper cao.

## VII. Phân tích chuyên sâu 3 đề tài ứng viên

### 7.1 Candidate 1 — Quantization Error × Gradient Geometry as Predictors of QLoRA Stability

**Research question.** Với một layer (l), các đại lượng (e_l=\|W_l-Q_l\|_F/\|W_l\|_F), gradient SNR, cosine alignment giữa gradient quantized/full-precision, diagonal Fisher và effective rank có dự đoán được (a) variance giữa seeds, (b) rank tối thiểu để đạt quality, (c) degradation OOD không?

**Bằng chứng gần nhất.** QLoRA chứng minh 4-bit NF4 có thể fine-tune ở scale lớn; LoftQ chỉ ra quantization discrepancy làm (Q+AB^T) lệch khỏi (W) và dùng alternating quantization–SVD để khắc phục; QA-LoRA và QuAILoRA điều chỉnh group/initialization. Một nghiên cứu repeatability QLoRA trên IEEE Access ([DOI 10.1109/ACCESS.2024.3470850](https://ieeexplore.ieee.org/document/10700744/)) được dùng như evidence cần replication, nhưng vì full text bị giới hạn trong phiên này tôi không suy diễn chi tiết ngoài metadata/claim đã thấy. Các công trình trên chứng minh cơ chế *có thể quan trọng*, nhưng không thiết lập predictor ổn định xuyên model/task.

**Hypotheses.** H1: (e_l) đơn lẻ dự đoán kém hơn vector kết hợp ((e_l,\mathrm{SNR}_l,\mathrm{alignment}_l,\mathrm{rankEnergy}_l)). H2: predictor được fit trên task A chuyển được trong cùng model family nhưng suy giảm khi đổi family/bitwidth. H3: dùng predictor để cấp rank/bit chỉ có lợi ở budget thấp; ở budget cao lợi ích biến mất.

**Novelty type.** Empirical/theoretical diagnostic; method novelty (rank-bit policy) chỉ được claim nếu predictor được preregister, đánh giá out-of-distribution và vượt baselines ở cùng budget. Không claim “first” trước khi rà soát các preprint 2025–2026.

**Falsification.** (i) tương quan out-of-sample gần 0 hoặc đổi dấu; (ii) residual-only tốt ngang/tốt hơn; (iii) policy không vượt uniform tại cùng trainable params/tokens; (iv) seed variance không khác đáng kể sau CI.

### 7.2 Candidate 2 — Budget-Calibrated Early Telemetry for PEFT HPO/Rank Selection

**Research question.** Sau 5%, 10% và 20% budget, loss slope, gradient norm/SNR, update-to-weight ratio, low-rank spectrum và curvature proxy có dự báo ranking cuối của cấu hình (rank, LR, target module, quantization) không? Dự báo có được calibrate khi đổi task/model không?

**Bằng chứng gần nhất.** Hyperband/BOHB đã chứng minh early stopping/multi-fidelity là nền tảng; NeurIPS 2024 về Bayesian fine-tuning chỉ ra loss và metric landscapes có thể lệch; các paper PEFT/scale study cho thấy hyperparameter và task/data dependence mạnh. Gap hợp lệ là *calibration và transfer của telemetry*, không phải phát minh lại Hyperband.

**Hypotheses.** H1: rank correlation của telemetry tăng theo budget nhưng có ngưỡng task-dependent. H2: surrogate có uncertainty (heteroscedastic) giảm regret so với mean-only ASHA/Hyperband ở cùng GPU-hour. H3: tín hiệu hình học chuyển tốt hơn loss đơn thuần khi metric là OOD/generalization.

**Falsification.** Rank inversion cao hơn baseline; calibration error lớn; policy tiết kiệm compute nhưng làm giảm quality hoặc chỉ thắng nhờ ít steps; không chuyển được sang model family mới.

### 7.3 Candidate 3 — Joint Data Selection and PEFT Optimization under Fixed Compute Budget

**Research question.** Khi dữ liệu hạn chế, liệu chọn subset dựa trên influence/gradient similarity *kết hợp* với optimizer/rank schedule thích nghi có tạo interaction benefit so với tối ưu độc lập không?

**Bằng chứng gần nhất.** LESS đã là optimizer-aware selection; Deita và TSDS cho thấy chất lượng/diversity/target alignment quan trọng; DELIFT mở rộng nhiều giai đoạn. Vì vậy “data selection mới” không còn là gap rõ. Phần chưa được chứng minh đầy đủ là full-factorial, equal-budget, cross-model study đo interaction và selector overhead.

**Hypotheses.** H1: interaction chỉ xuất hiện ở 1–20% data hoặc non-IID, không ở full data. H2: optimizer-aware utility có lợi khi optimizer là AdamW/Adafactor nhưng không nhất thiết khi LOMO/SGD. H3: rank allocation nên phụ thuộc diversity/difficulty của subset.

**Falsification.** Hiệu ứng cộng tính (không interaction); selector cost làm mất toàn bộ savings; subset tốt trên target nhưng giảm OOD/continual transfer; lợi ích chỉ do thay đổi token count.

## VIII. Literature matrix cho các công trình gần nhất

| Nhóm | Paper | Tín hiệu/thuật toán | Budget được kiểm soát? | Seeds/robustness | Code/benchmark | Gap còn lại |
|---|---|---|---|---|---|---|
| Rank | AdaLoRA | SVD triplet importance | Có rank budget; không luôn cùng wall-time | Báo cáo theo benchmark paper | Có implementation phổ biến | Cross-model/quantization/seed calibration |
| Rank | DyLoRA | Nested rank | So sánh rank/training time trong setting gốc | Hạn chế scale | ACL PDF/code ecosystem | Layer heterogeneity và optimizer interaction |
| Rank | SalientLoRA | Singular-value time-series salience | Rank pruning schedule | Stability được nhắc, cần independent replication | NeurIPS PDF | Tín hiệu có transfer không? |
| Rank | AlphaLoRA | Heavy-tailed self-regularization | Expert allocation | Ba model/10 benchmark theo abstract | ACL + GitHub | Không phải trực tiếp rank/quantization |
| Quant | QLoRA | NF4 + double quant + paged optimizer | Memory/quality ở scale lớn | Seed variance chưa là trọng tâm | Official code | Geometry/stability predictor |
| Quant | LoftQ | Alternating Q/SVD init | Target rank/bit | Cần cross-seed | ICLR PDF/code | Dynamics sau init và rank adaptation |
| Quant | QA-LoRA | Group-wise quant/adaptation | Quantized deployment | Chưa đầy đủ run variance | Official GitHub | Stability under optimizer/bitwidth |
| Quant | QuAILoRA | Quantization-aware init | Same memory claim | Workshop/preprint context | PMLR/code | Independent replication, OOD |
| Optimizer | LOMO | Fused update/low memory | Memory strong, quality tradeoff | Tuning sensitive | ACL/GitHub | Equal FLOPs and PEFT comparison |
| Optimizer | GaLore | Gradient low-rank projection | Optimizer state/memory | Large-scale claims | arXiv/code | Projection rank schedule, FT vs pretrain |
| Data | LESS | Adam-aware low-rank gradient similarity | 5% examples often | Transfer evidence in paper | PMLR/GitHub | Joint optimizer/rank, long-term utility |
| Data | Deita | Complexity/quality/diversity | 6K examples | Human+automatic eval | ICLR/code | OOD/contamination/optimizer interaction |
| Data | TSDS | OT alignment + KDE diversity | 1% selection examples | Task-specific | NeurIPS/GitHub | Selector overhead and model shift |
| Stability | Du & Nguyen | 7 instability measures | Not primarily compute budget | Seeds are central | ACL | PEFT+quantization replication |
| HPO | Hyperband/BOHB | Early stopping/multi-fidelity | Explicit resource | Depends on stochastic noise | Mature open source | PEFT telemetry/calibration |
| HPO | NeurIPS 2024 BO model fusion | Multi-objective BO | Outer-loop expensive | Fine-tuning sensitivity noted | Paper/code | Early signal transfer and uncertainty |
| Continual | I-LoRA/Bayesian PEFT | Replay/interpolation/Laplace | Memory/task sequence | Domain-specific | arXiv/code | Rank budget and OOD forgetting |

## IX. Research gap, novelty analysis và related-work comparison

### Gap 1: stability predictor (Candidate 1)

1. **Vấn đề:** quantization và low-rank restriction có thể làm thay đổi hướng/variance gradient, nhưng người dùng cần biết layer nào và khi nào.
2. **Hiện tại:** QLoRA giảm memory; LoftQ/QA-LoRA/QuAILoRA sửa discrepancy/init; AdaLoRA/SalientLoRA sửa rank.
3. **Gần nhất:** các paper trên; instability study.
4. **Hạn chế được tài liệu gợi ý:** discrepancy sau quantization; seed instability; rank heterogeneity.
5. **Chưa giải quyết đầy đủ (suy luận):** predictor out-of-sample kết hợp geometry và residual dưới fixed budget.
6. **Bằng chứng trực tiếp vs suy luận:** trực tiếp = paper mô tả cơ chế/variance; suy luận = predictor sẽ chuyển được.
7. **Ý nghĩa:** biến recipe tuning thành chẩn đoán có thể kiểm chứng, giúp quyết định rank/bit trước khi chạy đầy đủ.
8. **Hypothesis:** như §7.1.
9. **Thí nghiệm:** factorized bit×rank×model×seed, log layer telemetry.
10. **Nếu đúng:** empirical predictor + policy; theoretical novelty chỉ nếu có bound/criterion.
11. **Nếu sai:** negative result chỉ ra proxy không đủ và giúp tránh heuristic.

### Gap 2: early telemetry (Candidate 2)

Chuỗi lập luận tương tự: literature chứng minh early stopping/BO và sensitivity, nhưng chưa đủ bằng chứng cho *calibrated transfer* của gradient/curvature telemetry trong PEFT. Đóng góp có thể là benchmark/protocol và policy uncertainty-aware; nếu không có transfer, vẫn thu được bản đồ rank inversion và điều kiện không dùng early stopping.

### Gap 3: data–optimizer interaction (Candidate 3)

LESS đã optimizer-aware nên không được claim “chưa ai kết hợp”. Gap hẹp hơn: đánh giá interaction dưới *cùng token/step/compute*, gồm chi phí tạo selector, nhiều optimizer và rank schedule, cùng OOD/continual holdout. Nếu không có interaction, kết quả bác bỏ giả định joint optimization và có giá trị phương pháp luận.

## X. Research questions, hypotheses và methodology chung

### Nguyên tắc thiết kế

- **Đơn vị so sánh:** báo cáo riêng bốn trục: cùng compute budget, cùng wall-clock, cùng trainable parameters, cùng quality target.
- **Không dùng một metric:** luôn có quality, variance/CI, peak GPU memory, wall time, throughput, approximate FLOPs, tokens/steps và energy nếu đo được.
- **Seeds:** pilot ≥3; kết luận chính ≥5, lý tưởng 7–10 khi data nhỏ/claim stability. Dùng cùng seed set giữa baseline; báo cáo paired differences.
- **Statistics:** mean/median, SD, bootstrap 95% CI, effect size (paired Cohen’s d hoặc Cliff’s delta), corrected multiple comparisons khi nhiều config. Không chỉ báo best run.
- **Leakage/contamination:** tách train/dev/test; không dùng benchmark test để chọn selector/HPO; kiểm tra n-gram/semantic overlap với pretraining/instruction corpus khi có thể; thêm dynamic/new benchmark hoặc held-out domain.
- **Reproducibility:** pin commit model/tokenizer/library/CUDA; lưu config, seed, data hash, prompt/chat template, sequence length, gradient accumulation, checkpoint; script một lệnh; ghi hardware/driver.

### Model/data gợi ý theo resource

- **Kịch bản A (máy cá nhân 8–16 GB VRAM):** RoBERTa-base/large, DeBERTa-v3-base, Pythia-410M/1B, Qwen2.5-0.5B, Phi-3.5-mini nếu quantized; GLUE/SuperGLUE subset, AG News, small instruction corpus.
- **Kịch bản B (cloud giới hạn):** Qwen2.5-1.5B/3B, SmolLM2-1.7B, Llama-3.2-1B/3B; MMLU-mini/GSM8K subset, Alpaca/OpenHermes filtered, domain holdout; 1–4 GPUs 24–48 GB.
- **Kịch bản C (lab):** 7–8B models, broader instruction/continual/federated simulations; đo FLOPs/energy thật.

## XI. Protocol thực nghiệm cho 3 đề tài cuối

### Protocol P1 — Quantization × geometry

1. **RQ/H:** P1–P3 ở §7.1; preregister predictor, features và split trước khi xem test.
2. **Baselines:** full-precision LoRA, QLoRA-NF4, INT4/INT8 LoRA, LoftQ, QA-LoRA; static ranks r=2/4/8/16; optional AdaLoRA/SalientLoRA khi compatible.
3. **Models/data:** Qwen2.5-0.5B/1.5B và Pythia-1B (same tokenizer family comparison); GLUE/SuperGLUE subset + instruction/math held-out. Chạy ít nhất 3 task type (classification, generation, reasoning).
4. **Seeds:** 7 seeds cho pilot model; 5 seeds cho mỗi cross-model condition.
5. **Ablation:** residual-only; gradient-only; Fisher-only; pairwise/full features; no LoftQ; no adaptive rank; bitwidth; target-module subset; sequence length.
6. **Metrics:** task score, OOD score, per-layer residual/Fisher/SNR/alignment, effective rank, seed SD/CI, loss spikes, peak memory, wall time, throughput, approximate FLOPs; energy via nvidia-smi power logging nếu khả thi.
7. **Fairness:** cùng examples/tokens, optimizer steps, max sequence length, trainable parameter budget; báo cáo riêng same-memory và same-time.
8. **Kết luận “tốt hơn”:** predictor phải cải thiện rank/bit policy ở ít nhất 2 model families và 3 tasks, CI không chồng quá mức theo paired test, không làm tăng selector overhead vượt savings.

### Protocol P2 — early telemetry HPO

1. **Search space cố định:** rank {2,4,8,16}, LR log-uniform, target modules {qv,qkvo,all-linear}, bits {16,8,4}, scheduler {constant, cosine}; không mở rộng sau khi xem test.
2. **Methods:** random search, Hyperband/ASHA, BOHB hoặc Bayesian baseline, telemetry-only policy; cùng tổng GPU-hour và maximum completed tokens.
3. **Telemetry:** loss slope/curvature finite difference, gradient norm/SNR, update-to-weight, low-rank spectrum, validation mini-batch score; log tại 5/10/20/40% budget.
4. **Seeds/tasks:** 5 seeds × 4 tasks; hold out one task/model family for transfer. Use rank-correlation and calibration, not only final best score.
5. **Metrics:** simple regret, anytime regret, Spearman/Kendall ranking, expected calibration error for “promote/stop”, quality–time–memory Pareto area.
6. **Ablation:** loss-only, telemetry-only, telemetry without uncertainty, no curvature, no cross-task transfer; early budget levels.
7. **Falsification:** if telemetry cannot beat ASHA at equal budget or has unstable transfer, report decision boundary and stop method claim.

### Protocol P3 — joint data–optimizer

1. **Data:** one candidate corpus with explicit train/dev/test and deduplication; target capability represented by few-shot *selection* examples not used for final test. Selection fractions 1%, 5%, 10%, 20%, 100%.
2. **Selectors:** random, length, Deita-like quality/diversity, LESS, TSDS where code available; log preprocessing/gradient datastore cost.
3. **Optimizers/adapters:** AdamW-LoRA, AdaLoRA, Adafactor-LoRA or LOMO-compatible full/PEFT; optional gradient-SNR schedule from pilot. Full factorial only on reduced grid, then confirm top interactions.
4. **Controls:** same token budget (if subset smaller, repeat/pack or cap steps explicitly), same optimizer updates, same trainable params, same wall-time comparisons separately; no “fewer tokens = cheaper and better” ambiguity.
5. **Seeds/statistics:** 5 seeds per cell for main grid; 7 for final comparison; mixed-effects model with task/model as random effects if enough cells.
6. **Metrics:** in-domain, OOD, capability-specific, forgetting/transfer to a second task, score variance, selector cost, tokens/FLOPs/time/memory; Pareto frontier.
7. **Falsification:** no interaction, or interaction disappears after equalizing tokens/compute, or gains do not transfer to second model. These are publishable empirical conclusions if protocol and code are strong.

## XII. Rủi ro, hạn chế và tiêu chí bác bỏ

| Rủi ro | Dấu hiệu | Phương án dự phòng |
|---|---|---|
| Literature gap bị lấp bởi preprint mới | Tìm thấy LAARA/QR-Adaptor tương tự | Đổi thành replication/cross-model evaluation hoặc theorem/diagnostic; ghi ngày snapshot. |
| Kết quả chỉ thắng nhờ tuning | Baseline dùng default, method được search rộng | Dùng cùng search budget, report all configs và preregister. |
| Selector overhead lớn | Precompute gradients vượt training savings | Đo amortized cost; thu hẹp corpus hoặc dùng proxy rẻ. |
| Seed variance lớn | CI chồng, rank inversion | Chuyển claim sang stability/uncertainty; tăng seeds thay vì cherry-pick. |
| Benchmark contamination | Điểm cao bất thường trên static benchmark | Dedup, held-out/new/dynamic benchmark; không dùng test để tune. |
| Không đủ GPU | OOM/elapsed quá lớn | SLM/encoder, sequence ngắn, gradient accumulation, subset benchmark; giữ nguyên câu hỏi cơ chế. |
| Predictor không transfer | Cross-model correlation thấp | Đóng góp negative result + điều kiện thất bại; không ép method. |

## XIII. Kiểm tra khả năng hữu dụng và research gap đến 31/12/2028

Mốc 2028 ở đây là **mốc đánh giá lại**, không phải lịch trình bắt buộc và cũng không phải dự báo rằng một hướng chắc chắn còn mới. Một đề tài chỉ được giữ nếu vượt các cổng kiểm tra sau:

| Cổng kiểm tra | Câu hỏi phải trả lời | Bằng chứng tối thiểu | Nếu không đạt |
|---|---|---|---|
| 1. Relevance | Bài toán còn gắn với bottleneck thật (GPU memory, datastore, token/compute) không? | Đo wall-clock, peak memory, token/FLOP và chi phí tiền xử lý trên ít nhất hai quy mô model. | Thu hẹp sang một regime tài nguyên cụ thể hoặc bỏ hướng. |
| 2. Gap không bị lấp | Có paper mới nào đã giải đúng câu hỏi, cùng budget và cùng protocol không? | Forward/backward citation search cập nhật ngay trước khi nộp; phân biệt published/preprint. | Chuyển contribution sang replication, stress test hoặc phân tích điều kiện thất bại. |
| 3. Khả năng kiểm chứng | Giả thuyết còn đo được trên model/dataset mở và có baseline mạnh không? | Có code, checkpoint, split và metric định trước; không phụ thuộc API kín. | Chuyển sang benchmark/protocol hoặc chọn model nhỏ hơn. |
| 4. Ý nghĩa ngoài một cấu hình | Hiệu ứng có giữ được qua seed, model, bitwidth hoặc data regime không? | Ít nhất 3 seeds, một cross-model hoặc cross-regime hold-out, CI và phân tích sensitivity. | Chỉ trình bày như kết quả giới hạn/negative result, không claim phương pháp tổng quát. |
| 5. Khả năng xuất bản | Kết quả có đủ một câu hỏi khoa học, ablation và artifact để viết bài không? | Manuscript 6–8 trang cho hội thảo hoặc 8–15 trang cho bài báo, mọi claim truy được tới log. | Gửi technical report/replication trước, chưa tuyên bố novelty. |

### Quy trình theo giai đoạn (không gắn cứng vào một năm)

1. **Nền tảng và literature matrix:** đọc, kiểm tra trạng thái venue, dựng citation graph và khóa định nghĩa gap.
2. **Replication:** chạy baseline quan trọng với seed và budget được kiểm soát.
3. **Hypothesis test:** chỉ thêm method nếu replication cho thấy một cơ chế hoặc failure boundary ổn định.
4. **Method/analysis:** xây dựng thuật toán tối thiểu, ablation và falsification; negative result vẫn là đầu ra hợp lệ.
5. **Manuscript và cập nhật gap:** trước khi nộp, lặp lại cổng 2 và cập nhật literature đến ngày nộp.

## XIV. Nhật ký truy vấn, nguồn và giới hạn

### Truy vấn đã dùng (25/10/2026)

| Cụm nguồn | Truy vấn đại diện | Mục đích |
|---|---|---|
| arXiv/OpenReview/ACL/PMLR | `LoRA adaptive rank allocation gradient geometry 2024 2025`; `AdaLoRA DyLoRA SalientLoRA AlphaLoRA RankAdaptor` | Rank/PEFT và nearest methods |
| NeurIPS/ICLR/ICML | `QLoRA QA-LoRA LoftQ LQ-LoRA quantization aware fine tuning`; `GaLore LOMO low memory optimizer` | Quantization/optimizer |
| PMLR/ACL/GitHub | `LESS influential data targeted instruction tuning`; `TSDS data selection task-specific finetuning`; `Deita data efficient instruction tuning` | Data selection và code |
| JMLR/PMLR/NeurIPS | `Hyperband BOHB multi-fidelity early stopping`; `Bayesian optimization language model fine tuning` | HPO/early telemetry |
| ACL/IEEE/arXiv | `fine-tuning instability random seeds PEFT QLoRA repeatability`; `catastrophic forgetting LoRA continual` | Stability/continual |
| NeurIPS/IEEE/ACL | `federated LoRA heterogeneous rank communication`; `gradient sparsification structured updates` | Resource/federated |

### Nguồn không thể xác minh đầy đủ

- Google Scholar, Semantic Scholar, OpenAlex và Crossref không được crawl exhaustively trong môi trường này; dùng web index và trang chính thức thay thế.
- Một số IEEE Xplore/ACM/Springer full text bị giới hạn quyền; chỉ dùng metadata/abstract/DOI khi không tải được toàn văn.
- Preprint 2025–2026 có thể thay đổi trạng thái; phải kiểm tra lại venue/version trước khi nộp bài.
- Không dùng blog/forum làm bằng chứng chính cho claim khoa học; nếu có nhắc chỉ coi là gợi ý tìm kiếm.

## Trả lời trực tiếp 8 câu hỏi cuối

1. **Nhánh đông và nguy cơ khó tạo novelty:** generic PEFT/LoRA variants, quantization recipes, data quality selection, federated LoRA và low-memory optimizers.
2. **Gap đáng khảo sát:** stability predictor dựa trên quantization residual + gradient geometry; calibrated early telemetry cho PEFT HPO; joint data–optimizer–rank dưới fixed budget; cross-model transfer của rank maps.
3. **Ba đề tài cân bằng:** Candidate 1–3 ở §VII.
4. **Công trình gần nhất và phải khác:** Candidate 1 — QLoRA/LoftQ/QA-LoRA/QuAILoRA; khác ở predictor và cross-seed/cross-model protocol. Candidate 2 — Hyperband/BOHB/NeurIPS BO; khác ở telemetry geometry + calibration/transfer. Candidate 3 — LESS/Deita/TSDS/DELIFT; khác ở factorial interaction và equal-budget accounting.
5. **Phù hợp replication:** Seed-Stable QLoRA Audit hoặc Cross-Model Rank Allocation Transfer.
6. **Có thể tiến tới thuật toán mới:** telemetry-calibrated rank–bit policy hoặc optimizer-aware data–rank co-selection, nhưng chỉ sau pilot.
7. **Thu hẹp cho sinh viên:** một model 0.5–1.5B, 2–3 task, 4–8 rank/config, 5 seeds, 1 quantizer; ưu tiên measurement/replication trước method. Giữ fixed token/step và log memory/time.
8. **Cần xác minh trước khi tuyên bố contribution:** citation search hai chiều đến ngày nộp; trạng thái peer review; code/version; seed count; selector/HPO cost; contamination; equal-budget fairness; cross-model/task replication; confidence intervals; ablation loại bỏ từng thành phần.

## Tài liệu tham khảo và đường dẫn kiểm chứng (chọn lọc)

1. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685, 2021/ICLR 2022. https://arxiv.org/abs/2106.09685
2. Valipour et al., *DyLoRA*, EACL 2023. https://aclanthology.org/2023.eacl-main.239.pdf
3. Zhang et al., *AdaLoRA*, ICLR 2023. https://openreview.net/pdf?id=lq62uWRJjiY
4. Dettmers et al., *QLoRA*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html
5. Lv et al., *Full Parameter Fine-tuning … (LOMO)*, ACL 2024. https://aclanthology.org/2024.acl-long.445.pdf
6. Zhao et al., *GaLore*, ICML 2024 (PMLR), arXiv:2403.03507. https://proceedings.mlr.press/v235/zhao24s.html
7. Liu et al., *DoRA*, ICML 2024. https://proceedings.mlr.press/v235/liu24bn.html
8. Xu et al., *QA-LoRA*, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/e6c2e85db1f1039177c4495ccd399ac4-Abstract-Conference.html
9. Li et al., *LoftQ*, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/39ec972afab01e0d8ddc6834a9d12ac1-Paper-Conference.pdf
10. Kim et al., *LQ-LoRA*, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/d97a16cb83d74195b76e0bf1e85bf072-Abstract-Conference.html
11. Hayou et al., *LoRA+*, ICML 2024. https://openreview.net/pdf?id=NEv8YqBROO
12. Xia et al., *LESS*, ICML 2024. https://proceedings.mlr.press/v235/xia24c.html
13. Liu et al., *Deita*, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/6091f2bb355e960600f62566ac0e2862-Abstract-Conference.html
14. Kang et al., *Get More for Less*, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/b36dc39b319ba6ba2a0fd7601951efb4-Paper-Conference.pdf
15. Liu et al., *TSDS*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/13848b5893119ff772b69812c95914fa-Abstract-Conference.html
16. Qing et al., *AlphaLoRA*, EMNLP 2024. https://aclanthology.org/2024.emnlp-main.1141/
17. *SalientLoRA*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/ed9f00cb7dd5fbdc2175d55e2fdf1b05-Paper-Conference.pdf
18. Du & Nguyen, *Measuring the Instability of Fine-Tuning*, ACL 2023. https://aclanthology.org/2023.acl-long.342/
19. Andriushchenko et al., *A Modern Look at the Relationship between Sharpness and Generalization*, ICML 2023. https://proceedings.mlr.press/v202/andriushchenko23a.html
20. Zhang et al., *When Scaling Meets LLM Finetuning*, ICLR 2024. https://deepmind.google/research/publications/49667/
21. Li et al., *Hyperband*, JMLR 2018. https://www.jmlr.org/beta/papers/v18/16-558.html
22. Falkner et al., *BOHB*, ICML 2018. https://proceedings.mlr.press/v80/falkner18a.html
23. Jang et al., *Model Fusion through Bayesian Optimization in Language Model Fine-Tuning*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/34d3cf97696022b179171e5abda42c0b-Abstract-Conference.html
24. Ren et al., *Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning*, arXiv:2402.18865. https://arxiv.org/abs/2402.18865
25. Chen & Garner, *Bayesian PEFT for Overcoming Catastrophic Forgetting*, arXiv:2402.12220. https://arxiv.org/abs/2402.12220
26. Koo et al., *Towards Robust and Efficient Federated Low-Rank Adaptation*, ACL 2025. https://aclanthology.org/2025.acl-long.19.pdf
27. EleutherAI, *lm-evaluation-harness*. https://github.com/EleutherAI/lm-evaluation-harness
28. Li et al., *HetLoRA: Federated Fine-Tuning of Foundation Models with Heterogeneous Devices*, EMNLP 2024. https://aclanthology.org/2024.emnlp-main.717/
29. Koo et al., *Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients* (LoRA-A2), ACL 2025. https://aclanthology.org/2025.acl-long.19.pdf
30. He et al., *GoLore / Subspace Optimization for Large Language Models with Convergence Guarantees*, ICML 2025. https://proceedings.mlr.press/v267/he25i.html
31. Refael et al., *AdaRankGrad*, ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/fe99ca7c293fe083c8fe7e85dce2f814-Abstract-Conference.html
32. Zhu et al., *APOLLO*, MLSys 2025. https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf
33. Ansell et al., *SpIEL*, arXiv:2401.16405. https://arxiv.org/abs/2401.16405
34. Bischl et al., *Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges*, WIREs 2023. https://doi.org/10.1002/widm.1484
35. Adriaensen et al., *Efficient Bayesian Learning Curve Extrapolation using LC-PFN*, NeurIPS 2023. https://papers.neurips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract-Conference.html
36. Egelé et al., *Is One Epoch All You Need for Multi-Fidelity HPO?*, ESANN 2023. https://doi.org/10.14428/esann/2023.ES2023-84
37. Axiotis et al., *Data Efficient Learning via Clustering-Based Sensitivity Sampling*, ICML 2024. https://proceedings.mlr.press/v235/axiotis24a.html
38. Borsos et al., *Data Summarization via Bilevel Optimization*, JMLR 2024. https://www.jmlr.org/papers/v25/21-1132.html
39. Deng et al., *Influential Language Data Selection via Gradient Trajectory Pursuit*, ICLR 2025. https://openreview.net/forum?id=che9LCwPQM
40. Zhao et al., *Long Is More for Alignment*, ICML 2024. https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhao24b/zhao24b.pdf
41. Wang et al., *A Survey on Data Selection for LLM Instruction Tuning*, arXiv:2402.05123. https://arxiv.org/abs/2402.05123
42. Coleman et al., *Selection via Proxy*, ICLR 2020. https://openreview.net/forum?id=HJg2b0VYDr
43. Mindermann et al., *RHO-LOSS*, ICML 2022. https://proceedings.mlr.press/v162/mindermann22a.html
44. Liang & Li, *InfLoRA*, CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR2024_paper.html
45. Wang et al., *FLoRA*, NeurIPS 2024. https://papers.neurips.cc/paper_files/paper/2024/hash/28312c9491d60ed0c77f7fff4ad86dd1-Abstract-Conference.html
46. Zhang et al., *Holdout-Loss-Based Data Selection via In-Context Learning*, ICLR 2026. https://www.microsoft.com/en-us/research/publication/holdout-loss-based-data-selection-for-llm-finetuning-via-in-context-learning/

**Trạng thái báo cáo:** hoàn thành một vòng khảo sát có giới hạn và thiết kế protocol; chưa phải systematic review đăng ký theo PRISMA và chưa thể thay thế bước xác minh novelty cuối cùng trước khi viết bài.

## Addendum 26/10/2026 — Quyết định sau khi research thêm

Sau khi kiểm tra thêm QERA, SRR, L4Q, QA-BLoRA, QeRL, AQLoRA, các kết quả phủ định về gradient-based rank allocation và các công trình data-selection mới, hướng được chọn để đi sâu là:

**Quantization–Gradient Alignment as a Predictor of QLoRA Run Stability**  
(*Dùng độ lệch lượng tử–gradient để dự đoán độ ổn định của QLoRA*).

Phạm vi được khóa ở **training-time layer-risk/stability predictor cho QLoRA**; không claim “first” cho gradient alignment, joint bit–rank post-training allocation, token dropping hoặc calibration ngoài LLM PEFT.

Các công trình post-training như GuidedQuant và FOEM, cùng CLoQ ở khởi tạo LoRA có calibration, khiến claim “gradient–quantization alignment” hoặc “calibration-aware initialization” đơn thuần không còn an toàn. Vì vậy phần còn được kiểm chứng trong đề tài là trajectory/multi-seed stability và transfer của QLoRA dưới cùng ngân sách, không phải một PTQ objective mới.

Lý do không chọn data selection × optimizer làm bài toán chính: GIST, QLESS, OPUS, BLISS, DualSFT và các preprint gần đây đã làm giao điểm đó khá đông; một selector mới dễ trở thành thay đổi heuristic. Data selection chỉ nên là extension sau khi core stability predictor có tín hiệu.

Bản đề xuất tập trung, gồm gap chain, RQ/H, protocol, baseline, tiêu chí bác bỏ và kế hoạch nộp trong nước, nằm ở [focused_topic_qga_lora_vi.md](focused_topic_qga_lora_vi.md).

### Khóa benchmark và định vị nghiên cứu

Để thống nhất với bản focused topic và roadmap, bài toán thực nghiệm chính được khóa ở **hỏi–đáp tiếng Việt trên UIT-ViQuAD2.0**. Model chính được cập nhật thành `sail/Sailor2-1B-Chat`, một causal LM đã thích nghi tiếng Việt/Đông Nam Á; `Qwen/Qwen2.5-0.5B-Instruct` giữ vai trò control cùng protocol. Hai model dùng để phát triển predictor trên train/dev; sau khi freeze predictor, threshold, rank policy, QA template và config mới mở test held-out, rồi kiểm chứng thêm trên model scale/family khác nếu còn ngân sách. UIT-ViQuAD2.0 ở đây là benchmark tạo trajectory và metric cho nghiên cứu thuật toán, **không phải sản phẩm chatbot**. Nếu split test không truy cập được, phải tạo held-out split có seed/hash cố định trước khi xem kết quả.
