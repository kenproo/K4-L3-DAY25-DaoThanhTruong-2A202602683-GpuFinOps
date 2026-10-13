# BÁO CÁO KỸ THUẬT PHÂN TÍCH TỐI ƯU HÓA CHI PHÍ GPU (FINOPS WRITE-UP)
**Dự án:** NimbusAI — GPU Cost Optimization Workshop  
**Học viên:** Đào Thanh Trường  
**Mã sinh viên:** 2A202602683  
**Chương trình:** VinUni AICB · Phase 2 · Track 2 (Infrastructure) · Day 25  
**Ngày báo cáo:** 13/10/2026  

---

## 1. Tổng quan: Baseline vs. Optimized Spend

Qua quá trình kiểm toán toàn diện hạ tầng tính toán tại NimbusAI, kết quả tối ưu hóa đa tầng (Multi-tier FinOps) đạt được như sau:

| Chỉ số kinh tế | Trước tối ưu (Baseline) | Sau tối ưu (Optimized) | Chênh lệch tiết kiệm | % Tiết kiệm |
|---|:---:|:---:|:---:|:---:|
| **Tổng chi phí GPU / tháng** | **$27,133** | **$14,626** | **$12,507** | **46.1%** |
| **Đơn giá Inference (`$/1M-token`)** | **$6.488** | **$1.126** | **$5.362** | **82.6%** |
| **Chi phí Inference / ngày** | $48.87 | $8.48 | $40.39 | 82.6% |
| **Lãng phí GPU Idle / tháng** | $600 | $0 | $600 | 100.0% |
| **Năng lượng trung bình / truy vấn** | 1.84 Wh | 0.24 Wh | 1.60 Wh | 86.9% |
| **Phát thải Carbon / truy vấn** | 0.700 gCO2e | 0.091 gCO2e | 0.609 gCO2e | 87.0% |

> **Nhận định cốt lõi:** Bằng cách chuyển đổi hệ quy chiếu từ việc trả tiền theo `$/GPU-giờ` sang chuẩn hóa giá trị đầu ra theo `$/1M-token`, NimbusAI đã cắt giảm được gần một nửa tổng hóa đơn GPU hàng tháng, đồng thời hạ giá thành phục vụ mỗi triệu token xuống hơn **5.7 lần** ($6.488 \rightarrow $1.126).

---

## 2. Phân tích chi tiết từng đòn bẩy tiết kiệm (Savings Levers)

Biểu đồ thác nước (`outputs/savings.png`) ghi nhận 4 đòn bẩy chính đóng góp vào tổng khoản tiết kiệm **$12,507/tháng**:

| Đòn bẩy tối ưu | Khoản tiết kiệm / tháng | Tỷ trọng đóng góp | Cơ chế tác động |
|---|:---:|:---:|---|
| **1. Purchasing Strategy (Spot / Reserved)** | **$10,040** | **80.3%** | Chuyển workload training sang Spot; cam kết Reserved cho inference |
| **2. Inference Levers (Cascade / Cache / Batch)** | **$1,212** | **9.7%** | Phối hợp tầng mô hình, tái sử dụng prefix và gộp batch |
| **3. Right-sizing Util-lies** | **$655** | **5.2%** | Hạ cấp GPU dư thừa compute sang GPU cân bằng băng thông |
| **4. Kill Idle GPUs** | **$600** | **4.8%** | Tắt tự động các instance có độ sử dụng < 10% |

### So sánh đòn bẩy Mua sắm (Purchasing) vs. đòn bẩy Suy luận (Inference):
* **Quy mô số tiền tuyệt đối:** Đòn bẩy mua sắm (Purchasing) chiếm tới **80.3%** giá trị tiền tiết kiệm được ($10,040/tháng). Nguyên nhân là do các job huấn luyện mô hình (Training & Fine-tuning) tiêu thụ cụm GPU H100 và A100 liên tục với công suất lớn. Việc chuyển các job ngắt quãng được (`interruptible=1`) sang Spot instance với giá rẻ hơn 40–60% kết hợp thuật toán tính điểm hòa vốn Reserved ($\ge 55\%$ duty cycle) đã giải phóng một khoản ngân sách khổng lồ.
* **Tỷ lệ phần trăm hiệu quả:** Đòn bẩy Inference lại có tỷ lệ cắt giảm cao nhất (**82.6%**). Nhờ đòn bẩy tích lũy chiết khấu đa tầng (*Discount Stack*):
  $$\text{Effective Cost} = \text{Batch Discount (50%)} \times \text{Cached Input (10%)} = 0.05 \text{ (giảm đến 95%)}$$
  khiến chi phí phục vụ token giảm ngoạn mục từ $6.488 xuống còn $1.126 trên mỗi 1 triệu token.

---

## 3. Bản chất của "GPU-Util Lie" và Tác động Tài chính

### 3.1. Các GPU bị phát hiện "nói dối"
Trong Mission 1, thuật toán kiểm toán đã bóc trần 2 GPU rơi vào nhóm *GPU-Util Lie* ($\text{Util} \ge 90\%$ nhưng $\text{MFU} < 30\%$):
1. **`gpu-h100-4`**: Báo cáo `nvidia-smi` đạt **98.2% Util**, nhưng Model FLOPs Utilization (**MFU**) chỉ đạt **19.4%** (0.194), Model Bandwidth Utilization (**MBU**) đạt **20.7%**.
2. **`gpu-a10g-1`**: Báo cáo **96.9% Util**, nhưng MFU chỉ đạt **26.8%**, MBU đạt **30.2%**.

### 3.2. Cơ chế kỹ thuật đằng sau hiện tượng
* Chỉ số `gpu_util_pct` trong công cụ giám sát như `nvidia-smi` thực chất chỉ đo tỷ lệ phần trăm thời gian mà **ít nhất một kernel đang chiếm giữ GPU clock**. Nó hoàn toàn không đo lường khối lượng phép tính ma trận (Tensor Core FLOPs) thực sự được thực thi.
* Đối với mô hình ngôn ngữ lớn (LLM), quá trình suy luận chia làm hai pha rõ rệt theo **Roofline Model**:
  - **Prefill (Prompt Processing):** Mang tính toán ma trận lớn ($GEMM$), Arithmetic Intensity cao (~455 FLOP/byte) $\rightarrow$ *Compute-bound*. MFU thường cao.
  - **Decode (Token Generation):** Sinh từng token tuần tự ($GEMV$), mỗi bước phải nạp toàn bộ trọng số mô hình từ bộ nhớ HBM vào SRAM cho 1 token duy nhất. Arithmetic Intensity rất thấp (~1–2 FLOP/byte so với điểm uốn Ridge Point 295 FLOP/byte của H100) $\rightarrow$ *Memory-bandwidth bound*.
* Tại `gpu-h100-4`, GPU liên tục bị **Memory Stall** (nghẽn nạp dữ liệu từ HBM) hoặc chờ truyền dữ liệu liên GPU. Xung nhịp đồng hồ vẫn chạy (báo 98% busy), nhưng các nhân Tensor Core hầu như không thực hiện tính toán.

### 3.3. Tác hại tài chính
* NimbusAI đang chi trả **$2.50/giờ** để thuê một cỗ máy H100 có năng lực đỉnh 990 TFLOPs FP16, nhưng thực tế chỉ thu lại được ~192 TFLOPs năng lực thực tế.
* Đây là sự lãng phí vô hình: Trả 100% tiền nhưng chỉ nhận 20% giá trị tính toán. Nếu chỉ nhìn vào `nvidia-smi`, các kỹ sư sẽ lầm tưởng hạ tầng đã quá tải và xin cấp thêm ngân sách mua thêm H100, trong khi thực tế phần cứng đang bị tắc nghẽn kiến trúc phần mềm hoặc chọn sai GPU.

---

## 4. Các phần mở rộng chuyên sâu (Your Turn Extensions)

Bài làm đã triển khai thành công **3 phần mở rộng** (vượt yêu cầu $\ge 2$ của Rubric):

### 4.1. Extension 2: Right-sizing theo MBU & Đơn giá VRAM (`$/GB-VRAM`)
* **Vị trí triển khai:** `missions/m1_efficiency_audit.py`
* **Phương pháp:**
  - Tính đơn giá VRAM cho từng dòng GPU trong catalog:
    $$\text{Cost per GB-VRAM} = \frac{\text{On-demand Price (\$/hr)}}{\text{HBM Capacity (GB)}}$$
    Kết quả: **MI300X** đạt đơn giá rẻ nhất ($0.0102/GB-hr, 192GB, 5.3 TB/s), tiếp theo là **A100** ($0.0224/GB-hr, 80GB, 2.0 TB/s), trong khi **H100** là $0.0312/GB-hr và **A10G** lên tới $0.0417/GB-hr.
  - Với các workload inference bị nghẽn băng thông nhưng không tận dụng hết compute (MBU < 0.35), thay vì trả tiền cho H100, ta chuyển sang GPU có băng thông đỉnh đáp ứng vừa vặn băng thông thực tế đạt được (*Achieved BW*).
* **Kết quả đo lường cụ thể:**
  - `gpu-h100-4` (Achieved BW = 0.69 TB/s) $\rightarrow$ chuyển sang **A100** (Peak BW 2.0 TB/s, 80GB), tiết kiệm **$511/tháng** (giảm 28.4%).
  - `gpu-h100-5` (Achieved BW = 0.91 TB/s) $\rightarrow$ chuyển sang **A100**, tiết kiệm **$511/tháng** (giảm 28.4%).
  - `gpu-a100-0` & `gpu-a100-1` (Achieved BW ~0.5 TB/s) $\rightarrow$ chuyển sang **A10G**, tiết kiệm **$569/tháng** mỗi GPU (giảm 44.1%).
  - `gpu-a10g-0` & `gpu-a10g-1` (Achieved BW < 0.2 TB/s) $\rightarrow$ chuyển sang **L4** ($0.80/hr), tiết kiệm **$144/tháng** mỗi GPU (giảm 20.0%).
* **Tổng tiềm năng tiết kiệm:** **$2,448/tháng** nếu áp dụng triệt để chính sách MBU Right-sizing.

### 4.2. Extension 3: Kinh tế học của Prompt Caching (`cache_is_worth_it`)
* **Vị trí triển khai:** `finops/pricing.py` và `missions/m2_inference_levers.py` (kèm test tại `tests/test_extensions.py`)
* **Mô hình toán học:**
  Việc ghi vào cache phát sinh phụ phí lưu trữ ban đầu ($P_{write}$ thường là 1.25x giá đọc cơ sở). Mỗi lần đọc trúng cache (*Cache Read Hit*), ta tiết kiệm được $(1 - \text{read\_discount}) \times P_{in}$. Do đó, điểm hòa vốn số lần đọc lặp lại ($N^*$) được xác định bởi:
  $$N^* = \frac{P_{write}}{P_{in} \times (1 - \text{read\_discount})} = \frac{1.25 \times P_{in}}{P_{in} \times 0.90} \approx 1.39 \text{ lần đọc}$$
* **Đo lường trên dữ liệu thực tế:**
  - Điểm hòa vốn cho cả hai tier `small` và `large` đều là **1.39 lần đọc**.
  - Trong thực tế log token của NimbusAI, các phiên hội thoại đa lượt và tài liệu RAG có tần suất tái sử dụng prefix trung bình là **5.2 lần**.
  - Vì $5.2 \ge 1.39$, hàm `cache_is_worth_it()` trả về `True`, cho phép áp dụng triệt để prompt caching.
  - **Insight:** Nếu ứng dụng có số lượt tái sử dụng prefix $< 1.39$ (ví dụ: các câu hỏi đơn lẻ không kèm ngữ cảnh cố định), việc bật caching sẽ gây lỗ ròng do phụ phí ghi cao hơn khoản tiết kiệm đọc.

### 4.3. Extension 5: Lập lịch nhận thức Carbon (Carbon-aware Scheduling)
* **Vị trí triển khai:** `missions/m3_purchasing.py` (kèm test tại `tests/test_extensions.py`)
* **Phương pháp:**
  Tổng điện năng tiêu thụ của 5 tác vụ huấn luyện/batch có thể ngắt quãng (`interruptible=1`) đạt **4,227.0 kWh/tháng**. Phân tích điều phối trên 5 vùng địa lý:

| Vùng triển khai (Region) | Giá điện ($/kWh) | Cường độ Carbon (gCO2/kWh) | Tiền điện / tháng | Phát thải Carbon / tháng | % Giảm Carbon |
|---|:---:|:---:|:---:|:---:|:---:|
| **europe-north1 (Na Uy)** | $0.090 | **30** | $380.43 | **126.8 kg CO2e** | **-92.1% (Sạch nhất)** |
| **us-east-wa (Washington)** | **$0.055** | 90 | **$232.49** | 380.4 kg CO2e | **-76.3% (Rẻ nhất)** |
| **us-west-2 (Oregon)** | $0.070 | 120 | $295.89 | 507.2 kg CO2e | -68.4% |
| **us-east-1 (Virginia - Gốc)** | $0.120 | 380 | $507.24 | 1,606.3 kg CO2e | 0.0% (Baseline) |
| **europe-central2 (Ba Lan)** | $0.180 | 660 | $760.86 | 2,789.8 kg CO2e | +73.7% (Bẩn nhất) |

* **Đánh giá Trade-off & Khuyến nghị:**
  - **Về Carbon:** Chuyển sang `europe-north1` giảm tới **1,479.5 kg CO2e/tháng** (-92.1%) nhờ nguồn điện thủy điện sạch của Bắc Âu.
  - **Về Chi phí điện:** Chuyển sang `us-east-wa` giảm tiền điện từ $507.24 xuống $232.49 (-54.2%).
  - **Về Độ trễ (Latency):** Do các job huấn luyện lớn và batch evaluation là các tác vụ xử lý phi đồng bộ, không có người dùng tương tác trực tiếp theo thời gian thực (end-user interaction latency = 0). Do đó, hoàn toàn **không có bất kỳ đánh đổi tiêu cực nào về trải nghiệm người dùng** khi chuyển tải công việc sang Na Uy hoặc Washington.

### 4.4. Extension 4 (Bổ sung): Quản lý Ngân sách Suy luận Reasoning
* Phân tích cho thấy traffic có cờ `is_reasoning=1` chỉ chiếm **8.4% số lượng request** (201/2,400), nhưng lại ngốn tới **16.5% chi phí** và **94.0% tổng điện năng suy luận** (29,788 Wh so với 1,888 Wh của toàn bộ 2,199 request thông thường).
* Lý do: Chế độ reasoning sinh chuỗi suy nghĩ nội tâm dài gấp nhiều chục lần, bắt GPU phải chạy vòng lặp decode kéo dài.
* Đề xuất quy tắc Routing: Áp dụng Classifier lọc trước request; chỉ cấp quyền reasoning mode cho các tác vụ code chuyên sâu hoặc bài toán toán học phức tạp, giúp cắt giảm ngay 50% chi phí reasoning và tiết kiệm ~14.9 kWh/ngày.

---

## 5. Ba khuyến nghị hành động cho Ban Lãnh đạo NimbusAI (Xếp theo ROI)

Dựa trên kết quả phân tích số liệu, dưới đây là lộ trình hành động FinOps được ưu tiên theo thời gian và tỷ suất hoàn vốn (ROI):

### 1. Hành động 1: Kích hoạt bộ 3 đòn bẩy Inference (ROI cao nhất — Thực hiện trong 24 giờ)
* **Việc cần làm:** Cấu hình gateway LiteLLM/vLLM bật cơ chế **Cascade Routing** (đẩy 70% request đơn giản sang model `small`), bật **Prompt Caching** cho các system prompt/RAG context, và phân luồng các tác vụ phi thời gian thực qua **Batch API** (chiết khấu 50%).
* **Chi phí đầu tư:** 0 USD phần cứng, chỉ cần chỉnh cấu hình API routing.
* **Hiệu quả:** Giảm ngay đơn giá phục vụ từ **$6.488 xuống $1.126/1M-token** (tiết kiệm 82.6% chi phí token).

### 2. Hành động 2: Tái cơ cấu Mua sắm GPU — Spot & Reserved Commitment (Tiết kiệm tiền mặt lớn nhất — Thực hiện trong tuần 1)
* **Việc cần làm:** 
  - Kích hoạt cơ chế checkpointing tự động (chu kỳ 30–60 phút) cho các job huấn luyện và chuyển ngay sang thuê **Spot Instance** (tiết kiệm ~40%).
  - Với các cụm GPU inference chạy dịch vụ chính (duty cycle $\ge 55\%$), ký hợp đồng **Reserved Instance 1–3 năm** để hưởng mức chiết khấu 45%.
* **Hiệu quả:** Tiết kiệm ngay lập tức **$10,040 USD tiền mặt mỗi tháng**.

### 3. Hành động 3: Chuẩn hóa Tagging phục vụ Chargeback & Điều phối Carbon-aware (Thực hiện trong tháng 1)
* **Việc cần làm:**
  - Thiết lập chính sách kiểm soát hạ tầng (Policy-as-Code): Bắt buộc mọi yêu cầu cấp phát GPU phải có tag `team` và `project`. Với độ phủ tag hiện tại đạt **92%** ($\ge 80\%$), chính thức chuyển từ *Showback* sang *Chargeback* (trừ ngân sách trực tiếp vào các đội R&D/Product) để nâng cao ý thức tiết kiệm tài nguyên.
  - Cấu hình Kubernetes scheduler điều phối toàn bộ các batch/training workloads sang cụm hạ tầng tại vùng **`europe-north1`** hoặc **`us-east-wa`**.
* **Hiệu quả:** Cắt giảm **92.1% lượng phát thải carbon** cho công ty, hạ chi phí tiền điện 54%, và triệt tiêu hoàn toàn lãng phí GPU chạy không tải ($600/tháng).
