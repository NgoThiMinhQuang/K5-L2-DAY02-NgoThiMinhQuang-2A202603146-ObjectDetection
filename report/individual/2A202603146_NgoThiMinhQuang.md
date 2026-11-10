# BÁO CÁO CÁ NHÂN — NGÀY 2: OBJECT DETECTION

## 1. Thông tin học viên

| Thông tin | Nội dung |
|---|---|
| Họ và tên | Ngô Thị Minh Quang |
| Mã số sinh viên | 2A202603146 |
| Lớp | L2A |
| Email | minhquang17012005@gmail.com |
| Repository | [GitHub — Day 2 Object Detection](https://github.com/NgoThiMinhQuang/K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection) |
| Google Colab | [Notebook huấn luyện YOLO26n](https://colab.research.google.com/drive/1gQec4pAkOKcSup2J4eYudVt2ceeXLETG?usp=sharing) |

## 2. Schema và quy tắc gán nhãn

### 2.1. Bộ nhãn sử dụng

Bài thực hành yêu cầu gán nhãn các phương tiện giao thông theo định dạng YOLO Detection, gồm bốn lớp đối tượng:

| Class ID | Tên lớp | Mô tả |
|---|---|---|
| 0 | `car` | Xe ô tô con |
| 1 | `truck` | Xe tải và các phương tiện chở hàng thuộc nhóm truck |
| 2 | `bus` | Xe buýt, xe khách thuộc nhóm bus |
| 3 | `van` | Xe van, minivan theo quy định gán nhãn |

Mỗi bounding box được lưu theo cấu trúc:

`class_id x_center y_center width height`

Các tọa độ được chuẩn hóa trong khoảng `[0, 1]`.

### 2.2. Nguyên tắc gán nhãn

Trong quá trình thực hiện, tôi áp dụng các nguyên tắc:

- Mỗi phương tiện được gán một bounding box riêng biệt.
- Bounding box cần ôm sát phần phương tiện có thể quan sát được.
- Với phương tiện bị che khuất hoặc cắt bởi rìa ảnh, chỉ gán khi có đủ đặc điểm nhận diện lớp.
- Không gán xe máy, xe đạp, người đi bộ, biển báo, đèn giao thông hoặc các đối tượng ngoài bốn lớp yêu cầu.
- Không tự suy đoán loại phương tiện khi hình ảnh không cung cấp đủ thông tin.

Chi tiết được ghi trong [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md).

### 2.3. Ba trường hợp khó

| Ảnh | Vấn đề | Hướng xử lý |
|---|---|---|
| `drive_002` | Một số phương tiện bị che khuất | Chỉ gán khi phần nhìn thấy đủ để xác định loại xe; box ôm phần nhìn thấy |
| `drive_003` | Nhiều phương tiện gần nhau tại giao lộ | Phân biệt từng phương tiện và tạo box riêng, không gộp nhiều xe |
| `drive_004` | Ánh sáng yếu, phương tiện ở xa hoặc bị mờ | Kiểm tra khả năng nhận diện lớp trước khi gán nhãn, tránh suy đoán |

Các trường hợp còn khó phân biệt gồm `van` với `bus` (đặc biệt minibus) và ranh giới `car` với `truck`; theo `GUIDELINE_MINI.md`, xe bán tải thông thường thuộc `car` (không phải `truck`).

## 3. Quá trình annotation và self-QC

### 3.1. Chuẩn bị dữ liệu

Bộ dữ liệu sử dụng trong bài gồm:

| Tập dữ liệu | Số lượng ảnh | Mục đích |
|---|---:|---|
| Train | 50 | Gán nhãn và huấn luyện YOLO26n |
| Eval | 10 | Validation và đánh giá mô hình |
| Tổng cộng | 60 | Toàn bộ dữ liệu của bài thực hành |

Bộ nhãn chính thức được lưu tại:

`dataset/labels/train/`

Gồm 50 file từ `drive_001.txt` đến `drive_050.txt`, tương ứng với 50 ảnh train.

### 3.2. Thực hiện gán nhãn

Tôi sử dụng CVAT để quan sát ảnh giao thông, kiểm tra bounding box và chỉnh sửa nhãn theo schema bốn lớp.

Trong quá trình rà soát, các vấn đề được chú ý gồm:

- Bounding box thừa hoặc trùng trên cùng phương tiện.
- Nhận diện nhầm giữa `car`, `truck`, `bus` và `van`.
- Bounding box lệch, quá rộng hoặc chưa ôm sát xe.
- Một số phương tiện nhỏ, xa hoặc bị che khuất chưa được nhận diện chính xác.
- Các đối tượng không thuộc phạm vi yêu cầu.

Tôi đã xem lại các ảnh từ `drive_001` đến `drive_010`, đồng thời tập trung rà soát thêm những ảnh có nhiều sai lệch như `drive_012`, `drive_024`, `drive_027`, `drive_030` và `drive_041`.

### 3.3. Kiểm tra cấu trúc nhãn

Sau khi hoàn thành gán nhãn, tôi sử dụng công cụ kiểm tra có sẵn trong repository:

```bash
python tools/check_yolo_labels.py --labels dataset/labels/train
```

**Kết quả: ĐẠT cấu trúc 50/50 file nhãn.**

Các file đáp ứng các yêu cầu kỹ thuật: đúng định dạng YOLO Detection, mỗi dòng có năm giá trị, class ID thuộc khoảng 0–3 và tọa độ chuẩn hóa hợp lệ.

### 3.4. Commit khóa nhãn

- Commit khóa nhãn ban đầu (hoàn thành 50 nhãn train, trước commit finalize/eval với Gold QC): **`9c06e50`** (`feat: complete initial ADAS annotations`, thêm đầy đủ 50 file `dataset/labels/train/drive_001.txt`–`drive_050.txt`).
- Commit hoàn thiện bộ nhãn được sử dụng để train: **`f2af7e30ef063a4be3f68abd07b2a80131e5d58d`** (`feat: finalize ADAS annotations and evaluation`).

Hai mốc commit được phân biệt để thể hiện lịch sử gán nhãn ban đầu và bản hoàn thiện sau rà soát/đánh giá.

## 4. Gold-set QC và chỉnh sửa annotation

### 4.1. Phương pháp đánh giá

Tôi sử dụng công cụ `tools/evaluate_annotations.py` để đánh giá chất lượng nhãn so với Gold Set.

Một bounding box được tính là True Positive (TP) khi:

- Class dự đoán trùng với class của Gold Set.
- IoU giữa hai bounding box đạt ít nhất 0.50.
- Mỗi box chỉ được ghép với tối đa một box tương ứng theo quy tắc matching của evaluator.

Các chỉ số được sử dụng gồm Precision, Recall, F1-score và Mean IoU của các box ghép đúng.

### 4.2. Kết quả đánh giá trước/sau rework

- **Commit khóa nhãn ban đầu (xác minh):** `9c06e50` (`feat: complete initial ADAS annotations`).
- **Commit hoàn thiện sau rework/QC:** `f2af7e30ef063a4be3f68abd07b2a80131e5d58d` (nguồn: `outputs/annotation_evaluation.json`).

| Mốc | Precision | Recall | F1-score | Mean IoU | `passed` (evaluator gốc) |
|---|---:|---:|---:|---:|---|
| Trước rework (`9c06e50`) | — | — | — | — | — |
| Sau rework (`f2af7e3`) | 0.4226 | 0.7228 | 0.5333 | 0.8446 | `false` |

**Lưu ý:** Không có file/log kết quả `annotation_evaluation` (hoặc số liệu Precision/Recall/F1/Mean IoU tương đương) trước rework được lưu/xác minh trong Git history (tại `9c06e50` không tồn tại `outputs/annotation_evaluation.json`; file này được thêm tại `f2af7e3`). Do đó, **không có dữ liệu số liệu trước rework được xác minh** để điền — không tự tạo số liệu.

Kết quả sau rework (giữ nguyên, theo evaluator gốc): TP 472, FP 645, FN 181, Precision 0.4226, Recall 0.7228, F1-score 0.5333, Mean IoU 0.8446, `passed=false` (ngưỡng F1 ≥ 0.80, Mean IoU ≥ 0.70). Mean IoU vượt ngưỡng 0.70, F1 chưa đạt 0.80.

**Theo thông báo của giảng viên được cung cấp trong quá trình thực hiện, học viên được phép tiếp tục bước huấn luyện với F1 trên 0.50.** Do đó, tôi giữ nguyên kết quả đánh giá thực tế và tiếp tục thực hiện notebook Colab mà không sửa giá trị `passed` do evaluator tạo ra.

### 4.3. Phân tích kết quả

Mean IoU đạt 0.8446 cho thấy các bounding box được ghép đúng thường có độ trùng khớp hình học tương đối tốt.

Tuy nhiên, Precision chỉ đạt 0.4226, phản ánh số lượng box không ghép được với Gold Set còn lớn. Bên cạnh đó, Recall đạt 0.7228, cho thấy vẫn tồn tại các đối tượng chưa được ghép đúng.

Các nguyên nhân có khả năng ảnh hưởng đến kết quả gồm nhầm loại phương tiện, box thừa hoặc trùng, đối tượng quá nhỏ, xe bị che khuất và sự khác biệt trong cách áp dụng guideline đối với các trường hợp mơ hồ.

Tôi thực hiện rà soát và chỉnh sửa dựa trên ảnh gốc cùng quy tắc gán nhãn. Gold Set chỉ được sử dụng phục vụ kiểm tra và phân tích kết quả, không sao chép thành dữ liệu nhãn nộp bài.

## 5. Huấn luyện và đánh giá YOLO26n trên Google Colab

### 5.1. Cấu hình thực nghiệm

Sau khi hoàn thiện bước annotation, tôi tiến hành huấn luyện mô hình YOLO26n thông qua notebook `notebooks/day2_annotation_to_yolo.ipynb`.

| Thành phần | Cấu hình |
|---|---|
| Mô hình | YOLO26n |
| Checkpoint khởi tạo | `yolo26n.pt` |
| Môi trường | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Số epochs | 50 |
| Kích thước ảnh | 640 × 640 |
| Batch size | Tự động (`batch=-1`) |
| Seed | 20260909 |
| Số ảnh train | 50 |
| Số ảnh validation/eval | 10 |
| Class | car, truck, bus, van |
| Commit dữ liệu | `f2af7e3` |

Mô hình được huấn luyện trên 50 ảnh trong `dataset/images/train` và các nhãn tương ứng.

Mười ảnh `drive_051.jpg` đến `drive_060.jpg` được sử dụng làm validation theo cấu hình notebook, đồng thời để đánh giá kết quả cuối trên checkpoint tốt nhất.

**Lưu ý:** Vì tập eval cũng được sử dụng trong quá trình validation và lựa chọn `best.pt`, kết quả trên 10 ảnh này không được xem là đánh giá trên một tập test độc lập hoàn toàn.

### 5.2. Kết quả huấn luyện

Quá trình huấn luyện hoàn thành 50/50 epochs và tạo checkpoint `best.pt`.

Kết quả đánh giá mô hình trên 10 ảnh eval:

| Chỉ số | Giá trị |
|---|---:|
| **Precision** | **0.4105** |
| **Recall** | **0.2303** |
| **mAP@50** | **0.2723** |
| **mAP@50:95** | **0.2017** |
| Số ảnh eval | 10 |

Đây là các chỉ số đánh giá **mô hình Object Detection sau huấn luyện**, khác với F1-score và Mean IoU được sử dụng để đánh giá chất lượng nhãn annotation ở Mục 4.

### 5.3. Kết quả theo từng lớp

Theo log validation của checkpoint tốt nhất:

| Lớp | Số đối tượng trong eval | Precision | Recall | mAP50 |
|---|---:|---:|---:|---:|
| car | 94 | 0.833 | 0.637 | 0.740 |
| truck | 5 | 0.677 | 0.200 | 0.239 |
| bus | 2 | 0.000 | 0.000 | 0.0079 |
| van | 12 | 0.135 | 0.083 | 0.103 |

*Các giá trị được làm tròn từ log validation khi hoàn tất 50 epochs.*

**Nhận xét theo lớp:**

- **Car:** Là lớp có hiệu quả tốt nhất trong bốn lớp, với Recall 0.637 và mAP50 khoảng 0.740. Mô hình đã học được tương đối tốt đặc điểm của ô tô con.
- **Truck:** Recall chỉ đạt 0.200, cho thấy khả năng phát hiện đầy đủ xe tải còn hạn chế.
- **Bus:** Recall bằng 0.000, thấp nhất trong bốn lớp. Tập eval chỉ có hai đối tượng bus nên chỉ số này cần được diễn giải thận trọng.
- **Van:** Recall đạt 0.083, thể hiện khó khăn trong việc phát hiện và phân biệt van với các phương tiện có hình dạng tương tự.

Nhìn chung, mô hình hoạt động tốt hơn trên lớp car nhưng chưa ổn định đối với những lớp ít đối tượng.

### 5.4. Minh chứng thực nghiệm

**Biểu đồ quá trình huấn luyện**

![Biểu đồ kết quả huấn luyện](assets/yolo26_runs/vehicle_annotation/results.png)

**Confusion matrix**

![Confusion Matrix](assets/yolo26_runs/vehicle_annotation/confusion_matrix.png)

**Precision–Recall Curve**

![Precision–Recall Curve](assets/yolo26_runs/vehicle_annotation/BoxPR_curve.png)

**Một số ảnh dự đoán trên tập eval**

![Dự đoán drive_051](assets/eval_predictions/visuals/drive_051.jpg)

![Dự đoán drive_053](assets/eval_predictions/visuals/drive_053.jpg)

![Dự đoán drive_056](assets/eval_predictions/visuals/drive_056.jpg)

Các minh chứng bổ sung:

- [Kết quả theo epoch — results.csv](assets/yolo26_runs/vehicle_annotation/results.csv)
- [Confusion matrix chuẩn hóa](assets/yolo26_runs/vehicle_annotation/confusion_matrix_normalized.png)
- [Precision–Confidence Curve](assets/yolo26_runs/vehicle_annotation/BoxP_curve.png)
- [Recall–Confidence Curve](assets/yolo26_runs/vehicle_annotation/BoxR_curve.png)
- [F1–Confidence Curve](assets/yolo26_runs/vehicle_annotation/BoxF1_curve.png)
- [Kết quả đánh giá mô hình — JSON](assets/model_eval_metrics.json)

### 5.5. Phân tích kết quả

Mặc dù mô hình hoàn thành huấn luyện 50 epochs, mAP50 và Recall tổng thể vẫn ở mức thấp.

Một số yếu tố có thể ảnh hưởng:

- Bộ train chỉ gồm 50 ảnh, hạn chế độ đa dạng phương tiện và điều kiện quan sát.
- Số lượng đối tượng giữa các lớp không cân bằng, trong đó car chiếm ưu thế.
- Xe ở xa, nhỏ, bị che khuất hoặc có ánh sáng phức tạp gây khó khăn cho nhận diện.
- Các loại phương tiện có đặc điểm hình học gần nhau, đặc biệt van, bus và truck.
- Nhãn train vẫn còn những sai lệch được thể hiện qua kết quả Gold QC.

Kết quả cho thấy khả năng phát hiện và phân loại đối tượng phụ thuộc đáng kể vào chất lượng, số lượng và mức độ đa dạng của dữ liệu annotation.

## 6. Trả lời câu hỏi phân tích

### Câu 1. Lớp nào có Recall thấp nhất? Chỉ ra ảnh eval minh họa.

Lớp **bus (class 2)** có Recall thấp nhất, bằng **0.000** trong log validation của checkpoint tốt nhất.

Tập eval gồm hai đối tượng bus và mô hình không ghi nhận True Positive cho lớp này ở kết quả validation tương ứng.

Có thể sử dụng ảnh `drive_053.jpg` để minh họa các phương tiện bus trong cảnh giao thông, kết hợp đối chiếu giữa nhãn thực tế và ảnh dự đoán để phân tích các trường hợp phát hiện hoặc phân loại chưa chính xác.

Minh chứng:

- [Ảnh dự đoán drive_053](assets/eval_predictions/visuals/drive_053.jpg)
- [Validation labels](assets/yolo26_runs/vehicle_annotation/val_batch0_labels.jpg)
- [Validation predictions](assets/yolo26_runs/vehicle_annotation/val_batch0_pred.jpg)

Lưu ý rằng một nhãn bus xuất hiện trong ảnh dự đoán không nhất thiết được tính là True Positive nếu không đáp ứng đồng thời điều kiện đúng class và IoU của phép đánh giá.

### Câu 2. Nếu mAP50 cao nhưng mAP50–95 thấp, bounding box đang gặp vấn đề gì?

mAP50 đo chất lượng phát hiện với ngưỡng IoU 0.50, trong khi mAP50–95 lấy trung bình nhiều ngưỡng IoU từ 0.50 đến 0.95.

Khi mAP50 lớn hơn đáng kể mAP50–95, một nguyên nhân có thể là mô hình đã xác định được khu vực chứa đối tượng nhưng tọa độ bounding box chưa đủ chính xác ở các ngưỡng IoU cao.

Trong thực nghiệm này:

- mAP50 = 0.2723.
- mAP50–95 = 0.2017.

Sự chênh lệch cho thấy hiệu năng giảm khi áp dụng các yêu cầu khắt khe hơn về độ trùng khớp, mặc dù kết quả còn chịu ảnh hưởng của cả độ tin cậy dự đoán và khả năng phân loại.

### Câu 3. Có lỗi phân biệt truck/bus/van/car nào liên quan đến guideline không?

Có. Một số nhóm phương tiện có hình dạng tương tự, dẫn đến nguy cơ nhầm lẫn:

- `van` với `bus` ở các phương tiện minibus.
- `van` với `car` ở các xe MPV hoặc xe thân kín.
- `car` với `truck` ở ranh giới xe chở hàng nhẹ; theo `GUIDELINE_MINI.md`, xe bán tải thông thường thuộc `car` (không phải `truck`).
- `truck` với `bus` trong các cảnh có góc quan sát hoặc độ phân giải hạn chế.

Theo guideline, cần quan sát cấu trúc thân xe và các đặc điểm có thể nhận diện trước khi quyết định class. Không tự suy đoán phương tiện quá mờ hoặc bị che khuất nghiêm trọng.

Những trường hợp chưa nhất quán trong annotation có thể ảnh hưởng đến quá trình mô hình học các đặc trưng phân loại.

### Câu 4. Vì sao không dùng 10 ảnh eval để train dù đã có nhãn?

Tập eval được giữ riêng nhằm đánh giá chất lượng dự đoán của mô hình trên những ảnh không được sử dụng trực tiếp để cập nhật trọng số trong quá trình train.

Nếu đưa các ảnh eval vào train, mô hình có thể học thuộc đặc điểm của chúng, khiến kết quả đánh giá trở nên lạc quan và không phản ánh đúng khả năng tổng quát.

Vì vậy, tôi giữ nguyên cách chia:

- 50 ảnh train để huấn luyện.
- 10 ảnh eval để validation và đánh giá theo notebook.

Do tập eval cũng được sử dụng để lựa chọn checkpoint tốt nhất, cần tránh coi đây là một tập test độc lập.

## 7. Khó khăn, kết quả đạt được và tự đánh giá

### 7.1. Khó khăn và cách xử lý

| Vấn đề | Cách xử lý | Kết quả |
|---|---|---|
| Sai lệch annotation và FP cao | Kiểm tra lại bounding box, class và các đối tượng ngoài phạm vi | F1 đạt 0.5333, Mean IoU đạt 0.8446 |
| Xe nhỏ, xa, che khuất | Quan sát ảnh gốc và áp dụng guideline khi gán nhãn | Cải thiện một phần chất lượng nhãn nhưng vẫn còn sai lệch |
| Nhầm car, truck, bus, van | Kiểm tra đặc điểm phương tiện và rà các trường hợp mơ hồ | Một số lỗi được sửa, chưa loại bỏ hoàn toàn nhầm lớp |
| Kết quả YOLO26n chưa cao | Giữ đúng split dữ liệu, đánh giá từng class và phân tích kết quả | mAP50 đạt 0.2723, mAP50–95 đạt 0.2017 |

### 7.2. Tự đánh giá theo các nhóm tiêu chí rubric

| Nhóm tiêu chí | Kết quả thực hiện |
|---|---|
| Định dạng và cấu trúc YOLO | Đạt cấu trúc 50/50 file |
| Độ đầy đủ đối tượng | Recall annotation 0.7228; vẫn có FN |
| Độ chính xác hình học | Mean IoU của các match 0.8446 |
| Tính nhất quán phân lớp | Còn một số nhầm lẫn car/truck/bus/van |
| Mini guideline | Có quy tắc và ba trường hợp khó |
| Gold-set QC | F1 0.5333, Mean IoU 0.8446; evaluator trả `passed=false` theo ngưỡng gốc |
| Huấn luyện và đánh giá YOLO26n | Hoàn thành 50 epochs, có metrics, biểu đồ và 10 ảnh dự đoán |

Bảng này chỉ tổng hợp mức độ hoàn thành, không tự quy đổi thành điểm chính thức của giảng viên.

### 7.3. Kết luận

Qua bài thực hành, tôi đã thực hiện quy trình từ gán nhãn ảnh giao thông, kiểm tra cấu trúc YOLO, đánh giá chất lượng annotation đến huấn luyện và đánh giá mô hình YOLO26n.

Bộ nhãn hoàn thành đủ 50 ảnh và đạt yêu cầu cấu trúc. Kết quả Gold QC có F1 = 0.5333 và Mean IoU = 0.8446. Sau đó, mô hình YOLO26n được huấn luyện trong 50 epochs trên Google Colab với GPU Tesla T4.

Trên tập validation 10 ảnh, mô hình đạt mAP50 = 0.2723 và mAP50–95 = 0.2017. Lớp car có kết quả tốt nhất, trong khi truck, bus và van còn hạn chế.

Qua kết quả thực nghiệm, tôi nhận thấy chất lượng annotation, độ cân bằng giữa các lớp và sự đa dạng của dữ liệu đóng vai trò quan trọng đối với khả năng phát hiện đối tượng. Đây là những hướng cần tiếp tục cải thiện nếu mở rộng bài thực hành.

## 8. Công cụ hỗ trợ và tài liệu tham khảo

### 8.1. Công cụ sử dụng

- **CVAT:** Gán nhãn và chỉnh sửa bounding box.
- **Python:** Kiểm tra cấu trúc, xử lý dữ liệu và đánh giá annotation.
- **Ultralytics YOLO26n:** Huấn luyện và đánh giá mô hình Object Detection.
- **Google Colab:** Môi trường thực thi notebook trên GPU.
- **Git/GitHub:** Quản lý phiên bản và lưu trữ bài nộp.

Tôi thực hiện và rà soát phần annotation trên CVAT, trực tiếp chạy notebook huấn luyện trên Colab, kiểm tra các chỉ số thực nghiệm và chịu trách nhiệm về kết quả được nộp.

### 8.2. Tài liệu tham khảo

1. `GUIDE.md` — Hướng dẫn thực hiện bài lab.
2. `RULES.md` — Quy định thực hành.
3. `CHECKPOINTS.md` — Các mốc kiểm tra.
4. `SUBMISSION.md` — Quy định nộp bài.
5. `RUBRIC.md` — Tiêu chí đánh giá.
6. `GUIDELINE_MINI.md` — Quy tắc gán nhãn và các trường hợp khó.
7. [Ultralytics Documentation](https://docs.ultralytics.com/) — Tài liệu huấn luyện và đánh giá mô hình YOLO.

**Cam kết dữ liệu:** Không đưa Gold Set, thông tin bảo mật hoặc checkpoint `.pt` vào repository nộp bài. Các chỉ số được trình bày theo kết quả đánh giá thực tế, không thay đổi kết quả evaluator để tạo trạng thái đạt giả.
