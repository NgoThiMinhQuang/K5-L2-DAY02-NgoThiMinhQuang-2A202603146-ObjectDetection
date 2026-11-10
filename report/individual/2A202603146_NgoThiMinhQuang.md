# Báo cáo cá nhân — Ngày 2 Object Detection

## 1. Thông tin học viên

- Họ tên: Ngô Thị Minh Quang
- MSSV: 2A202603146
- Email: *(bổ sung theo thông tin cá nhân khi nộp)*
- Lớp: *(bổ sung theo lớp thực tế)*
- Link repo bài làm: https://github.com/NgoThiMinhQuang/K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection
- Link Colab đã chạy: *(điền link Colab thực tế đã chạy, cấp quyền xem cho người chấm)*

## 2. Schema và mini guideline

- Schema (`data.yaml`, 4 lớp, ID cố định): `0=car`, `1=truck`, `2=bus`, `3=van`.
- Tham chiếu [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md) — định nghĩa lớp, quy tắc box và 3 ca mơ hồ.
- 3 ca mơ hồ đã ghi/quyết định:
  - `drive_002` (che khuất một phần): gán nhãn từng xe nếu phần nhìn thấy đủ xác định lớp; box chỉ bao phần nhìn thấy (không suy đoán phần bị che).
  - `drive_003` (nhiều phương tiện gần nhau tại giao lộ): vẽ box riêng cho từng xe thuộc 4 lớp, không gộp nhiều phương tiện.
  - `drive_004` (ban đêm, ánh sáng yếu, xe xa/mờ): chỉ gán khi nhận diện được loại xe; không tự đoán với phương tiện quá mờ/không đủ đặc điểm (đặc biệt `van` mơ hồ cần đối chiếu kỹ).
- Phần cần giảng viên làm rõ thêm: ranh giới `van`↔`bus` (minibus) và `car`↔`truck` (pickup/xe bán tải) trong cảnh xa/mờ.

## 3. Annotation và self-QC

- Số file nhãn train: **50/50** (`dataset/labels/train/drive_001.txt`–`drive_050.txt`, khớp basename ảnh train).
- Kết quả checker và lỗi đã xử lý: `python tools/check_yolo_labels.py --labels dataset/labels/train` → **ĐẠT cấu trúc 50/50** (đúng 5 cột, class 0–3, tọa độ chuẩn hóa [0,1], box hợp lệ). Đã rà/sửa nhiều vòng (CVAT + kiểm tra ảnh gốc) để giảm box thừa/trùng, siết box, bổ sung/đổi class theo guideline.
- Danh sách ít nhất 10 ảnh đã rà bằng mắt: `drive_001, drive_002, drive_003, drive_004, drive_005, drive_006, drive_007, drive_008, drive_009, drive_010` (ưu tiên ảnh nhiều xe/che khuất/ban đêm/giao lộ dày đặc; đã rà thêm các ảnh lỗi cao như `drive_012, drive_024, drive_027, drive_030, drive_041`).
- Lỗi bỏ sót/box/class phát hiện khi rà và cách sửa: box thừa/FP (gán xe nhỏ/ngoài phạm vi hoặc trùng), nhầm class `van/bus/truck/car`, box lệch/không sát; xử lý bằng xóa box thừa/trùng, đổi class theo đặc điểm xe và guideline, chỉnh box ôm sát phần nhìn thấy, bổ sung xe bị sót khi nhận diện rõ. Không gán đối tượng ngoài 4 lớp.
- Commit khóa nhãn ban đầu: *(điền hash commit đã khóa/nộp trước khi nhận gold theo mốc lớp; tham chiếu `git log`/GitHub)*

## 4. Gold QC và rework

Kết quả đánh giá annotation (cùng class, IoU≥0.50, one-to-one greedy theo `tools/evaluate_annotations.py`) trên bộ nhãn hiện tại (`outputs/annotation_evaluation.json`):

| Chỉ số | Giá trị |
|---|---:|
| TP | 472 |
| FP | 645 |
| FN | 181 |
| Precision | 0.4226 |
| Recall | 0.7228 |
| F1 | **0.5333** |
| mean_iou_of_matches | **0.8446** |
| passed (ngưỡng evaluator 0.80/0.70) | **false** |

- Mean IoU của các match đạt ngưỡng (0.8446 ≥ 0.70), cho thấy box khớp có độ bám sát tương đối tốt; F1 chưa đạt ngưỡng evaluator cũ (0.5333 < 0.80), chủ yếu do **FP cao** và nhầm class `van/bus/truck`/bỏ sót một số xe khó. **Giảng viên đã cho phép tiếp tục với F1 > 0.50.**
- Rework: đã thực hiện nhiều vòng rà/sửa thủ công trên CVAT (ưu tiên ảnh lỗi cao), kiểm tra bằng `check_yolo_labels.py` và `evaluate_annotations.py` (Gold chỉ dùng để đánh giá, không dùng để sao chép/chỉnh nhãn). Lý do còn FP/FN: xe nhỏ/xa/che khuất, ranh giới `van`↔`bus`/`truck`↔`car` mơ hồ trong ảnh ADAS; không suy nguyên nhân chỉ từ FP/FN mà đối chiếu trực tiếp trên ảnh gốc theo guideline.

## 5. Kết quả YOLO26n (Colab, thực tế)

- Model: **YOLO26n** (`yolo26n.pt`), train trên **50 ảnh** `dataset/images/train` với nhãn đã QC; **không** đưa `eval` vào train.
- Eval/validation: **10 ảnh** `dataset/images/eval` (`drive_051.jpg`–`drive_060.jpg`) — dùng làm `val` trong notebook/chọn `best.pt` (không phải test set độc lập).
- Epochs: **50** | Train images: **50** | Eval images: **10** | Runtime: Google Colab, **GPU Tesla T4** (commit `f2af7e3`).
- Kết quả đọc từ `assets/model_eval_metrics.json` (best checkpoint, `eval_images=10`):

| Chỉ số trên 10 ảnh eval | Giá trị |
|---|---:|
| Precision | 0.4105 |
| Recall | 0.2303 |
| mAP50 | 0.2723 |
| mAP50–95 | 0.2017 |

- Minh chứng (link tương đối):
  - 10 ảnh dự đoán: [assets/eval_predictions/visuals/drive_051.jpg](assets/eval_predictions/visuals/drive_051.jpg) … [assets/eval_predictions/visuals/drive_060.jpg](assets/eval_predictions/visuals/drive_060.jpg)
  - Log/biểu đồ train: [assets/yolo26_runs/vehicle_annotation/results.csv](assets/yolo26_runs/vehicle_annotation/results.csv), [assets/yolo26_runs/vehicle_annotation/results.png](assets/yolo26_runs/vehicle_annotation/results.png), [assets/yolo26_runs/vehicle_annotation/BoxP_curve.png](assets/yolo26_runs/vehicle_annotation/BoxP_curve.png), [assets/yolo26_runs/vehicle_annotation/BoxR_curve.png](assets/yolo26_runs/vehicle_annotation/BoxR_curve.png), [assets/yolo26_runs/vehicle_annotation/BoxPR_curve.png](assets/yolo26_runs/vehicle_annotation/BoxPR_curve.png), [assets/yolo26_runs/vehicle_annotation/BoxF1_curve.png](assets/yolo26_runs/vehicle_annotation/BoxF1_curve.png), [assets/yolo26_runs/vehicle_annotation/confusion_matrix.png](assets/yolo26_runs/vehicle_annotation/confusion_matrix.png), [assets/yolo26_runs/vehicle_annotation/confusion_matrix_normalized.png](assets/yolo26_runs/vehicle_annotation/confusion_matrix_normalized.png)
  - Đánh giá eval: [assets/yolo26_runs/vehicle_eval/](assets/yolo26_runs/vehicle_eval/) (confusion matrix, PR/P/R/F1 curves, val batch labels/pred)

### Phân tích ưu/nhược điểm theo từng lớp (car, truck, bus, van)

*Phân tích định tính dựa trên kết quả thực tế (Precision 0.4105, Recall 0.2303, mAP50 0.2723, mAP50–95 0.2017), confusion matrix train/eval và ảnh dự đoán trên 10 ảnh eval; tập train 50 ảnh lệch phân bố lớp (`car` chiếm đa số, `truck/bus/van` ít).*

| Lớp | Ưu điểm (quan sát) | Nhược điểm / lỗi thường gặp |
|---|---|---|
| **car (0)** | Lớp có nhiều mẫu nhất nên mô hình nhận diện tương đối tốt hơn các lớp còn lại; nhiều dự đoán đúng xuất hiện trên ảnh eval (xem visuals/confusion matrix). | Vẫn có FP/FN: nhầm với `van`/`truck` ở góc xa, xe nhỏ hoặc che khuất; box localization chưa chặt ở IoU cao. |
| **truck (1)** | Khi đặc điểm rõ (thùng/kích thước lớn, gần camera) có thể phát hiện đúng trong một số trường hợp. | Recall thấp do ít mẫu train và dễ nhầm với `car` (pickup/xe lớn) hoặc `bus`; thường bị bỏ sót trong cảnh đông/xa. |
| **bus (2)** | Xe buýt lớn, đặc trưng rõ ở một số ảnh giúp tách khỏi `car` khi đủ bối cảnh. | Recall thấp, mẫu hiếm và hay nhầm với `van`/minibus hoặc `truck`; nhạy với che khuất/biến dạng kích thước. |
| **van (3)** | Một số trường hợp nhận dạng đúng khi hình dáng van rõ ràng. | Lớp khó/hiếm nhất: dễ nhầm với `car` (MPV/SUV) và `bus` (minibus), biên giới guideline mơ hồ; recall thấp và dễ bỏ sót. |

**Nhận xét chung:** mAP/recall tổng thấp phản ánh dữ liệu train nhỏ và lệch lớp; mô hình thiên về `car`, còn `truck/bus/van` kém ổn định (bỏ sót/nhầm chéo). Khoảng cách mAP50 > mAP50–95 cho thấy localization ở ngưỡng IoU cao chưa tốt.

### Trả lời 4 câu hỏi phân tích

1. **Lớp nào có recall thấp nhất? Chỉ ra một ảnh eval minh họa.**  
   Dựa trên confusion matrix ([assets/yolo26_runs/vehicle_annotation/confusion_matrix.png](assets/yolo26_runs/vehicle_annotation/confusion_matrix.png), [assets/yolo26_runs/vehicle_annotation/confusion_matrix_normalized.png](assets/yolo26_runs/vehicle_annotation/confusion_matrix_normalized.png) và [assets/yolo26_runs/vehicle_eval/confusion_matrix.png](assets/yolo26_runs/vehicle_eval/confusion_matrix.png)), recall tổng thể thấp (0.2303) và các lớp hiếm **`truck`, `bus`, `van`** có recall thấp hơn rõ rệt so với **`car`** (tập train lệch mạnh về `car`, số mẫu `truck/bus/van` ít, dẫn đến bỏ sót/nhầm nhiều). Trong số này, các lớp hiếm (đặc biệt `van`/`bus` hoặc `truck` tùy ảnh) nằm trong nhóm recall thấp nhất. Ảnh minh họa: [assets/eval_predictions/visuals/drive_054.jpg](assets/eval_predictions/visuals/drive_054.jpg) (cảnh nhiều xe/đa lớp, thể hiện bỏ sót hoặc nhầm các xe không phải `car`), đối chiếu [assets/yolo26_runs/vehicle_eval/val_batch0_labels.jpg](assets/yolo26_runs/vehicle_eval/val_batch0_labels.jpg) vs [assets/yolo26_runs/vehicle_eval/val_batch0_pred.jpg](assets/yolo26_runs/vehicle_eval/val_batch0_pred.jpg).

2. **Nếu mAP50 cao nhưng mAP50–95 thấp, box đang gặp vấn đề gì?**  
   Kết quả thực tế cho thấy mAP50 (**0.2723**) cao hơn mAP50–95 (**0.2017**). Điều này nghĩa mô hình phát hiện đúng vùng tương đối ở IoU ~0.5 nhưng **localization chưa chặt ở ngưỡng IoU cao**: box lệch biên, chưa ôm sát thân xe hoặc kích thước/tỷ lệ chưa khớp tốt, nên khi đánh giá ở IoU 0.75–0.95 hiệu năng giảm rõ.

3. **Có lỗi `truck`/`bus`/`van`/`car` nào liên quan tới guideline không?**  
   Có. Theo guideline và quan sát thực nghiệm: nhầm **`van`↔`bus`** (minibus/xe chở khách cỡ trung), **`van`↔`car`/`truck`** (MPV/SUV/pickup), **`truck`↔`bus`** và **`car`↔`truck`** xuất hiện khi ranh giới hình thái không rõ (xa, che khuất, ban đêm). Việc chỉ gán khi nhận diện được lớp và box ôm phần nhìn thấy (theo guideline) cũng ảnh hưởng đến recall với xe nhỏ/mờ. Các lỗi này nhất quán với nhầm class trong annotation và confusion matrix của mô hình.

4. **Vì sao không dùng 10 ảnh eval để train dù chúng có nhãn?**  
   10 ảnh eval dùng để **đánh giá** (và trong notebook làm validation để chọn `best.pt`) nhằm ước lượng khả năng tổng quát trên dữ liệu chưa thấy; nếu đưa vào train, mô hình sẽ “nhớ” tập này và các chỉ số (Precision/Recall/mAP) trở nên lạc quan, không phản ánh đúng chất lượng thực tế. Vì vậy phải giữ nguyên split **50 train / 10 eval**, **không** dùng eval để train.

## 6. Khó khăn và cách xử lý

| Vấn đề | Cách xử lý | Kết quả |
|---|---|---|
| F1 annotation thấp (FP cao, nhầm `van/bus/truck`) | Rà/sửa thủ công nhiều vòng trên CVAT theo ảnh gốc + guideline, check/eval (Gold chỉ để chấm) | F1 **0.5333**, Mean IoU **0.8446** (passed=false theo ngưỡng 0.80, GV cho phép F1>0.50) |
| `van/bus/truck` mơ hồ, xe nhỏ/xa/che khuất | Ưu tiên box rõ, không đoán bừa; siết box phần nhìn thấy; đổi/xóa class/box sai khi có căn cứ ảnh | Giảm một phần lỗi, vẫn còn nhầm class/FP |
| Model recall/mAP thấp (50 ảnh train, 10 eval, lệch lớp) | Giữ đúng split, không dùng eval train; phân tích kết quả thực nghiệm và confusion matrix | P 0.4105 / R 0.2303 / mAP50 0.2723 / mAP50–95 0.2017 |

## 7. Tự đánh giá theo rubric

- Hoàn thành/định dạng YOLO (10đ): 50/50 file, 5 cột, normalized, class 0–3 — **đạt cấu trúc**.
- Đầy đủ vật thể (20đ): recall annotation 0.7228, vẫn còn FN/FP — **cần cải thiện**.
- Độ bám sát/hình học (20đ): mean IoU match 0.8446 — **khá tốt**.
- Phân lớp nhất quán (15đ): còn nhầm `van/bus/truck/car` — **hạn chế**.
- Mini guideline + 3 ca mơ hồ (10đ): có trong `GUIDELINE_MINI.md`.
- Gold-set QC (15đ): `F1=0.5333` (chưa đạt 0.80), `mean IoU=0.8446` (đạt 0.70); GV cho phép tiếp tục với F1>0.50; có kết quả đánh giá thực tế.
- Pipeline YOLO & phân tích (10đ): Colab chạy 50 epoch, giữ split, có metrics/ảnh eval và nhận xét lỗi thực nghiệm.

## 8. Công cụ AI và tài liệu tham khảo

- Công cụ AI đã dùng: **OpenCode (AI coding assistant)** hỗ trợ kiểm tra cấu trúc, chạy/evaluating script, phân tích lỗi, chuẩn bị tài liệu/báo cáo và thao tác repo. **Tự kiểm tra/chịu trách nhiệm:** gán nhãn/sửa nhãn thủ công trên CVAT theo ảnh gốc, quyết định class/box theo guideline, chạy/kiểm tra Colab và số liệu, nội dung nộp cuối.
- Tài liệu: `GUIDE.md`, `RULES.md`, `CHECKPOINTS.md`, `SUBMISSION.md`, `RUBRIC.md`, tài liệu Ultralytics/YOLO26. Không đưa `gold_train_labels/`, secret/token hay checkpoint `.pt` vào repo.
