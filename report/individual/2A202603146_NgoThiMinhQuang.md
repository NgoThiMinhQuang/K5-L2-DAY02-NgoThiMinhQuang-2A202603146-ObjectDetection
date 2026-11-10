# Báo cáo cá nhân — Ngày 2 Object Detection

## 1. Thông tin học viên

- Họ tên: Ngô Thị Minh Quang
- MSSV: 2A202603146
- Email: minhquang17012005@gmail.com *(theo thông tin làm việc trên CVAT local; điều chỉnh nếu cần)*
- Lớp: *(điền theo lớp thực tế)*
- Link repo bài làm: https://github.com/NgoThiMinhQuang/K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection
- Link Colab đã chạy: *(chưa chạy — xem mục 5 và hướng dẫn Colab)*

## 2. Schema và mini guideline

Tôi đã tham chiếu [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md). Tóm tắt 3 ca mơ hồ tôi đã ghi nhận/quyết định:
- `drive_002` (che khuất một phần): gán nhãn từng xe nếu phần nhìn thấy đủ xác định lớp; box chỉ bao phần nhìn thấy (đúng guideline về occlusion/cắt khung).
- `drive_003` (nhiều phương tiện gần nhau tại giao lộ): vẽ box riêng cho từng xe thuộc 4 lớp, không gộp nhiều phương tiện.
- `drive_004` (ban đêm, ánh sáng yếu, xe xa/mờ): chỉ gán khi nhận diện được loại xe; không tự đoán với phương tiện quá mờ/không đủ đặc điểm (đặc biệt `van` mơ hồ cần đối chiếu kỹ).

Phần cần giảng viên làm rõ thêm (nếu có): ranh giới `van`/`bus` (minibus) và `car`/`truck` (pickup/xe bán tải) trong các cảnh xa/mờ.

## 3. Annotation và self-QC

- Số file nhãn train: 50 file `.txt` trong `dataset/labels/train/` (khớp basename 50 ảnh `drive_001`–`drive_050`).
- Kết quả checker và lỗi đã xử lý: `python tools/check_yolo_labels.py` → **ĐẠT cấu trúc 50/50** (đúng 5 cột, class 0–3, tọa độ chuẩn hóa [0,1], box không vượt biên). Với bộ `dataset/labels_reviewed/train` sau chỉnh sửa an toàn cũng **ĐẠT 50/50**.
- Danh sách ít nhất 10 ảnh đã rà bằng mắt: *(cần bổ sung danh sách ảnh tôi đã tự rà trực tiếp trên CVAT/ảnh gốc — ví dụ các ảnh ưu tiên drive_002,003,004,009,012,030,041,...; tôi xác nhận lại và điền chính xác).*
- Lỗi bỏ sót/box/class phát hiện khi rà và cách sửa: phát hiện FP cao (gán thừa/box nhỏ-xa), nhầm class `bus↔van↔truck`, trùng box; đã xử lý **7 trường hợp trùng lặp có căn cứ hình học cao** trên bản `dataset/labels_reviewed/train` (1 trùng cùng class `drive_047`; 6 trùng vị trí IoU≥0.956 nhưng xung đột class — đã bỏ box trùng, giữ 1 box và ghi chú **cần xác minh class trên ảnh gốc**, không dùng Gold). Log: `outputs/review_automation/applied_changes.json`.
- Commit khóa nhãn ban đầu: *(điền hash commit đã khóa/nộp trước khi nhận gold theo mốc lớp; kiểm tra `git log`/GitHub).*

## 4. Gold QC và rework

Đánh giá bằng `tools/evaluate_annotations.py` (cùng class, IoU≥0.50, one-to-one greedy như tool). **Gold chỉ dùng để đánh giá, không dùng để sao chép/chỉnh tọa độ nhãn.**

| Chỉ số | Trước sửa (baseline, `dataset/labels/train`) | Sau sửa an toàn (reviewed, 7 fix trùng lặp) |
|---|---:|---:|
| TP | 466 | 466 |
| FP | 763 | 756 |
| FN | 187 | 187 |
| Precision | 0.3792 | 0.3813 |
| Recall | 0.7136 | 0.7136 |
| F1 | 0.4952 | 0.4971 |
| mean_iou_of_matches | 0.8254 | 0.8253 |
| passed | false | false |

- Kết quả đánh giá reviewed: `outputs/review_automation/after_eval_all50.json`. File `outputs/annotation_evaluation.json` (hiện có trong repo local) phản ánh đánh giá trước đó; cần cập nhật khi có phiên bản đạt yêu cầu để nộp (`git add -f outputs/annotation_evaluation.json`).
- **Rework/chưa đạt:** `passed=false` (yêu cầu `F1 >= 0.80` và `mean_iou_of_matches >= 0.70`). Mean IoU match đã đạt ngưỡng (0.8253) nhưng **F1 chưa đạt** (0.4971), chủ yếu do **FP cao** và lỗi **class/completeness** (nhầm `van/bus/truck`, gán thừa/thiếu). Tôi không suy nguyên nhân chỉ từ FP/FN mà đã đối chiếu bằng Phase 1/gợi ý YOLO26n (chỉ bằng chứng phụ, không phải ground truth) và lập danh sách cần xem lại: `outputs/review_automation/review_queue_all.json` (496 flag), `outputs/gold_review_phase1/phase1_report.json`. 6 trường hợp trùng box xung đột class cần **xác minh class trên ảnh gốc** (không dùng Gold).
- Ảnh/lỗi đã kiểm và quyết định sửa/giữ: đã ghi trong `outputs/review_automation/applied_changes.json`; các trường hợp khó (`van` mơ hồ, box nhỏ/xa không có hỗ trợ YOLO) tôi **giữ nguyên/đánh dấu cần xem lại** theo guideline, không tự bịa nhãn.

## 5. Kết quả YOLO

- Runtime, GPU/CPU, phiên bản Ultralytics và Torch: *(chưa chạy huấn luyện trong phiên local này; môi trường local: CPU, `ultralytics 8.4.208`, `torch 2.15.0+cpu`).* **Chưa có kết quả huấn luyện Colab thực tế** — tôi không bịa số liệu.
- Các tham số thay đổi so với notebook nếu có và lý do: notebook dùng `yolo26n.pt`, `epochs=50`, `imgsz=640`, `batch=-1`, `val` theo `data.yaml` (10 ảnh eval, dùng làm validation chọn `best.pt` trong notebook — **không đưa eval vào train**). Đã sửa lỗi placeholder `REPO_URL`/assert trong notebook để sẵn sàng chạy Colab.
- Đường dẫn best checkpoint trong phiên chạy: *(chưa có)*

| Chỉ số trên 10 ảnh eval | Giá trị |
|---|---:|
| Precision | *(chưa chạy)* |
| Recall | *(chưa chạy)* |
| mAP50 | *(chưa chạy)* |
| mAP50–95 | *(chưa chạy)* |

Hướng dẫn chạy Colab: `outputs/review_automation/COLAB_RUN_GUIDE.md`. Sau khi chạy, giải nén `day2_colab_outputs.zip`, đặt `eval_predictions/` và `yolo26_runs/` vào `report/individual/assets/` (không nộp `.pt`) và cập nhật bảng/ảnh/link Colab.

Trả lời bốn câu hỏi cuối notebook (dựa trên quy tắc/hiểu bài; phần minh họa ảnh/recall lớp thấp nhất cần **kết quả Colab thực tế**):
1. Lớp nào recall thấp nhất? Ảnh nào minh họa? — *Cần điền từ kết quả eval/ảnh prediction thực tế sau khi chạy Colab.* Dự kiến `truck/bus/van` dễ recall thấp do class imbalance (ít hơn `car`) và lỗi annotation; cần minh họa bằng 1 ảnh eval cụ thể.
2. Nếu mAP50 cao nhưng mAP50–95 thấp, cần kiểm điều gì trên box? — Cần kiểm **độ bám sát/tightness và localization** của box (IoU với vật thể): box phát hiện đúng vị trí sơ bộ nhưng lệch biên/rộng-hẹp nên IoU cao (0.5) đạt nhưng IoU 0.75–0.95 kém; liên quan quy tắc “box ôm sát phần thân xe nhìn thấy”.
3. Lỗi class nào có thể liên quan guideline? Dữ kiện nào hỗ trợ nhận xét? — Nhầm `van`↔`bus`/`truck`/`car` và `truck`↔`bus` có thể liên quan guideline định nghĩa lớp (minibus/bus, van/minivan, pickup/car vs truck). Dữ kiện hỗ trợ: Phase 1 cross-class hints (IoU cao khác class), 6 duplicate xung đột class, 496 flag review (`review_queue_all.json`), confusion giữa `bus`/`van` trong phân tích. Cần xác minh trên ảnh gốc.
4. Vì sao không dùng 10 ảnh eval để train? Vai trò validation của chúng trong notebook là gì? — 10 ảnh eval **không dùng để train** để giữ đánh giá công bằng/tránh rò rỉ dữ liệu. Trong notebook, tập eval được dùng làm **validation** (`val` trong `data.yaml`) để theo dõi và chọn `best.pt`, nên **không phải test set độc lập**.

## 6. Khó khăn và cách xử lý

| Bước/thời điểm | Lỗi quan sát | Cách đã thử | Kết quả/còn thiếu |
|---|---|---|---|
| Sau gold QC | F1 thấp (0.4952), FP cao, nhầm class `van/bus/truck` | Chạy `check_yolo_labels.py`/`evaluate_annotations.py`, Phase 1 matching, YOLO26n làm bằng chứng phụ (không ground truth), tạo review queue 50 ảnh, sửa 7 trùng lặp hình học an toàn, tạo tool Streamlit review | F1=0.4971, **passed=false**; còn 496 flag + 6 xung đột class cần xác minh ảnh gốc |
| Notebook Colab | Placeholder `REPO_URL`/assert dễ gây chạy sai | Sửa assert kiểm tra placeholder/rỗng/không phải GitHub URL | Notebook sẵn sàng; **chưa chạy Colab**, chưa có metrics/ảnh eval |
| Môi trường local | CPU, không có CUDA; giới hạn phán đoán `van` tự động | Ưu tiên Colab GPU cho train; không tự bịa nhãn/số liệu | Cần chạy Colab và rà nhãn thủ công có căn cứ |

## 7. Tự đánh giá theo rubric

Đối chiếu [RUBRIC.md](../../RUBRIC.md) (thực tế hiện tại, không tự nhận hoàn thành bước chưa chạy):
- Hoàn thành/định dạng YOLO (10đ): **có cơ sở** — 50/50, 5 cột, normalized, class 0–3.
- Đầy đủ vật thể (20đ): **còn thiếu/đang rà** — recall 0.7136, FN 187 theo gold; cần rà bổ sung/sửa class.
- Độ bám sát/hình học (20đ): **tương đối tốt trên match** (mean IoU 0.8253) nhưng vẫn có box lệch/cần siết.
- Phân lớp nhất quán (15đ): **điểm yếu** — nhầm `van/bus/truck/car` (cross-class hints, xung đột duplicate class).
- Mini guideline + 3 ca mơ hồ (10đ): **có** (`GUIDELINE_MINI.md`).
- Gold-set QC (15đ): **chưa đạt ngưỡng** (`F1=0.4971 < 0.80`, `passed=false`); đã có đánh giá/rework log nhưng cần đạt điều kiện.
- Pipeline YOLO & phân tích (10đ): **chưa hoàn thành** — chưa chạy Colab, chưa có metrics/ảnh eval thực tế.

## 8. Công cụ AI và tài liệu tham khảo

- Công cụ AI đã dùng: **OpenCode (AI coding assistant)** hỗ trợ đọc/kiểm tra cấu trúc, chạy script đánh giá/checker, phân tích Phase 1 (matching không dùng Gold để sửa), tạo queue bằng chứng phụ từ YOLO26n, tạo/cải tiến công cụ review Streamlit, chỉnh sửa notebook/config rõ ràng, soạn nháp báo cáo từ kết quả thực tế. **Phần tôi tự kiểm tra/chịu trách nhiệm:** nội dung guideline/quyết định nhãn, rà ảnh gốc, xác nhận class `van/bus/truck` mơ hồ, chọn phiên bản nhãn nộp, chạy Colab và kiểm tra số liệu thực tế trước khi nộp.
- Tài liệu tham khảo: `GUIDE.md`, `RULES.md`, `CHECKPOINTS.md`, `RUBRIC.md`, `SUBMISSION.md`, tài liệu Ultralytics/YOLO. Không ghi mật khẩu/token, không đính kèm `gold_train_labels/`.
