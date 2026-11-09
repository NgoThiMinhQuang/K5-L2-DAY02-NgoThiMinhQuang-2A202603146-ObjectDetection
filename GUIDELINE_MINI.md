# Mini guideline — bài làm cá nhân

> Điền nội dung này trước khi bắt đầu gán nhãn. Giữ lại các quyết định sau khi self-QC.

# Mini Guideline – Object Detection ADAS

## 1. Định nghĩa lớp

| Lớp | Định nghĩa của bạn | Ví dụ / Điểm phân biệt |
|---|---|---|
| `car` | Xe ô tô con, sedan, SUV, taxi và xe bán tải. | Xe bán tải thông thường thuộc car, không phải truck. |
| `truck` | Xe tải chở hàng, xe đầu kéo và xe tải có thùng chở hàng. | Phân biệt với car dựa trên thiết kế xe tải chuyên chở hàng hóa. |
| `bus` | Xe buýt và minibus được thiết kế để vận chuyển hành khách. | Phân biệt với van dựa trên kiểu dáng và đặc điểm xe chở khách. |
| `van` | Xe van, minivan dùng để chở hàng hóa hoặc hành khách. | Không nhầm với xe buýt lớn hoặc ô tô con thông thường. |

## 2. Quy tắc chung

- **Box bám sát vật thể:** Vẽ bounding box hình chữ nhật ôm sát phần thân xe nhìn thấy. Không lấy thêm quá nhiều nền đường, cây cối hoặc các xe bên cạnh.
- **Vật thể bị che khuất:** Vẫn gán nhãn nếu phần xe nhìn thấy đủ để nhận diện lớp. Chỉ vẽ box bao quanh phần nhìn thấy, không suy đoán phần bị che.
- **Vật thể bị cắt bởi rìa ảnh:** Vẽ bounding box đến đúng rìa ảnh, không mở rộng ra ngoài phạm vi ảnh.
- **Vật thể rất nhỏ/mơ hồ:** Nếu không thể xác định chắc chắn loại xe, không tự đoán nhãn. Ghi lại tên ảnh và tình huống để hỏi giảng viên.
- **Nhiều xe trong cùng ảnh:** Mỗi xe hợp lệ phải có một bounding box riêng, không gộp nhiều xe vào một box.
- **Xe đang đỗ:** Vẫn gán nhãn bình thường theo loại xe.
- **Hai xe chồng lên nhau:** Vẽ hai bounding box riêng nếu có thể nhận diện từng xe.
- **Vật thể ngoài phạm vi:** Không gán nhãn xe máy, xe đạp, người đi bộ, đèn giao thông, biển báo hoặc hình xe trên bảng quảng cáo.



## 3. Ghi lại quyết định cho ít nhất ba ca mơ hồ

| Ảnh | Tình huống | Quyết định cuối | Lý do theo schema |
|---|---|---|---|
| `drive_002` | Các phương tiện di chuyển gần nhau, có thể bị che khuất một phần. | Gán nhãn từng xe nếu phần nhìn thấy đủ để xác định loại xe. Box chỉ bao quanh phần xe nhìn thấy. | Theo guideline, xe bị che khuất một phần vẫn phải được gán nhãn nếu có thể nhận diện lớp. |
| `drive_003` | Nhiều phương tiện xuất hiện gần nhau tại khu vực giao lộ, dễ vẽ box chồng lấn hoặc gộp xe. | Vẽ bounding box riêng cho từng xe thuộc bốn lớp, không gộp nhiều phương tiện thành một box. | Mỗi vật thể hợp lệ phải có một bounding box riêng và phải được phân loại đúng theo schema. |
| `drive_004` | Cảnh giao thông ban đêm, ánh sáng yếu và xe ở xa khiến việc phân biệt loại xe khó khăn. | Chỉ gán nhãn khi nhận diện được loại xe. Không tự đoán đối với phương tiện quá mờ hoặc không đủ đặc điểm. | Schema yêu cầu phân loại chính xác car, truck, bus, van; không gán nhãn theo suy đoán. |

