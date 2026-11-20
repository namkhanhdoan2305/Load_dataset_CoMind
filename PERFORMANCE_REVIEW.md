# Kiểm tra tốc độ lượt Colab đầu — 20/11/2026

## Thời lượng recording đã xác minh

Recording: `8ee6c694-38be-4a90-ad5a-75ef4449e972`.

Đã đọc trực tiếp box `moov` của hai MP4 nguồn bằng HTTP Range, chỉ lấy metadata khoảng 1,6 MB/video. Không suy thời lượng từ dung lượng file hoặc transcript.

| Thuộc tính | Leader | Helper |
|---|---:|---:|
| Thời lượng video track | 1703,6 s | 1703,6 s |
| Đổi sang phút | 28 phút 23,6 giây | 28 phút 23,6 giây |
| Số frame | 51.108 | 51.108 |
| FPS | 30 | 30 |
| Thời lượng movie/container | 1703,654 s | 1703,654 s |

Nguồn là `https://comind.ethz.ch/dataset/<recording_ID>/mp4s/leader_trimmed_sync.mp4` và file helper tương ứng. Lưới graph 5 FPS tạo khoảng 8.518 graph/view, tổng khoảng 17.036 graph/session. Chunk 4 giây tương ứng 426 chunk/view. Recording này dài gần nửa giờ; chưa đo phân bố thời lượng của toàn bộ dataset.

## Số đo do người dùng cung cấp

```text
leader: chunk 1/426, 21 graphs, 133 total nodes, 698 total edges,
last t=4.00s, chunk took 258.7s
```

Đây là thời gian của chunk đầu, khoảng 4 phút 19 giây cho 4 giây video, chậm hơn thời gian thực khoảng 64,7 lần. Nếu các chunk còn lại giữ tốc độ này, ngoại suy là 30,6 giờ/view, khoảng 61,2 giờ/session hai view. Chưa dùng ngoại suy này làm benchmark ổn định: cần xem các chunk sau và đo từng stage. Thời gian chunk bao gồm decode, ba model, chuyển CPU/GPU và ghi/check hash output.

## Những chi phí xác định được từ mã

Trong một chunk đầy đủ đầu tiên:

- Detector xử lý 21 keyframe. Các chunk sau giữ frame biên để nối tracking, nên frame biên có thể được xử lý lại rồi loại bản graph trùng khi commit.
- SAM2 xử lý 41 tracking frame. Code đưa box của từng detection vào tracker ở các keyframe, sau đó gọi propagation theo từng frame.
- Mã upstream SAM2 `add_new_points_or_box` thực hiện suy luận mask và tạo output mask theo kích thước video cho mỗi lần thêm box. Pipeline hiện chỉ cần bbox cuối, nhưng vẫn trả chi phí segmentation này.
- Mã upstream `propagate_in_video` có bước preflight và xử lý từng object. Pipeline khởi tạo lại generator ở mỗi frame; mỗi keyframe có thể tạo nhiều prompt theo số object. Không thể đồng nhất một frame video với một forward pass duy nhất.
- `offload_state_to_cpu=True` giữ state trên CPU và chuyển lại GPU khi tính. Đây là lựa chọn giảm VRAM với chi phí tốc độ.
- Ba model được chuyển lên GPU lần lượt và chuyển về CPU sau mỗi stage/chunk. Mã hiện chưa bật autocast mixed precision.

Các chi phí trên có thật trong code; chưa có số đo chứng minh stage nào chiếm phần lớn 258,7 giây. Không kết luận SAM2 là thủ phạm duy nhất hoặc đổi GPU sẽ giải quyết được.

Mã upstream đã đọc đúng commit SAM2 khóa trong requirements:
[sam2_video_predictor.py](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py).

## Bổ sung để đo tiếp

Commit `9dcdfaf` chỉ bổ sung log, không đổi graph FPS, model, trọng số, tracking hay timestamp. Có thể tiếp tục checkpoint cũ.

- In tên GPU thực tế.
- In frame count, FPS, thời lượng video và chunk tiếp tục.
- In thời gian decode.
- In START, tiến độ và DONE riêng cho detector, SAM2 và relations. Chi phí chuyển model/offload được tính vào stage tương ứng.

Chưa có lượt inference T4 do assistant trực tiếp chạy. 47 kiểm thử dữ liệu/video/perception giả lập đã qua sau khi bổ sung log; chúng không đo throughput GPU.

## Quyết định tiếp theo

Đo một lượt ngắn trước khi mở rộng. Nếu đổi model, precision hay thuật toán tracking, dùng target/config mới để giữ riêng kết quả và không ghép graph từ hai cấu hình. Có thể thử REACT++ cùng tracker bbox nhẹ, nhưng cần kiểm tra tập nhãn và chất lượng trên cùng đoạn CoMind; chưa thay pipeline hiện hành.

Chi phí đang thấy là suy luận perception để tạo dữ liệu graph. Chi phí huấn luyện encoder belief trên graph đã lưu cần được đo riêng; thời gian này chưa chứng minh cần A100/H100 cho bước train.
