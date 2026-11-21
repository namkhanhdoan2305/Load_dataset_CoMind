# Load_dataset_CoMind

Pipeline hiện hành: **tải từng recording → REACT++ YOLO12m ONNX + tracker bbox → ghép graph/gaze/hand/lời nói → lưu Google Drive**. Mặc định chạy một recording, graph 5 FPS/view. Video chỉ lưu tạm trên Colab; giữ để review bằng `--keep-videos`.

[Hướng dẫn Colab](SGG_COLAB_GUIDE.md) · [Review sáu nhóm dữ liệu](DATA_REVIEW.md) · [Quy ước clock gaze/hand](SENSOR_SYNC.md)

## Chạy thử trên Colab T4

Nếu đã clone repo, dừng cell chạy cũ rồi cập nhật:

```python
%cd /content/Load_dataset_CoMind
!git pull --ff-only
```

Runtime mới thì clone và mount Drive trước:

```python
!git clone https://github.com/namkhanhdoan2305/Load_dataset_CoMind.git /content/Load_dataset_CoMind
%cd /content/Load_dataset_CoMind
from google.colab import drive
drive.mount('/content/drive')
```

Repo chỉ dùng một file dependencies. Không cài đồng thời hai package ONNX Runtime CPU/GPU:

```python
!python -m pip uninstall -y onnxruntime onnxruntime-gpu
!python -m pip install -r requirements.txt
```

```python
!python -u sgg_pipeline.py --sessions 1 --keep-videos
```

Kết quả: `/content/drive/MyDrive/CoMind_REACTpp/recordings/<ID>/`. Video tạm: `/content/comind-sgg-work/<ID>/mp4s/`. Video đã tải đủ được kiểm tra hash rồi tái sử dụng. Khi bỏ `--keep-videos`, chỉ video tạm được xóa sau khi cả hai view và export hoàn tất; video có sẵn trên Drive vẫn giữ nguyên.

Recording đầu seed 42 là `8ee6c694-38be-4a90-ad5a-75ef4449e972`, mỗi view dài 28 phút 23,6 giây. Thời gian chạy T4 cần đo từ log thực tế.

## Kiểm tra và mở rộng

Chỉ đọc manifest, in đúng danh sách file và dung lượng, không tải payload/model:

```python
!python sgg_pipeline.py --sessions 1 --dry-run
```

Sau khi hai view complete, kiểm tra output. Nếu graph được tạo trước bản ghép sensor/word/annotation mới, chạy `sensor_sync.py` trước; các lượt mới tự ghép khi hoàn tất recording:

```python
!python sensor_sync.py /content/drive/MyDrive/CoMind_REACTpp/recordings/RECORDING_ID
!python review_recording.py /content/drive/MyDrive/CoMind_REACTpp/recordings/RECORDING_ID
```

Mở rộng cùng seed và target, tiếp tục các recording đã commit:

```python
!python -u sgg_pipeline.py --sessions 30
```

`--sessions 50` giữ 30 phiên đầu khi seed không đổi. `--sessions 55` chạy hết official train. Ngân sách mặc định 5 giờ tính từ lúc bắt đầu đọc manifest và được kiểm tra giữa file/chunk. Chạy lại cùng seed và đích lưu để dùng checkpoint graph đã có.

`sgg-status.json` ghi thứ tự phiên, `completed` (đã xuất graph và step), `failed` (lỗi theo phiên và theo bước) và `needs_review` (đã xuất nhưng còn thiếu dữ liệu hoặc đồng bộ). Nếu mốc MP4/VRS chưa xác minh được, phiên vẫn xuất graph và lời nói, giữ CSV gốc nếu có và để gaze/hand ở `null` trong step. Hai cờ đồng bộ phản ánh phần này trong `sync-status.json`. Lỗi riêng của một recording được báo rõ và lô tiếp tục; lần chạy sau thử lại phiên đó. `complete_with_issues` nghĩa là đã đi hết danh sách nhưng vẫn có phiên cần xử lý tiếp.

Trên Kaggle dùng `--target /kaggle/working/CoMind_REACTpp --work-dir /tmp/comind-sgg-work` và chừa thời gian lưu Output, ví dụ `--max-hours 11.5`. Để tiếp tục ở một version mới, gắn Output đã lưu qua Add Input và chép nguyên cây `CoMind_REACTpp` vào `/kaggle/working/` trước khi gọi pipeline.

## Mã nguồn đang dùng

| File | Vai trò |
|---|---|
| `sgg_pipeline.py` | Lệnh chạy chính, tải từng recording, resume và lưu Drive |
| `reactpp_models.py` | Checkpoint ONNX được khóa và tracker bbox |
| `comind_source.py` | Manifest, official split, HTTP resume và hash |
| `comind_data.py` | Chọn đúng file, chọn phiên, kiểm tra tải, tách annotation |
| `time_contract.py` | Clock ID, giây và microsecond |
| `sensor_sync.py` | Kiểm chứng origin MP4/VRS và ghép gaze/hand |
| `annotation_timing.py` | Chuẩn hóa timing nhãn benchmark, giữ ngoài input |
| `belief_data.py` | Xuất input tuần tự, word/segment và loader nhãn belief |
| `review_recording.py` | Audit file/hash/timestamp sau xử lý, chỉ đọc |

Bản dọn bỏ model DINO/SAM2/RelateAnything cũ, upload HF, lệnh tải hàng loạt raw video và các báo cáo/hướng dẫn hết dùng. Hugging Face chỉ dùng để tải checkpoint REACT++ công khai.

## Dữ liệu và giới hạn

Sáu nhóm gồm leader, helper, leader gaze, leader hand, hai transcript và ba annotation chung. Chọn đường dẫn chính xác, không lấy merged video, gapfill, phụ đề ASS, GoPro, scan/SLAM hoặc nguyên VRS. Đồng bộ dùng metadata MP4/VRS có kiểm chứng và hash nguồn; mẫu thiếu/invalid có mask riêng. Xem [DATA_REVIEW.md](DATA_REVIEW.md) về word stream, tail cuối phiên và các cờ đồng bộ.

Những phần chưa sinh tự động: task-progress/nhãn belief, matching cùng vật giữa hai view và phép biến đổi gaze/hand sang world coordinates. Độ chính xác SGG và đồng bộ audio/video vật lý cần đánh giá riêng.

Nguồn CoMind: CVG Group, ETH Zurich và cộng sự, [trang dataset](https://comind.ethz.ch/), [paper](https://arxiv.org/abs/2607.06691), CC BY 4.0. Các hàm nguồn và official split được giữ từ downloader CoMind version 20260626-01.

Kiểm thử: `python -m unittest discover -s tests -v`. Các test ảnh cần dependencies NumPy/OpenCV; test CPU sử dụng checkpoint thật đã được chạy riêng, chưa có benchmark GPU T4 của assistant.
