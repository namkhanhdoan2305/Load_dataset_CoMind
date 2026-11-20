# Load_dataset_CoMind

Tải bộ CoMind chọn lọc trên Colab CPU, lưu vào Google Drive hoặc chuyển tới Hugging Face. Máy cá nhân chỉ mở notebook.

## Thử SGG trên hai recording

Đọc [SGG_COLAB_GUIDE.md](SGG_COLAB_GUIDE.md) để cài model trên Colab T4 và chạy `python sgg_pipeline.py --sessions 2 --keep-videos`. Graph xuất 5 FPS/view, tracking 10 FPS, lưu Drive và tiếp tục được theo checkpoint. Hai graph liên kết theo mốc MP4. Gaze/hand vẫn cần ánh xạ clock cảm biến sang MP4, trạng thái được ghi rõ trong `sync-status.json`. Hướng dẫn có cấu trúc đầu ra và cell xem graph sau khi chạy.

## Tải trước 30 phiên vào Google Drive

Mở Colab runtime CPU, chạy lần lượt các cell. Phần tải Drive dùng thư viện chuẩn Python có sẵn.

```python
!git clone https://github.com/namkhanhdoan2305/Load_dataset_CoMind.git /content/Load_dataset_CoMind
%cd /content/Load_dataset_CoMind
```

Nếu đã clone trong runtime này, dùng `%cd /content/Load_dataset_CoMind` rồi `!git pull --ff-only`.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Tính dung lượng đúng 30 phiên đã chọn bằng manifest nhỏ trước khi tải payload.

```python
!python load_comind_drive.py --sessions 30 --dry-run
```

```python
!python load_comind_drive.py --sessions 30
```

File được lưu vào `/content/drive/MyDrive/CoMind`. Mặc định chọn 30 phiên từ split train chính thức, xáo trộn bằng seed 42. `subsets/sessions-30-seed-42.json` ghi rõ danh sách UUID. Cần tách train/validation trước khi huấn luyện.

Dry-run ngày 20/11/2026 của bộ mặc định báo **99,407 GB** (92,580 GiB), gồm annotations chung một lần. Có **1 phiên thiếu leader hand tracking**, UUID `6473136b-6037-46be-a8b0-fb86b5997376`. Giá trị có thể thay đổi khi tác giả cập nhật dữ liệu; dry-run đọc manifest hiện tại.

Mỗi file được đối chiếu hash từ manifest gốc. Chạy lại cùng lệnh để tiếp tục file `.part`, kiểm tra file đã có và tải các file còn thiếu. `download-status.json` lưu tiến độ đã xác minh. Ngân sách mặc định 5 giờ được kiểm tra trước mỗi file, script hoàn thành file đang chạy rồi dừng. Thời gian một lượt có thể vượt 5 giờ bởi thời gian tải file cuối và bước kiểm tra hash. Thời gian hoàn thành cả 30 phiên phụ thuộc mạng và Drive I/O.

Mỗi recording được chia thành sáu nhóm như cây thư mục dưới đây. File đã tải theo cấu trúc cũ được chuyển sang nhóm tương ứng bằng cách sao chép, kiểm tra hash rồi mới bỏ bản ở đường dẫn cũ. File `.part` được giữ tiến độ khi chuyển. Cần có dung lượng trống cho một bản sao tạm của file đang chuyển.

```text
CoMind/
  recordings/<recording_ID>/
    leader_video/leader_trimmed_sync.mp4
    helper_video/helper_trimmed_sync.mp4
    leader_gaze/general_eye_gaze.csv
    leader_hands/hand_tracking_results.csv
    transcripts/
      leader_trimmed_sync_transcript.json
      helper_trimmed_sync_transcript.json
    annotations/
      joint_attention.json
      interaction_anticipation.json
      handover.json
  annotations/                # Ba JSON gốc chung, tải một lần
  metadata/<recording_ID>/     # Manifest gốc, file được chọn và ánh xạ đường dẫn
  subsets/
  split/
  download-status.json
```

Annotation theo phiên giữ nguyên event ID, nhãn và giá trị null từ tác giả. Event ID được bảo toàn theo nguồn, cần đọc schema gốc để diễn giải thời gian.

Một số phiên thiếu hand tracking từ nguồn. Mặc định tiếp tục với các file có sẵn và ghi `missing_modalities` trong manifest chọn lọc. Script dừng nếu một trong hai video bắt buộc bị thiếu. Thêm `--strict-modalities` để dừng với mọi nhóm bị thiếu.

Mở rộng lên 50 phiên bằng cùng seed và thư mục. Danh sách mới giữ toàn bộ 30 phiên cũ.

```python
!python load_comind_drive.py --sessions 50
```

Có thể đổi đích bằng `--target /content/drive/MyDrive/CoMind`, ngân sách bằng `--max-hours 8`, hoặc thêm calibration bằng `--leader-calibration`. Khi bật thêm calibration, script giữ các file đã có và tải bổ sung, metadata cập nhật theo bộ chọn lọc mới.

## Thành phần dữ liệu

Lấy hai video ego MP4 đồng bộ, gaze và hand tracking của leader, transcript và nhãn chung chính thức. Tùy chọn lấy thêm leader calibration. Chạy lại để tiếp tục tải và đối chiếu hash, HF upload thành công được xác minh trước khi xóa file tạm.

| Nhóm | Đường dẫn |
| --- | --- |
| Leader video | `mp4s/leader_trimmed_sync.mp4` |
| Helper video | `mp4s/helper_trimmed_sync.mp4` |
| Leader gaze | `mps_leader_trimmed_vrs/eye_gaze/general_eye_gaze.csv` |
| Leader hands | `mps_leader_trimmed_vrs/hand_tracking/hand_tracking_results.csv` |
| Transcript | Hai file `leader_trimmed_sync_transcript.json` và `helper_trimmed_sync_transcript.json` |
| Nhãn chung | Ba JSON consolidated chính thức, tải một lần, tách thêm bản theo recording |

MP4 giữ nguyên audio và dữ liệu nguồn. Nhãn chung gồm joint attention, interaction anticipation và handover. Nhãn belief và tiến trình công thức cần được xây dựng thêm. Hai video là dữ liệu gốc để trích graph trong bước chuẩn bị tiếp theo.

Đọc [hướng dẫn Colab, Google Drive và đăng nhập HF](COLAB_HF_GUIDE.md).

```bash
python comind_to_hf.py --split train --limit 1 --dry-run
python comind_to_hf.py --split train --limit 1 --drive-dir /content/drive/MyDrive/CoMind
```

Bản gốc CoMind do ETH Zurich và cộng sự công bố, giấy phép CC BY 4.0. Ghi nguồn khi chia sẻ lại dữ liệu. [Trang tác giả](https://comind.ethz.ch/) và [paper](https://arxiv.org/abs/2607.06691).

Kiểm tra luồng xử lý với dữ liệu tổng hợp bằng `python -m unittest discover -s tests -v`.

## Kiểm tra file thừa của bản tải trước

Bộ lọc cũ `mps_leader_*` chọn cả gapfill và trimmed. Bộ lọc mới dùng tên file chính xác, lấy dữ liệu MPS leader trimmed. Các bản `summary.json`, phụ đề ASS, transcript phụ và MPS gapfill được loại khỏi danh sách tải mới.

Liệt kê đường dẫn được chọn trước khi tải bằng lệnh sau.

```python
!python load_comind_drive.py --sessions 30 --list-files
```

Kiểm kê file thừa đã có trong thư mục Drive bằng lệnh sau. Lệnh chỉ đọc và liệt kê, giữ nguyên dữ liệu.

```python
!python load_comind_drive.py --sessions 30 --audit-existing
```

Các file thừa do bản trước đã tải vẫn tồn tại tới khi bạn chủ động xóa. Manifest `source-healthcheck.json` chứa danh mục đầy đủ từ máy chủ, `selection.json` ghi đúng danh sách file được chọn và đích lưu.
