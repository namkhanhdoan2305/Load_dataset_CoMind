# Chạy CoMind từ GitHub trên Colab

Mã được clone vào máy ảo Colab. Dữ liệu đi từ ETH tới Colab rồi tới Google Drive hoặc Hugging Face. Máy cá nhân chỉ mở giao diện notebook.

## Chuẩn bị mã

Chọn runtime CPU trong Colab. Chạy các cell sau lần đầu trong mỗi runtime.

```python
!git clone https://github.com/namkhanhdoan2305/Load_dataset_CoMind.git /content/Load_dataset_CoMind
%cd /content/Load_dataset_CoMind
```

Nếu thư mục đã tồn tại, chạy `%cd /content/Load_dataset_CoMind` rồi `!git pull --ff-only`.

## Xem dung lượng

Đọc manifest của một phiên train trước, chưa tải video và chưa cần đăng nhập HF.

```python
!python comind_to_hf.py --split train --limit 1 --dry-run
```

Để tính toàn bộ 80 phiên, dùng `--split all --dry-run`. Tổng dung lượng hiển thị gồm annotations chung một lần. Phiên thiếu modality hoặc manifest lỗi sẽ khiến script dừng để bạn kiểm tra. `--allow-missing-modalities` cho phép tiếp tục khi đã chủ động chấp nhận modality thiếu.

## Lưu Google Drive

Với 400 GB còn trống, có thể lưu bộ sáu nhóm được chọn, ước lượng khoảng 300 GB cho toàn bộ 80 phiên. Dung lượng chính xác lấy từ dry-run. Dung lượng tài khoản Drive và ổ đĩa tạm Colab là hai hạn mức riêng.

Mount Drive bằng cell Python này.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Tải thử một phiên trực tiếp vào Drive.

```python
!python comind_to_hf.py --split train --limit 1 --drive-dir /content/drive/MyDrive/CoMind
```

Tải toàn bộ bộ chọn lọc sau khi kiểm tra dung lượng còn trống.

```python
!python comind_to_hf.py --split all --drive-dir /content/drive/MyDrive/CoMind
```

Chạy lại đúng lệnh để tiếp tục file `.part`; file hoàn tất được kiểm tra lại hash và bỏ qua bước tải nếu khớp. File `.part` nằm trên Drive nên vẫn còn sau khi runtime reset. Drive có thể gặp giới hạn I/O hoặc dung lượng, khi đó script báo lỗi và giữ file tải dở. Kiểm tra mục Storage trên Drive để biết dung lượng tài khoản thực tế, `disk_usage` của mount Colab có thể phản ánh filesystem đệm.

## Lưu Hugging Face

HF hiện công bố 100 GB private storage cho tài khoản miễn phí. Public storage áp dụng best-effort theo chính sách HF. Dataset CoMind được tác giả công bố theo CC BY 4.0 và cần ghi nguồn khi chia sẻ. Script tạo dataset card có nguồn và giấy phép nếu repo chưa có card.

Cài thư viện chỉ khi dùng HF.

```python
%pip -q install -r requirements.txt
```

Vào https://huggingface.co/settings/tokens tạo token có quyền ghi dataset repo của bạn. Đăng nhập bằng ô nhập ẩn ở cell riêng, token không nằm trong mã repo.

```python
from getpass import getpass
from huggingface_hub import login
login(token=getpass('HF token có quyền ghi: '), add_to_git_credential=False)
```

Hoặc lưu token trong Colab Secrets dưới tên `HF_TOKEN`, bật quyền truy cập cho notebook, rồi dùng cell này thay cho cell đăng nhập ở trên.

```python
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get('HF_TOKEN'), add_to_git_credential=False)
```

Thay `YOUR_USERNAME` bằng namespace HF của bạn. Lệnh mặc định tạo repo private và chạy thử một phiên.

```python
!python comind_to_hf.py --split train --limit 1 --repo-id YOUR_USERNAME/comind-belief-subset
```

Nếu có đủ hạn mức private, lấy toàn bộ bằng `--split all` và bỏ `--limit`. Nếu muốn repo public, thêm `--public`; giá trị này phải khớp visibility khi repo đã tồn tại. Một file được giữ tạm trên Colab, kiểm tra hash nguồn, upload, đối chiếu hash trên HF ở commit vừa tạo, rồi xóa bản tạm. Các file đã khớp trên HF được bỏ qua khi chạy lại. Runtime reset có thể khiến file đang dở phải tải lại, các file đã commit vẫn nằm trên HF.

## Chọn phiên và calibration

`--split train`, `--split test`, `--split all` giữ đúng các danh sách chính thức có sẵn trong bộ tải. `--limit N` chọn N UUID đầu tiên theo thứ tự danh sách, phù hợp kiểm tra luồng tải. Khi làm thí nghiệm cần chủ động chọn phiên đa dạng và tách train/validation theo phiên, người và bếp khi metadata cho phép. Giữ test chính thức cho đánh giá cuối.

Đặt các UUID cần dùng vào `ids.txt`, mỗi dòng một UUID, rồi dùng `--ids-file ids.txt`. Ví dụ mã trên Colab tạo danh sách 10 phiên train và 5 phiên validation riêng biệt để kiểm tra pipeline.

```python
from pathlib import Path
import comind_download as source
Path('train_pilot.txt').write_text('\n'.join(source.TRAIN_IDS[:10]) + '\n')
Path('val_pilot.txt').write_text('\n'.join(source.TRAIN_IDS[10:15]) + '\n')
```

```python
!python comind_to_hf.py --ids-file train_pilot.txt --drive-dir /content/drive/MyDrive/CoMind
!python comind_to_hf.py --ids-file val_pilot.txt --drive-dir /content/drive/MyDrive/CoMind
```

Các số lượng và UUID đầu danh sách chỉ là ví dụ chạy pilot. Dùng learning curve trên các tập train lồng nhau, giữ cố định validation, để quyết định cần thêm bao nhiêu phiên. Số sự kiện đổi vị trí, bỏ lỡ thông tin, nhận thông tin một phần và nhận lại thông tin quan trọng cho bài toán belief hơn số frame. Cần nhãn belief được kiểm chứng độc lập để chấm mô hình.

Thêm `--leader-calibration` để lấy `mps_leader_*/slam/online_calibration.jsonl`. Gaze MPS ở hệ tọa độ CPF; chiếu lên ảnh và đồng bộ với MP4 cần calibration và xử lý timestamp phù hợp. Liên kết hai view trong không gian 3D có thể cần thêm trajectory/pose, chưa nằm trong bộ sáu nhóm này.

## Tổ chức dữ liệu

```text
CoMind/
  split/train.txt
  split/test.txt
  annotations/
    source-healthcheck.json
    selection.json
    ...các nhãn chung chính thức...
  recordings/<UUID>/
    source-healthcheck.json
    selection.json
    mp4s/leader_trimmed_sync.mp4
    mp4s/helper_trimmed_sync.mp4
    mps_leader_trimmed_vrs/eye_gaze/...
    mps_leader_trimmed_vrs/hand_tracking/...
    transcripts/...
```

`source-healthcheck.json` giữ manifest đầy đủ từ tác giả, có thể liệt kê cả file ngoài bộ chọn lọc. `selection.json` liệt kê file được chọn. Danh sách split giữ nguyên bản chính thức dù chỉ tải một phần các phiên. File payload giữ nguyên byte và audio hiện có trong MP4.

Leader view cung cấp quan sát của người, helper view là quan sát một phần của robot giả lập. Nhãn chung gồm joint attention, socially conditioned object interaction anticipation và handover. Dữ liệu chưa cung cấp sẵn nhãn belief dày theo thời gian hoặc toàn bộ tiến trình công thức.

Khi train, sao chép từng batch phiên hoặc đặc trưng cần dùng từ Drive sang `/content`, chạy tại ổ tạm rồi lưu kết quả về Drive. Trích scene graph và đặc trưng một lần để tái sử dụng qua các lần train encoder. Đọc nhiều frame MP4 trực tiếp từ Drive qua từng epoch có thể làm I/O chậm.

## Nguồn

- CoMind và giấy phép CC BY 4.0 https://comind.ethz.ch/
- HF storage limits https://huggingface.co/docs/hub/storage-limits
- HF upload API https://huggingface.co/docs/huggingface_hub/guides/upload
- Colab runtime và Drive FAQ https://research.google.com/colaboratory/faq.html
