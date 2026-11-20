# Chạy thử SGG trên hai recording

Pipeline dùng Grounding DINO Tiny, SAM2.1 Small và RelateAnything ConvNeXt-Base pretrained. Graph xuất ở **5 FPS cho mỗi view**, tracking chạy **10 FPS**. Video nguồn 30 FPS tương ứng graph tại frame 0, 6, 12, 18... Ảnh được giảm cạnh dài về tối đa 960 pixel, tọa độ box đi kèm kích thước ảnh đã giảm và ảnh nguồn để chuyển đổi lại.

Đây là bản thử kỹ thuật. Luồng lưu/tiếp tục và liên kết thời gian đã được kiểm tra bằng video và graph giả lập. Cần chạy trên Colab T4 để kiểm tra model inference thực tế, tốc độ, bộ nhớ và chất lượng graph trên CoMind.

**Trạng thái đồng bộ đầy đủ còn thiếu.** Hai graph được kiểm tra và liên kết bằng giây tương đối của hai MP4 `trimmed_sync`. Transcript giữ thời gian start/end từ nguồn và từ đã kết thúc được ghép vào timeline. Gaze/hand giữ `tracking_timestamp_us` gốc. Chưa có ánh xạ được kiểm chứng từ đồng hồ cảm biến sang MP4 nên các giá trị này được ghi `unresolved` trong `sync-status.json`. Mã không tự coi timestamp gaze đầu tiên là giây 0. Annotation giữ event ID gốc và chưa chuyển ID này thành giây/frame. Cần giải quyết ánh xạ thời gian và kiểm chứng gaze/hand trước khi dùng làm cue đồng bộ cho belief.

## Các cell Colab

Chọn runtime **GPU T4**. Nếu đã clone repo, chạy cell cập nhật này.

```python
%cd /content/Load_dataset_CoMind
!git pull --ff-only
```

Với runtime mới, clone repo trước.

```python
!git clone https://github.com/namkhanhdoan2305/Load_dataset_CoMind.git /content/Load_dataset_CoMind
%cd /content/Load_dataset_CoMind
```

Mount đúng tài khoản Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Cài dependencies model. File requirements khóa commit mã nguồn của RelateAnything và SAM2. Upstream RelateAnything yêu cầu Python 3.12 trở lên, PyTorch và Transformers theo các phiên bản do tác giả khai báo. Lệnh cài có thể cập nhật các thư viện này trong runtime. Lệnh chạy inference bên dưới dùng tiến trình Python mới.

```python
%pip install -r requirements-sgg.txt
```

Chạy thử đúng hai recording đầu của cùng danh sách seed 42 dùng khi tải 30 recording.

```python
!python sgg_pipeline.py --sessions 2 --keep-videos
```

`--keep-videos` giữ video tạm trên Colab cho lần thử để có thể xem lại. Dùng `--target` nếu dữ liệu trước đó được tải vào thư mục khác.

```python
!python sgg_pipeline.py --sessions 2 --keep-videos \
    --target /content/drive/MyDrive/HRC_NCKH/CoMind
```

Mặc định ngân sách xử lý là 5 giờ, kiểm tra trước mỗi file và mỗi chunk. Một file đang tải hoặc chunk đang xử lý có thể làm lượt chạy vượt mốc này. Khi hết ngân sách, chạy lại đúng lệnh để tiếp tục. Dữ liệu đã commit được kiểm tra hash rồi bỏ qua. Video tạm có thể cần tải lại khi Colab reset, các graph đã commit trên Drive vẫn được tiếp tục.

## Đầu vào và nơi lưu

Mặc định Drive root là `/content/drive/MyDrive/CoMind`. Mã tái sử dụng video đã có ở cấu trúc downloader hiện tại hoặc cấu trúc cũ sau khi kiểm tra hash. Nếu cần tải video mới, file nằm trong `/content/comind-sgg-work/<recording_ID>/mp4s/` của máy ảo Colab. Video có sẵn trên Drive được giữ nguyên.

Khi bỏ `--keep-videos`, chỉ video tạm do pipeline tải mới được xóa sau khi cả hai graph hoàn tất, hash hợp lệ và timeline hai view khớp. Quá trình tải sáu nhóm dữ liệu nguồn bằng `load_comind_drive.py` vẫn hoạt động như trước; lệnh SGG là `sgg_pipeline.py`.

```text
MyDrive/CoMind/
  sgg-settings.json
  sgg-status.json
  annotations/                           # Ba JSON chung gốc
  metadata/<recording_ID>/               # Manifest, danh sách nguồn đã chọn
  recordings/<recording_ID>/
    leader_gaze/general_eye_gaze.csv
    leader_hands/hand_tracking_results.csv
    transcripts/
      leader_trimmed_sync_transcript.json
      helper_trimmed_sync_transcript.json
    annotations/
      joint_attention.json
      interaction_anticipation.json
      handover.json
    graphs/
      leader/graphs.jsonl.gz
      leader/progress.json
      helper/graphs.jsonl.gz
      helper/progress.json
      timeline.jsonl.gz
    sync-status.json
```

Phiên thiếu hand tracking có báo cáo `missing_source_modality`, không tạo dữ liệu tay giả để lấp chỗ trống.

## Nội dung graph

Mỗi dòng giải nén của `graphs.jsonl.gz` là một JSON graph tại một thời điểm. Các trường chính gồm `recording_id`, `view`, `time_s`, `source_frame_index`, kích thước ảnh, `nodes` và `edges`. Node có ID theo view, nhãn detector, box xyxy, điểm detector và thời điểm detector gần nhất. Edge có subject, relation, object và điểm quan hệ của model.

Các quan hệ của bản thử là `on`, `inside`, `holding`, `touching`. Detector dùng danh sách đồ vật bếp trong `sgg-settings.json`, tối đa 20 track mỗi chunk. Score của model được giữ nguyên để đánh giá/hiệu chỉnh trên dữ liệu gán tay, chưa được diễn giải thành xác suất chính xác trên CoMind.

SAM2 chỉ theo dõi tiến theo thời gian. ID giữa các chunk được nối bằng frame biên chung và matching IoU theo nhãn, đây là heuristic có thể lỗi sau che khuất hoặc với các đồ vật giống nhau. ID của hai view được đặt namespace riêng. Ghép hai node cùng một vật giữa leader/helper cần bước đối chiếu riêng.

Mỗi view chỉ có một file graph gzip, dù xử lý theo chunk 4 giây. Checkpoint ghi số graph, tổng node/edge, mốc cuối, carry tracking, hash và byte offset đã commit. Chunk cuối bị ngắt được tính lại, timeline tránh lặp mốc ở biên chunk.

`timeline.jsonl.gz` liên kết row leader và helper ở cùng `time_s`, sai khác cho phép tối đa 1 ms. File còn chứa `completed_transcript_words`, gồm các từ có thời điểm kết thúc không muộn hơn mốc graph và `audio_stream` tương ứng. `audio_stream` cho biết nguồn transcript; speaker ID vẫn giữ theo nguồn, chưa được suy thành danh tính leader/helper. Gaze/hand chưa được ghép vào graph khi đồng hồ chưa có ánh xạ.

## Kiểm tra kết quả trên Colab

```python
from pathlib import Path
import gzip
import json
from load_comind_drive import select_sessions

root = Path('/content/drive/MyDrive/CoMind')
for rid in select_sessions(2):
    folder = root / 'recordings' / rid
    print('RECORDING', rid)
    for view in ('leader', 'helper'):
        progress = folder / 'graphs' / view / 'progress.json'
        if not progress.exists():
            print(view, 'đang chờ xử lý')
            continue
        state = json.loads(progress.read_text())
        print(view, {k: state[k] for k in
                     ('complete', 'graph_count', 'node_count', 'edge_count', 'last_time_s')})
        with gzip.open(progress.parent / 'graphs.jsonl.gz', 'rt') as handle:
            print('Graph đầu tiên', json.loads(next(handle)))
    sync = folder / 'sync-status.json'
    if sync.exists():
        print('Đồng bộ', json.loads(sync.read_text()))
```

Trước khi mở rộng 30 phiên, xem số node/edge, ID tracking, các quan hệ dự đoán và thời gian mỗi chunk. Phiên hoàn tất SGG sẽ có `complete: true` ở cả hai `progress.json`. `all_modalities_aligned` hiện vẫn là `false` vì còn thiếu ánh xạ cảm biến.

## Dữ liệu đầu vào cho encoder belief

Sau mỗi recording hoàn tất, pipeline tự xuất ba file trong `recordings/<ID>/belief_inputs/`.

- `steps.jsonl.gz` chứa một bước tại mỗi mốc 0,2 giây, hai graph quan sát, sự kiện lời nói mới và ID ngữ cảnh gần đây.
- `speech_events.json` chứa transcript segment hoàn tất, `event_id` ổn định, start/end, nguồn audio và text. Segment được phát một lần tại bước đầu tiên có `time_s >= available_s`. Ngữ cảnh 10 giây giữ tham chiếu, bộ nhớ belief của encoder phải được truyền qua các bước. Dữ liệu giữ văn bản để có thể thay text encoder sau này.
- `manifest.json` ghi số bước, số sự kiện, hash và tình trạng nhãn/đồng bộ. `gaze_features` và `hand_features` hiện là null vì timestamp cảm biến chưa được ánh xạ. Graph quan sát human chưa có ý nghĩa là nhãn belief.

Nếu đã có graph và chỉ cần xuất định dạng đầu vào mới, chạy lệnh CPU sau với ID recording thực tế.

```python
!python belief_data.py /content/drive/MyDrive/CoMind/recordings/RECORDING_ID
```

Nhãn belief được cung cấp riêng tại `recordings/<ID>/labels/belief_labels.jsonl`, mỗi dòng một truy vấn được gán nhãn độc lập. Schema gồm `time_s`, `person_id`, `object_id`, `candidate_location_ids`, `allowed_location_ids`, `label_source`. Thời điểm nhãn phải khớp graph xuất. Tập allowed là vị trí được nhãn hỗ trợ, có thể gồm nhiều vị trí khi chỉ biết vật đã rời vị trí cũ. Người gán nhãn cần đối chiếu ID đồ vật và vị trí của từng view.

```python
from belief_data import iter_training_steps
for inputs, labeled_queries in iter_training_steps(recording_directory):
    # Truyền recurrent state theo thứ tự inputs['time_s'].
    # Tính loss tại labeled_queries; bước không có nhãn vẫn cập nhật state.
    pass
```

Loader giữ nhãn ngoài inputs và dừng nếu chưa có nhãn hoặc thời điểm không khớp. Pipeline chưa cung cấp bộ nhãn belief tự động. Hai nguồn audio có thể thu cùng một lời nói; provenance được giữ để kiểm tra tương quan trước khi dùng làm bằng chứng độc lập. Bản đầu phát toàn bộ segment sau khi segment kết thúc; phân tích mệnh đề tăng dần là bước phát triển tiếp.

Nguồn model [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [SAM2](https://github.com/facebookresearch/sam2), [RelateAnything](https://github.com/Maelic/RelateAnything). Dữ liệu và đồng bộ do tác giả mô tả tại [CoMind](https://comind.ethz.ch/) và [paper, phần 3.3](https://arxiv.org/html/2607.06691v1#S3.SS3).
