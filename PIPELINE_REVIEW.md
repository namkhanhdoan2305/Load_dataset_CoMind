# Review dữ liệu và đồng bộ — 20/11/2026

Đã cập nhật sang REACT++ YOLO12m ONNX + tracker bbox. Xem [hướng dẫn chạy hiện tại](SGG_COLAB_GUIDE.md). Các giới hạn đồng bộ bên dưới vẫn áp dụng.

## Kết luận kiểm tra mã

Pipeline đã thống nhất clock MP4 theo recording và ghép transcript bằng thời gian gốc. Chưa thể xác nhận đồng bộ đầy đủ mọi modality. Kiểm tra mã và kiểm thử giả lập không thay thế việc xem kết quả một recording thật trên Colab.

| Thành phần | Mã đang làm | Giới hạn cần review |
|---|---|---|
| Video leader/helper | Dùng hai MP4 `trimmed_sync`; graph trên cùng lưới 5 FPS; kiểm tra PTS decoder và ghép hai view | Dựa trên đồng bộ nguồn; không tự đo lại sai số vật lý giữa camera |
| Transcript segment | Giữ start/end nguồn; toàn bộ đoạn được phát đúng một lần sau end | Chưa xác minh độc lập độ lệch audio/video hoặc độ chính xác ASR |
| Transcript word | Timeline giữ từ đã kết thúc và timestamp integer; từ thiếu timestamp được đếm riêng | Word timeline chưa được dùng làm text input tăng dần trong encoder export |
| Gaze/hand | Ghép theo origin MP4, đối chiếu anchor RGB VRS đầu/cuối và hash CSV | Sample quá cũ/thiếu để null; pose invalid được mask; tọa độ vẫn CPF/device |
| Annotation | Lưu bản chung và tách đúng recording; giữ nguyên event ID, cả nhãn null | Chưa chuyển event ID thành thời gian |
| Task | Chưa xuất task graph, step hiện hành hay nhãn hoàn thành nhiệm vụ | Cần schema và sự kiện có mốc thời gian được kiểm chứng |
| Belief | Xuất đầu vào graph/text và loader nhãn riêng | Chưa có nhãn belief tự động, mô hình encoder/decoder hoặc training loop |
| Entity giữa hai view | Namespace `leader:<id>` và `helper:<id>` riêng | Chưa đối chiếu cùng một vật giữa hai camera |

Thứ tự các bước của task có thể cố định; thời điểm người thực hiện bắt đầu/kết thúc từng bước vẫn cần gắn với bằng chứng. Pipeline hiện chưa làm phần này. Không dùng số thứ tự annotation làm giây hoặc frame khi chưa xác minh quy ước nguồn.

## Quy ước mốc thời gian

- `clock_id`: `comind:<recording_ID>:synced_mp4`.
- `time_us`: số nguyên microsecond trên lưới 0, 200000, 400000...; `time_s` là cách biểu diễn cùng mốc bằng giây.
- Tracking và graph đều 5 FPS. Với video CFR 30 FPS, graph dùng frame 0, 6, 12... Chọn frame nguồn trước hoặc tại mốc lưới khi FPS lẻ.
- `source_pts_us` giữ timestamp decoder của frame thực tế. Code kiểm tra PTS với chỉ số frame/FPS, dung sai 1 ms; hai view phải cùng mốc lưới, chênh source PTS tối đa 1 ms. Đây là kiểm tra timestamp, không phải phép đo lại độ đồng bộ cảm biến.
- Các clock bị trộn recording hoặc sai đơn vị bị từ chối. Hash graph và checkpoint được kiểm tra trước khi export.
- Gaze/hand giữ `tracking_timestamp_us`, clock `comind:<ID>:leader_device_time`; lấy origin từ MP4, kiểm chứng với RGB VRS đầu/cuối. Không lấy dòng CSV đầu làm mốc 0. Chi tiết ở [SENSOR_SYNC.md](SENSOR_SYNC.md).

## Lời thoại đi qua nhiều step

Ví dụ đoạn nói bắt đầu 12,10 s, kết thúc 13,07 s. Trên lưới 0,2 s:

| Step | Nội dung đưa vào input |
|---|---|
| 12,20–13,00 s | Chưa đưa toàn bộ đoạn vào vì chưa tới end |
| 13,20 s | `new_speech_events` chứa đoạn đó một lần; thời điểm sẵn có vẫn là 13,07 s; trễ do step 0,13 s |
| 13,40 s trở đi | Giữ ID trong ngữ cảnh gần đây; không phát lại như bằng chứng mới |

`event_id` ổn định theo recording, stream, chỉ số segment và mốc nguồn. Context mặc định 10 giây chỉ điều khiển tham chiếu văn bản gần đây; belief state phải do encoder truyền qua thời gian, chưa tự giảm độ tin cậy vì lời nói đã cũ.

Đoạn nói dài sẽ trì hoãn nội dung tới cuối đoạn. Đây là lựa chọn phát sự kiện của bản thử. Timeline đã giữ từ kết thúc sớm hơn để có thể phát triển input tăng dần sau khi review; hiện input train vẫn dùng completed segment. Timestamp ASR nguồn được giữ nguyên, không coi đó là thời gian hoàn tất tính toán ASR online.

Tên stream leader/helper chỉ nguồn ghi âm. Speaker ID giữ theo nguồn, chưa được ánh xạ sang danh tính. Hai microphone có thể thu cùng câu; code giữ provenance, chưa khử lặp ngữ nghĩa và chưa coi hai bản thu là hai xác nhận độc lập. Transcript thiếu start/end segment gây lỗi; word thiếu thời gian được đếm riêng trong báo cáo đồng bộ.

`speech_events.json` là kho sự kiện toàn phiên để tra cứu. Input tại thời điểm t chỉ dùng `new_speech_events` và các ID đã xuất hiện; đưa toàn bộ kho tương lai vào mô hình sẽ gây rò rỉ thông tin.

`speech_alignment.json` cho biết mỗi event kết thúc lúc nào, phát ở step nào, trễ bao nhiêu, segment dài bao lâu và các sự kiện chưa phát vì kết thúc sau graph cuối được giữ lại. `physical_audio_video_alignment_verified=false` phản ánh việc chưa đối chiếu độc lập audio và hình. Cần nghe/xem đầu, giữa và cuối recording để phát hiện offset hoặc drift; ASR sai từ và nhận sai người nói là hai vấn đề riêng.

## Thư mục sau khi hoàn thành một recording

Đường dẫn mặc định trên Colab: `/content/drive/MyDrive/CoMind_REACTpp/`.

```text
CoMind_REACTpp/
  sgg-settings.json                  # Cấu hình perception, model revision được khóa
  sgg-status.json                    # Trạng thái toàn lượt chạy, recording/view hiện hành
  annotations/                      # Ba JSON annotation nguồn dùng chung + metadata
  metadata/<ID>/                    # Source healthcheck, selection và hash nguồn
  recordings/<ID>/
    leader_gaze/general_eye_gaze.csv
    leader_hands/hand_tracking_results.csv  # Có thể thiếu; báo rõ, không bù giả
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
    sensor-clock.json
    belief_inputs/
      steps.jsonl.gz
      speech_events.json
      speech_alignment.json
      sensor_alignment.json
      manifest.json
    belief_inputs_stride_2/          # Chỉ có khi yêu cầu export thêm
    belief_inputs_stride_5/          # Chỉ có khi yêu cầu export thêm
    labels/belief_labels.jsonl       # Người nghiên cứu cung cấp; pipeline không tự sinh
```

Video mới tải bằng SGG nằm tạm tại `/content/comind-sgg-work/<ID>/mp4s/`. `--keep-videos` giữ chúng trên máy ảo để review, không sao lưu qua Drive; Colab reset có thể mất chúng. Nếu đã tải video bằng downloader lên Drive, pipeline tái sử dụng và giữ nguyên hai thư mục `leader_video/`, `helper_video/` trên Drive.

Graph lưu ID track theo view, label, bbox xyxy, kích thước ảnh nguồn/ảnh giảm, score, thời điểm detection, quan hệ subject–predicate–object và score. Không lưu JPEG frame lâu dài. Timeline lưu liên kết hàng graph hai view và word vừa hoàn tất. Input step lưu hai graph, sự kiện text mới, ID context, đồng hồ, gaze/hand được ghép và mask/độ cũ của cue. Manifest giữ hash nguồn và output để đối chiếu giữa các phương án.

Leader graph hiện có trong export để dùng làm teacher/oracle theo thiết kế thí nghiệm. Khi đánh giá mô hình chỉ quan sát helper, phải kiểm soát đường vào này; dữ liệu thu từ góc nhìn người chưa phải nhãn niềm tin của người.

## Lượt thử một recording

Sau khi cài dependencies và mount Drive theo hướng dẫn:

```python
!git pull --ff-only
!python sgg_pipeline.py --sessions 1 --keep-videos
```

Seed mặc định 42; recording đầu là `8ee6c694-38be-4a90-ad5a-75ef4449e972`. Đổi target thì thay đường dẫn tương ứng trong các lệnh phía dưới.

Sau lượt chạy, kiểm tra hai `progress.json` đều complete, `sgg-status.json` complete, rồi xem `sync-status.json`, `belief_inputs/manifest.json` và `belief_inputs/speech_alignment.json`. `all_modalities_aligned` phản ánh clock mapping và sự hiện diện của các nguồn; xem thêm `sensor_alignment.json` về coverage/validity. Một recording đủ kiểm tra luồng dữ liệu và chọn cách xử lý; chưa đủ kết luận hiệu năng nghiên cứu.

Xem ít nhất vài đoạn ở đầu/giữa/cuối: caption đúng nội dung và mốc âm thanh, graph đúng vật/quan hệ, track ID qua che khuất, cùng vật giữa hai view và segment dài bị trì hoãn. Giữ graph nguồn và transcript nguyên bản để sau này thay cách chia step.

## So sánh độ dài step mà không chạy lại SGG

```python
!python belief_data.py /content/drive/MyDrive/CoMind_REACTpp/recordings/8ee6c694-38be-4a90-ad5a-75ef4449e972 --step-stride 2
!python belief_data.py /content/drive/MyDrive/CoMind_REACTpp/recordings/8ee6c694-38be-4a90-ad5a-75ef4449e972 --step-stride 5
```

| Stride | Khoảng bước | Tần số | Thư mục |
|---|---|---|---|
| 1 | 0,2 s | 5 FPS | `belief_inputs/` |
| 2 | 0,4 s | 2,5 FPS | `belief_inputs_stride_2/` |
| 5 | 1 s | 1 FPS | `belief_inputs_stride_5/` |

Code đọc cùng graph nguồn, kiểm tra lưới đều, giữ từng mốc theo stride và gom mọi speech event đã kết thúc kể từ bước giữ lại trước đó. Không bỏ dòng đơn giản từ steps đã xuất. Sự kiện giữ nguyên `event_id`, start/end, clock; audit ghi thêm trễ do lưới mới. Cuối phiên có thể còn sự kiện chưa tới bước được giữ tiếp theo, được báo trong audit; không phát ép vào quá khứ.

Giữ cố định perception, split theo recording, nhãn/query, candidate entity, cách dùng teacher và cấu hình học. Các lưới trên có mốc chung mỗi **2 giây**; đánh giá tại cùng các mốc này trong phần giao thời gian, giữ cùng số query. Loader từ chối nhãn ngoài lưới thay vì tự làm tròn. Đọc biến thể bằng `iter_training_steps(..., step_stride=2)` hoặc `5`.

So sánh này thay cả tần suất quan sát graph và cập nhật encoder; muốn tách hai tác động cần thí nghiệm riêng. Báo cáo cả chất lượng belief/false belief, speech latency và thời gian/bộ nhớ encoder. Vì SGG chạy một lần ở 5 FPS, đây chưa phải phép đo tiết kiệm thời gian perception khi hạ FPS.

## Kiểm chứng hiện có

78 kiểm thử tự động bằng dữ liệu nhỏ, video/perception giả lập đã qua: lọc đúng file nguồn, resume/hash, tách recording, clock/PTS, text không phát trước end, sự kiện phát một lần, gom speech khi đổi stride, báo tail, giữ graph gốc, từ chối nhãn sai lưới và chặn multimodal training khi clock chưa ánh xạ.

Chưa có lượt inference model thật trên T4 trong môi trường này. Báo cáo chưa chứng nhận tốc độ, VRAM, chất lượng SGG hay đồng bộ vật lý của recording. Kết quả một session do người dùng chạy là bước kiểm chứng tiếp theo.
