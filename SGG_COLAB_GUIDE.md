# Chạy một recording với REACT++ + YOLO

Pipeline hiện dùng **REACT++ YOLO12m ONNX**, detector và relation head trong cùng checkpoint. Tracker bbox giữ ID theo nhãn, IoU và vận tốc; graph/tracking đều 5 FPS, không chạy SAM2. Checkpoint khóa revision và SHA256. Xem [kiểm chứng tích hợp](REACTPP_VALIDATION.md).

Kết quả mới lưu vào **`MyDrive/CoMind_REACTpp`** để giữ riêng graph của pipeline DINO/SAM2 cũ. Không ghép tiếp graph cũ bằng model mới. Video cũ trên ổ tạm Colab được kiểm tra hash rồi tái sử dụng; video có sẵn trong `MyDrive/CoMind` cũng được đọc lại, không di chuyển/xóa.

## Chuyển từ lượt đang chạy

Dừng cell SGG cũ bằng nút Stop trước, giữ nguyên runtime để còn video đã tải. Sau đó chạy các cell sau theo thứ tự. Không cần cài lại SAM2, RelateAnything hoặc Transformers.

```python
%cd /content/Load_dataset_CoMind
!git pull --ff-only
```

Nếu là runtime mới, clone và mount Drive trước:

```python
!git clone https://github.com/namkhanhdoan2305/Load_dataset_CoMind.git /content/Load_dataset_CoMind
%cd /content/Load_dataset_CoMind
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

Chọn GPU T4. Gỡ bản ONNX Runtime cũ nếu có để tránh hai package CPU/GPU cùng cung cấp một module. Requirements mới khóa ONNX Runtime GPU 1.26.0, CUDA 12/cuDNN 9; cài thêm runtime libraries từ extras khi cần. Wheel Linux hỗ trợ Python 3.13 đã được kiểm tra trên PyPI. CUDA 12 libraries được preload trước khi tạo session.

```python
!python -m pip uninstall -y onnxruntime onnxruntime-gpu
!python -m pip install -r requirements-sgg.txt
```

Chạy đúng một recording. `-u` in log ngay, `--keep-videos` giữ video tạm để đối chiếu sau.

```python
!python -u sgg_pipeline.py \
    --sessions 1 \
    --seed 42 \
    --max-hours 5 \
    --target /content/drive/MyDrive/CoMind_REACTpp \
    --source-dir /content/drive/MyDrive/CoMind \
    --keep-videos
```

Khi gián đoạn, chạy lại cùng lệnh để tiếp tục checkpoint **REACT++** đã commit. Source dir chỉ tái sử dụng video, không nhập graph DINO/SAM2 cũ. Nếu video còn trong `/content/comind-sgg-work/<ID>/mp4s/` thì downloader xác minh rồi bỏ qua tải lại; nếu runtime đã mất thì có thể cần tải lại video. Ngân sách kiểm tra giữa file/chunk, một chunk đang xử lý có thể vượt mốc.

Một recording không tương đương video 1 phút: recording đầu seed 42 `8ee6c694-38be-4a90-ad5a-75ef4449e972` có hai video, mỗi video **28 phút 23,6 giây, 30 FPS**, khoảng 8.518 graph/view. Thời gian trên T4 cần đo thực tế.

## Log để kiểm tra

```text
REACT++ ready: providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], ...
leader: source ... frames @ 30.000 FPS, duration approximately 1703.6s, 426 chunks, ...
leader: chunk 1/426 decoded 21 samples in ...s
leader: REACT++ 1 graphs, elapsed=...s
leader: REACT++ 10 graphs, elapsed=...s
leader: chunk 1/426, ... chunk took ...s
```

CUDA provider phải khởi tạo thành công; code dừng nếu session rơi hoàn toàn về CPU. CPU provider trong danh sách vẫn có thể xử lý các toán tử phụ trợ; không có nghĩa cả model chạy CPU. Cần log throughput thực tế để đánh giá. Pipeline không upload dataset lên Hugging Face; chỉ tải checkpoint ONNX khoảng **116,8 MB** từ đó.

## Thư mục kết quả

```text
MyDrive/CoMind_REACTpp/
  sgg-settings.json
  reactpp-model-info.json             # Model SHA, ORT version, providers, toàn bộ nhãn
  sgg-status.json
  metadata/<recording_ID>/
  annotations/                       # Ba annotation chung gốc
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
      leader/{graphs.jsonl.gz, progress.json}
      helper/{graphs.jsonl.gz, progress.json}
      timeline.jsonl.gz
    sync-status.json
    belief_inputs/
      steps.jsonl.gz
      speech_events.json
      speech_alignment.json
      manifest.json
```

Video mới tải nằm ở `/content/comind-sgg-work/<ID>/mp4s/`, không tự sao lưu lên Drive. Bỏ `--keep-videos` sẽ xóa video tạm sau khi hai view và export hoàn tất; video đã có trên Drive vẫn giữ nguyên.

Mỗi graph giữ bbox theo ảnh giảm cạnh dài tối đa 960 px, kích thước nguồn, timestamp frame nguồn, node ID namespace theo view, nhãn native checkpoint và score. ONNX letterbox về 640 để inference, bbox được đổi ngược về ảnh 960 trước khi lưu. Relation có `score` của predicate và `triplet_score` riêng; lọc theo triplet score >=0,1. Object score >=0,25; tối đa 40 object và 100 relation/frame. Không diễn giải các score là độ chắc chắn đã hiệu chuẩn.

Checkpoint có **133 nhãn object, 56 predicate foreground**. Giữ toàn bộ từ vựng nguồn, ví dụ `in`, `on`, `holding`, `touching`; không đổi `in` thành `inside` hoặc tự gán nhãn ngoài checkpoint. Một số vật/cấu trúc bếp như whisk, drawer, hand không có class riêng trong checkpoint PSG. `reactpp-model-info.json` giữ danh sách đầy đủ để kiểm tra độ phủ trên CoMind.

Tracker chỉ phát node thực sự có detection hiện tại. Track vắng tạm được giữ tối đa 1 giây để ghép lại; không phát vật bị mất detection thành quan sát giả. Đây là matching bbox có thể đổi ID khi che khuất, chuyển động nhanh hoặc vật giống nhau. Hai view chưa ghép ID cùng vật. Model quan sát leader dùng làm teacher/oracle theo thiết kế thí nghiệm, không phải nhãn belief.

## Đồng bộ và input belief

Mốc graph vẫn là `time_us = 0, 200000, 400000...`, `clock_id = comind:<ID>:synced_mp4`. Giữ PTS decoder; kiểm tra CFR và chênh PTS giữa hai view tối đa 1 ms. Transcript phát segment một lần sau end; giữ ID context và báo độ trễ trong `speech_alignment.json`. Đồng bộ vật lý audio/video cần kiểm tra bằng session thật.

Gaze/hand giữ CSV gốc với clock thiết bị riêng; chưa ánh xạ sang MP4 nên feature đang null, `all_modalities_aligned=false`. Annotation giữ event ID nguồn; task graph/progress và nhãn belief chưa được tự sinh. Loader supervised vẫn yêu cầu nhãn riêng và chặn multimodal train mặc định khi clock chưa giải quyết.

Đổi step trên CPU từ cùng graph gốc, không cần chạy lại REACT++:

```python
!python belief_data.py /content/drive/MyDrive/CoMind_REACTpp/recordings/RECORDING_ID --step-stride 2
!python belief_data.py /content/drive/MyDrive/CoMind_REACTpp/recordings/RECORDING_ID --step-stride 5
```

Hai bản thêm nằm ở `belief_inputs_stride_2/` (0,4 s) và `belief_inputs_stride_5/` (1 s). Gom đầy đủ speech event giữa các bước; không dùng `rows[::stride]`. So 5 / 2,5 / 1 FPS tại mốc chung mỗi 2 giây, cùng query/nhãn và split recording. Sự kiện cuối chưa tới bước kế tiếp được báo trong audit.

Nguồn: [REACT++](https://github.com/Maelic/SGG-Benchmark), [checkpoint PSG](https://huggingface.co/maelic/REACTPlusPlus_PSG), [ONNX Runtime CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [CoMind](https://comind.ethz.ch/).
