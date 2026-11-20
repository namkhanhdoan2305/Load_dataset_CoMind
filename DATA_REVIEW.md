# Review sáu nhóm dữ liệu và đồng bộ — 21/11/2026

## Kết luận và phạm vi

Pipeline chọn đủ sáu **nhóm** đã thống nhất. Một recording có **6 payload nguồn** (hai MP4, gaze CSV, hand CSV, hai transcript JSON) và **3 annotation JSON chung** tải một lần, sau đó tách đúng recording. Số sáu nhóm không có nghĩa chỉ có sáu file tổng cộng.

Graph, gaze, hand và lời nói có cùng trục thời gian MP4 theo recording. Annotation benchmark đã được chuẩn hóa bằng `start_time/end_time` nguồn và đối chiếu `start_frame/end_frame`. Nhãn benchmark được lưu riêng, không đưa vào observation input. Task graph/tiến trình công thức và nhãn belief chưa được tự sinh.

Đã review mã, kiểm thử và đọc dữ liệu nguồn thật của recording `8ee6c694-38be-4a90-ad5a-75ef4449e972`. Chưa trực tiếp đọc output đang nằm trên Drive/Colab của người dùng; dùng lệnh audit bên dưới để kiểm chứng đúng output đó. Chưa xác minh đầy đủ 80 phiên, độ chính xác perception hoặc audio/video bằng người nghe/xem độc lập.

## Từng nhóm

| Nhóm | File nguồn chọn chính xác | Đồng bộ/lưu |
|---|---|---|
| Leader | `mp4s/leader_trimmed_sync.mp4` | Graph human-view 5 FPS; `time_us`, PTS decoder, source frame index/hash |
| Helper | `mp4s/helper_trimmed_sync.mp4` | Graph robot-view 5 FPS; cùng grid, chênh source PTS hai view ≤1 ms |
| Gaze | `mps_leader_trimmed_vrs/eye_gaze/general_eye_gaze.csv` | Origin MP4 kiểm chứng VRS; mẫu mới nhất trước frame, age ≤150 ms |
| Hand | `mps_leader_trimmed_vrs/hand_tracking/hand_tracking_results.csv` | Cùng clock leader; age ≤50 ms; confidence/landmark validity được mask |
| Lời nói | `transcripts/{leader,helper}_trimmed_sync_transcript.json` | Hash đúng nguồn; segment và word có start/end, đưa vào bước không sớm hơn end |
| Annotation | Joint attention, SCOIA, handover consolidated JSON | Hash nguồn, tách theo recording; interval microsecond và chỉ mục theo step |

Không dùng gapfill, transcript `.ass`, file `with_profanity` thay thế, merged video, GoPro, SLAM/scan hoặc VRS nguyên file trong phần tải payload này. Có đọc bounded metadata MP4/VRS để xác minh clock, không tải VRS 21 GB. Video mới tải nằm tạm trên Colab; giữ bằng `--keep-videos`. Khi xóa video sau xử lý, graph và provenance/hash vẫn giữ trên Drive.

## Mốc chung và kiểm chứng gaze/hand

`clock_id = comind:<ID>:synced_mp4`; graph 5 FPS ở `time_us = 0, 200000, 400000...`. Trong phiên mẫu, origin MP4 **3759402423575 ns** khớp RGB đầu VRS **0 ns sai khác**; anchor cuối sai khác **27,382833 ms**, trong dung sai resampling 1,5 frame ở 30 FPS. Không lấy dòng CSV đầu làm mốc 0. Đọc khoảng 2 MB metadata, không tải VRS nguyên file.

Đã đối chiếu hash CSV và ghép trên cả 8.518 mốc từ 0 đến 1703,4 s:

| Dữ liệu | Matched | Valid | Thiếu mẫu |
|---|---:|---:|---:|
| Gaze | 8517 | 8517 | 1 |
| Hand, ít nhất một tay valid | 8513 | 7712 | 5 |

Mẫu gaze đầu nằm sau video zero 100,425 µs nên step 0 để null. Hand bắt đầu sau zero khoảng 0,9666 s nên thiếu năm step đầu. Độ cũ cực đại đã chọn: gaze 99,979575 ms, hand 33,976575 ms. Mẫu thiếu/quá cũ giữ null; pose invalid không được thay bằng pose valid cũ. Thông tin này khác với sai số đồng hồ. Gaze còn trong CPF, hand còn trong frame thiết bị; chưa tự chiếu thành world-coordinate hoặc object gaze target.

## Transcript

Hai file thật đã được kiểm tra SHA256: **304 segment leader +375 segment helper =679 segment**, **3.548 từ có timestamp**, không có từ thiếu timing trong phiên mẫu. Tất cả start/end trong thời lượng video.

Review phát hiện input chỉ dùng completed segment có thể trì hoãn lời nói quá lâu: segment leader đầu kéo từ 1,988 tới 28,924 s. Đã bổ sung **`new_speech_words`** vào mỗi step; từ được phát sau end của chính từ đó. `new_speech_events` vẫn giữ completed segment làm lựa chọn tương thích. Khi thiết kế encoder chọn word stream hoặc segment stream; không cộng hai bản như bằng chứng độc lập. Hai microphone cũng có thể thu cùng câu, speaker ID chưa được suy thành danh tính leader/helper.

Segment và word gốc được giữ đầy đủ trong file kho sự kiện. Với graph cuối 1703,4 s, **1 segment và 1 từ cuối** kết thúc ở 1703,447 s chưa được phát vào step nào. Đây là phần nằm sau horizon của lưới graph, không phải thiếu tải. Audit báo tail rõ ràng; không đưa chúng lùi về quá khứ. Muốn query tới cuối media ngoài lưới cần thiết kế bước EOS riêng. Chưa kiểm chứng độc lập transcript đúng từng từ, speaker hoặc độ lệch audio/video vật lý.

## Annotation

Phiên mẫu có **84 joint-attention +15 interaction-anticipation +1 handover =100 event**, đều có timing hợp lệ; frame/time đối chiếu không có lỗi. Event ID được giữ nguyên, không dùng số trong ID làm timestamp. Ví dụ SCOIA ID `000703` có `start_frame=702`; phải dùng trường explicit của nguồn.

`annotations/timed_events.json` giữ label và interval. `belief_inputs/benchmark_labels.jsonl.gz` liên kết label ID theo từng step: started/completed/active và overlap với khoảng kể từ step trước. Event ngắn nằm hoàn toàn giữa hai graph vẫn có trong overlap; không bị mất vì không active tại frame lấy mẫu.

Label `transcript_10s`, bbox/action/category GT chỉ nằm ở phần annotation riêng. Không đưa sẵn các đáp án này vào `steps.jsonl.gz`. Ba benchmark này chưa phải task-progress dense label hoặc nhãn human-belief/false-belief.

## Cờ và output

- `timed_model_inputs_aligned`: hai graph qua PTS, sensor clock được kiểm chứng, đủ hai CSV, hai transcript đúng hash và timestamp trong media.
- `all_modalities_aligned`: thêm điều kiện ba nhóm annotation benchmark đã có mapping thời gian đầy đủ. Có nhãn null/withheld hoặc thiếu source thì báo chưa đầy đủ, không tự tạo nhãn.
- `physical_audio_video_alignment_verified`: vẫn false; việc ánh xạ metadata không thay thế kiểm tra vật lý audio/video.
- `speech_events_unemitted` / `speech_words_unemitted`: báo tail ngoài step cuối. Validity/coverage sensor có báo cáo riêng.

```text
recordings/<ID>/
  sensor-clock.json
  sync-status.json
  annotations/
    joint_attention.json
    interaction_anticipation.json
    handover.json
    timed_events.json
  graphs/leader/, helper/, timeline.jsonl.gz
  belief_inputs/
    steps.jsonl.gz
    speech_events.json
    speech_words.json
    speech_alignment.json
    sensor_alignment.json
    benchmark_labels.jsonl.gz
    manifest.json
```

Review cũng đã sửa kiểm tra thiếu transcript: trước đó một file có thể thỏa điều kiện nhóm hai file. Hiện từng đường dẫn bắt buộc được kiểm tra và ghi `missing_files` riêng.

## Kiểm chứng đúng output Colab của bạn

Đợi hai graph complete. Sau khi cập nhật code, chạy CPU để xuất lại timing/cue/nhãn; không chạy lại REACT++:

```python
%cd /content/Load_dataset_CoMind
!git pull --ff-only
!python sensor_sync.py /content/drive/MyDrive/CoMind_REACTpp/recordings/8ee6c694-38be-4a90-ad5a-75ef4449e972
!python review_recording.py /content/drive/MyDrive/CoMind_REACTpp/recordings/8ee6c694-38be-4a90-ad5a-75ef4449e972
```

Audit chỉ đọc dữ liệu, không tải file hoặc chạy GPU. Kiểm tra nguồn đúng hash, hai graph complete/hash, source-video provenance, input hash, mốc view/step/sensor, lời nói không lặp/không vượt thời điểm step, annotation tách đúng recording và derived-file hashes. Nếu video đã xóa, báo `graph_provenance_verified_raw_not_retained`, không giả vờ kiểm tra lại bytes đã xóa. Thành công có `status=metadata_and_file_checks_passed`, `errors=[]`; vẫn báo các giới hạn vật lý/tail/validity.

**87 kiểm thử tự động đã qua**, gồm regression cho thiếu transcript, timing annotation, event ngắn, word stream, source hash, không rò label vào input và audit output.
