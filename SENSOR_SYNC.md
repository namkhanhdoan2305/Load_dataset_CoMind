# Đồng bộ gaze/hand với MP4 và graph — 20/11/2026

## Mốc gốc đã tìm được

CoMind MP4 có metadata `udta/meta/ilst/desc` (`description`). Với recording thử `8ee6c694-38be-4a90-ad5a-75ef4449e972`, tag leader chứa **3759402423575**, đơn vị nanosecond của Aria device clock. Trước đây pipeline bỏ sót tag này nên để sensor alignment unresolved.

Đã kiểm chứng trực tiếp với VRS cùng recording:

| Kiểm tra | Giá trị |
|---|---:|
| Origin trong description MP4 | 3759402423575 ns |
| Timestamp RGB đầu VRS | 3759402423575 ns |
| Sai khác anchor đầu | **0 ns** |
| Timestamp RGB cuối VRS | 5462996473075 ns |
| Span RGB theo device clock | 1703,5940495 s |
| PTS frame MP4 cuối | 1703,566666667 s |
| Sai khác anchor cuối | **27,382833 ms** |

Video CFR 30 FPS. Sai khác cuối nằm trong dung sai resampling 1,5 frame (50 ms); đây không phải tuyên bố mọi modality chính xác tới nanosecond. Parser cũng hỗ trợ description dạng vector timestamp từng frame của Aria, kiểm tra đúng số frame và thứ tự. Khi có vector, tham chiếu dựa trên frame tương ứng và bị chặn không vượt mốc nominal hiện tại.

Đã đọc **1.602.544 byte metadata MP4** và **433.527 byte metadata VRS** cho kiểm chứng đầu/cuối. Đọc bằng HTTP Range có kiểm tra Content-Range và giới hạn byte; không tải VRS 21 GB. Nếu metadata thiếu, unit sai, anchor không khớp, video VFR hoặc định dạng VRS chưa hỗ trợ, code dừng kiểm chứng; không suy mốc từ dòng CSV đầu và không bật cờ đồng bộ giả.

Với scalar origin đã được kiểm chứng, công thức là:

```text
device_sample_ns = tracking_timestamp_us * 1000
sample_video_time_s = (device_sample_ns - origin_device_ns) / 1e9
reference_device_ns = origin_device_ns + leader_source_pts_us * 1000
```

Graph đã có `time_us` theo lưới 0; 200000; 400000... và PTS frame thực tế. Sensor ghép theo PTS frame leader, sau đó gắn vào step chung. Leader/helper vẫn dựa trên cặp MP4 synced của tác giả; kiểm tra PTS giữa hai view giữ nguyên dung sai 1 ms. Transcript vẫn dùng start/end tương đối của MP4 do nguồn cung cấp.

## Quy tắc ghép

- Chọn **mẫu mới nhất không vượt reference_device_ns**, giữ nguyên source timestamp.
- Gaze nguồn phiên thử khoảng **10 Hz**; giới hạn độ cũ 150 ms.
- Hand nguồn khoảng **30 Hz**; giới hạn độ cũ 50 ms.
- Không lấy mẫu ở tương lai dù nó gần hơn. Không nội suy dùng mẫu tương lai.
- Không tìm pose valid cũ để lấp một mẫu hiện tại invalid. Tracking confidence <0,5, landmark thiếu/NaN hoặc pose toàn zero được mask.
- Mẫu chưa có hoặc quá cũ: feature null. Mẫu hiện tại có nhưng tracking invalid: cấu trúc feature có `valid=false`, landmark/wrist pose không được đưa vào như bằng chứng valid.
- Giữ policy theo thời gian thực, không theo số step. Export stride 1/2/5 tại cùng mốc chọn cùng sensor sample.

Gaze còn ở CPF (góc radian, origin/depth mét). Hand landmarks/wrist pose còn ở frame thiết bị leader, đơn vị mét. Đồng bộ thời gian không tự biến tọa độ này thành world frame hoặc gán gaze vào object. Muốn chiếu lên bbox cần calibration và quy ước hình học riêng.

## Nội dung lưu

`recordings/<ID>/sensor-clock.json` lưu origin, mapping kind, MP4 frame timing, anchor VRS đầu/cuối, sai số và dung sai, hash nguồn MP4/VRS/CSV, số byte metadata đã đọc. File cache chỉ được tái sử dụng khi profile, recording và hash nguồn khớp.

Mỗi row `belief_inputs/steps.jsonl.gz` có:

```text
time_us, clock_id
robot_observation_graph, human_observation_graph
new_speech_events, recent_speech_event_ids
gaze_features
  aligned_time_us, clock_id
  source_timestamp_us, source_clock_id
  reference_video_time_us, reference_device_time_ns, age_ns
  valid, coordinate_frame, angle_unit, length_unit, values
hand_features
  aligned_time_us, clock_id
  source_timestamp_us, source_clock_id
  reference_video_time_us, reference_device_time_ns, age_ns
  valid, coordinate_frame, length_unit
  hands.left/right: valid, tracking_confidence, landmarks_device_m,
                    wrist_pose, surface_normals
```

`belief_inputs/sensor_alignment.json` lưu số matched/valid/missing, lý do thiếu, độ cũ lớn nhất, policy và hash CSV/clock mapping. CSV được đối chiếu đúng SHA256 trong source manifest để chặn ghép nhầm recording. Raw CSV vẫn giữ nguyên.

`sync-status.json` và `belief_inputs/manifest.json` ghi `sensor_clock_mapping_verified`, availability và `all_modalities_aligned`. `timed_model_inputs_aligned` kiểm tra clock sensor, đủ hai CSV, transcript đúng hash/bounds và graph qua PTS; `all_modalities_aligned` còn kiểm tra annotation benchmark timing. Xem [DATA_REVIEW.md](DATA_REVIEW.md). Sample thiếu/invalid có mask riêng, không tự làm clock trở thành sai. `physical_audio_video_alignment_verified` vẫn false vì chưa nghe/xem đối chiếu độc lập audio/video. Task progress, nhãn belief và matching cùng object giữa hai view vẫn cần bước riêng.

Các graph cũ đã trích có thể chứa nhãn trạng thái sensor tại lúc trích là unresolved. Sau retrofit, trạng thái cuối được báo trong sidecar và belief export; timestamp/node/edge của graph giữ nguyên. Timeline được xuất lại theo trạng thái clock mới.

## Chạy trên Colab

Pipeline SGG hiện hành tự chuẩn bị clock và ghép sensor. Nếu đã có hai graph complete, chỉ chạy CPU để bổ sung dữ liệu, **không chạy lại REACT++**:

```python
%cd /content/Load_dataset_CoMind
!git pull --ff-only
!python sensor_sync.py /content/drive/MyDrive/CoMind_REACTpp/recordings/8ee6c694-38be-4a90-ad5a-75ef4449e972
```

Nếu muốn chuẩn bị clock trước khi graph xong:

```python
!python sensor_sync.py /content/drive/MyDrive/CoMind_REACTpp/recordings/8ee6c694-38be-4a90-ad5a-75ef4449e972 --prepare-only
```

Lệnh prepare-only không xuất feature khi chưa có paired graphs. Muốn cập nhật các biến thể stride đã có, chạy lại `belief_data.py ... --step-stride N` sau khi clock sẵn sàng. Không có dependency GPU mới cho bước đồng bộ.

## Kiểm chứng đã thực hiện

**80 kiểm thử cho luồng hiện hành đã qua**, gồm parser metadata scalar/vector, CFR/VFR, unit/anchor sai, VRS front/end index, từ chối server bỏ qua Range, source CSV hash, causal lookup, mask pose invalid, stride consistency, export và loader.

Đã tải và kiểm tra hash **hai CSV thật** của recording thử, rồi ghép trên toàn bộ **8.518 mốc 5 FPS, từ 0 đến 1703,4 s**. Đây là kiểm chứng sensor trên lưới thời gian thật, không phải lượt chạy model SGG toàn phiên.

| Cue | Matched | Có giá trị valid | Thiếu mẫu |
|---|---:|---:|---:|
| Gaze | 8517 | 8517 | 1 |
| Hand (ít nhất một tay valid) | 8513 | 7712 | 5 |

Mẫu gaze đầu tiên trễ hơn video zero **100,425 microsecond**, nên step 0 để null theo causal policy. Hand CSV bắt đầu sau origin khoảng **0,9666 s**, nên năm step đầu thiếu mẫu. Độ cũ lớn nhất đã chọn: gaze **99,979575 ms**, hand **33,976575 ms**. Tất cả mẫu được chọn đều không vượt reference; không lấy tọa độ zero của pose invalid làm pose thật.

Chưa kiểm chứng đủ 80 recording và chưa đo độ chính xác gaze/pose vật lý bằng ground truth độc lập. Mỗi recording mới phải qua cùng kiểm tra metadata/hash trước khi ghép.

Nguồn quy ước: [Aria MP4 timestamp export](https://github.com/facebookresearch/projectaria_tools/blob/481eae4cca842b2acf143ffc44775689514f8787/projectaria_tools/tools/vrs_to_mp4/vrs_to_mp4_utils.py), [VRS file format](https://github.com/facebookresearch/vrs/blob/main/vrs/FileFormat.h), [CoMind processing](https://arxiv.org/html/2607.06691v1#S3.SS3).
