# Kiểm chứng tích hợp REACT++ — 20/11/2026

## Cấu hình hiện hành

- Model: `maelic/REACTPlusPlus_PSG`.
- Revision: `25aaf1838c3f6cee32f69a43871bab0870af4bea`.
- File: `yolo12m/react_pp_yolo12m.onnx`, 116.791.801 byte.
- SHA256: `38c4bdbb7e2a077131ffcbf57af726eb30574318e2344d5b4bfc1d3a48f5b2fe`.
- ONNX Runtime GPU 1.26.0 cho Colab, CUDA 12/cuDNN 9. Wheel Linux CPython 3.13 và extras đã được đối chiếu metadata PyPI.
- Preprocess/output contract đối chiếu `tools/eval_onnx_psg.py` và `tools/export_onnx.py` của tác giả tại commit `a6642012e66047c3544bed8784bbf9acdc87411b`.
- Input: `input`, float32 NCHW, batch 1, 640×640. Model khai báo batch động, code chỉ dùng batch 1.
- Output `boxes`: N×6, `[x1,y1,x2,y2,class_id,object_score]`; object ID bắt đầu từ 1.
- Output `rels`: M×5, `[subject_detection_index,object_detection_index,predicate_id,triplet_score,predicate_score]`; predicate 0 là background.

ONNX đã chứa backbone YOLO12m và relation head REACT++; không cần chạy YOLO riêng hoặc tải thêm backbone `.pt`. Mã hiện không cài/gọi Grounding DINO, SAM2 hoặc RelateAnything. File `sgg_models.py` còn trong repo để giữ lịch sử và các hàm bbox cũ; luồng hiện hành gọi `reactpp_models.Perception`.

## Các kiểm tra đã chạy

62 kiểm thử tự động đã qua trong môi trường riêng có NumPy/OpenCV, gồm các kiểm tra download/hash/resume, exact selection, thời gian và belief export trước đó, cộng thêm:

- Object ID bắt đầu từ 1 và predicate giữ đúng class metadata.
- Lọc/sắp xếp detection vẫn nối đúng endpoint của relation, không dùng thứ tự sau sorting làm source index.
- Node bị lọc kéo theo loại các edge trỏ vào node đó.
- Chặn index sai, class ngoài vocabulary, NaN và endpoint không tồn tại.
- Letterbox đúng màu BGR→RGB, float32 /255 và đổi ngược bbox.
- Tracker matching theo class, one-to-one, mất detection tạm, hết TTL và resume carry JSON.
- Track không thấy trong frame hiện tại không được xuất thành observation.
- Chunk sau bỏ frame biên đã tiêu thụ; không cập nhật tracker hai lần cùng frame.
- Chặn state từ tương lai và settings từ pipeline DINO/SAM2 cũ.
- Chặn session chỉ khởi tạo được CPU provider khi chạy luồng GPU.

Đã tải checkpoint ONNX thật và chạy inference **CPUExecutionProvider, ONNX Runtime 1.26.0**:

1. Đọc input/output schema và metadata thật: **133 object classes, 56 foreground predicates**.
2. Ảnh trống trả output rỗng và được parser chấp nhận.
3. Đọc khoảng 5,8 MB gồm đầu MP4 và metadata bằng HTTP Range của recording `8ee6c694-38be-4a90-ad5a-75ef4449e972`, tạo file sparse chẩn đoán tạm để decode các frame đầu; không tải nguyên video.
4. Decode frame CoMind thật, preprocess và chạy ONNX; có detection và relation hợp lệ.
5. Chạy tích hợp decoder + ONNX + tracker qua hai đoạn có biên chung, serialize carry rồi tiếp tục. Kết quả có đúng **5 graph tại 0; 0,2; 0,4; 0,6; 0,8 giây**, không lặp biên, các edge tham chiếu node tồn tại. Số node lần lượt 6/4/5/5/5; số edge 6/6/4/6/4.

Đây là kiểm tra thực thi và định dạng, chưa phải đánh giá độ chính xác hay kết luận về ID tracking trên cả recording. Không dùng thời gian CPU này dự báo throughput T4. Chưa chạy benchmark GPU T4 hoặc cả session trong môi trường assistant.

## Thay đổi và giới hạn cần giữ khi review

Graph vẫn 5 FPS và giữ nguyên quy ước clock/timestamp/transcript. Tracking bbox hiện cũng 5 FPS; bản SAM2 cũ theo dõi 10 FPS. Vì model, vocabulary và tracking đều thay đổi, kết quả hai pipeline cần xem như hai cấu hình perception khác nhau; không quy toàn bộ chênh lệch cho relation head.

PSG có các quan hệ `in`, `on`, `holding`, `touching`, `cooking`... nhưng thiếu class riêng cho một số vật như `whisk`, `drawer`, `hand`. Code giữ nhãn gốc và dump đầy đủ vào `reactpp-model-info.json`; không tự tạo class hoặc đổi `in` thành `inside`.

Tracker là heuristic bbox có vận tốc, giữ ID khi có match hợp lệ và vắng tối đa 1 giây. Chưa có appearance re-ID, hình học 3D hoặc matching cùng vật giữa hai view. Object đã ra khỏi view rồi quay lại lâu có thể có ID mới. Gaze/hand đã có [ánh xạ timestamp được kiểm chứng](SENSOR_SYNC.md). Trạng thái task, nhãn belief và kiểm chứng audio/video vật lý vẫn là các phần chưa hoàn tất đã nêu trong báo cáo đồng bộ.

Output mới ở `MyDrive/CoMind_REACTpp`. `--source-dir` chỉ đọc lại video ở thư mục nguồn cũ; không di chuyển file, không nhập graph/label cũ. Video tạm cùng work directory được xác minh rồi tái sử dụng. Thay model/settings trong cùng target gây lỗi thay vì ghép lẫn kết quả.

## Nguồn chính thức

- [Inference/evaluation contract](https://github.com/Maelic/SGG-Benchmark/blob/a6642012e66047c3544bed8784bbf9acdc87411b/tools/eval_onnx_psg.py).
- [Checkpoint được khóa](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/25aaf1838c3f6cee32f69a43871bab0870af4bea/yolo12m).
- [ONNX Runtime CUDA và preload libraries](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
- [Hướng dẫn chạy hiện tại](SGG_COLAB_GUIDE.md).
