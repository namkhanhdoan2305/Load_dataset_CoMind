# Load_dataset_CoMind

Tải bộ CoMind chọn lọc trên Colab CPU, lưu vào Google Drive hoặc chuyển tới Hugging Face. Máy cá nhân chỉ mở notebook.

Lấy hai video ego MP4 đồng bộ, gaze và hand tracking của leader, transcript và nhãn chung chính thức. Tùy chọn lấy thêm leader calibration. Chạy lại để tiếp tục tải và đối chiếu hash, HF upload thành công được xác minh trước khi xóa file tạm.

Đọc [hướng dẫn Colab, Google Drive và đăng nhập HF](COLAB_HF_GUIDE.md).

```bash
python comind_to_hf.py --split train --limit 1 --dry-run
python comind_to_hf.py --split train --limit 1 --drive-dir /content/drive/MyDrive/CoMind
```

Bản gốc CoMind do ETH Zurich và cộng sự công bố, giấy phép CC BY 4.0. Ghi nguồn khi chia sẻ lại dữ liệu. [Trang tác giả](https://comind.ethz.ch/) và [paper](https://arxiv.org/abs/2607.06691).

Kiểm tra luồng xử lý với dữ liệu tổng hợp bằng `python -m unittest discover -s tests -v`.
