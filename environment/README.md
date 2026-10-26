# Môi trường chạy

`requirements-download.txt` đủ để tải model/dataset và chuẩn hóa dữ liệu.
`requirements-train.txt` là stack QLoRA; trên Windows cần cài PyTorch CUDA wheel phù hợp GPU trước khi train. Ví dụ cho máy hiện tại:

`code/download_assets.py` tải hai checkpoint đã pin revision: Sailor2-1B-Chat (model chính) và Qwen2.5-0.5B-Instruct (control). Hash và đường dẫn tương đối của cả hai được ghi trong `environment/asset_verification.json` và `environment/download_manifest.json`; đường dẫn tương đối giúp clone repo sang máy khác mà không lộ đường dẫn cá nhân.

```powershell
python -m pip install --force-reinstall --no-deps torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-train.txt
```

Kiểm tra trước khi bật NF4:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Không ghi `HF_TOKEN` vào file. Hai repo đang dùng là public/ungated; token chỉ cần nếu Hub giới hạn tốc độ.

