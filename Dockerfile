# Sử dụng Python 3.9 bản slim để nhẹ và ổn định
FROM python:3.9-slim

# 1. Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Thiết lập thư mục làm việc tạm thời để cài đặt thư viện
WORKDIR /code

# 3. Copy file requirements.txt (Lúc này nó đã nằm ngay thư mục gốc của gói deploy)
COPY requirements.txt .

# 4. Cài đặt torch bản CPU trước để tránh lỗi RAM và build nhanh hơn
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 5. Cài đặt các thư viện còn lại trong project
RUN pip install --no-cache-dir -r requirements.txt

# 6. Thiết lập User (Hugging Face bắt buộc dùng User ID 1000 để bảo mật)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 7. Thiết lập thư mục ứng dụng chính trong Home của User
WORKDIR $HOME/app

# 8. Copy toàn bộ nội dung đã được lọc sạch vào máy ảo
# Lệnh này sẽ bốc unified_server.py, bus_core.py... vào thẳng $HOME/app
COPY --chown=user . $HOME/app

# 9. Chạy FastAPI trên cổng 7860 (Cổng mặc định của Hugging Face)
# Đảm bảo unified_server.py là file chạy chính của bạn
CMD ["uvicorn", "unified_server:app", "--host", "0.0.0.0", "--port", "7860"]