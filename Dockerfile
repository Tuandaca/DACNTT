FROM python:3.9-slim

# Cài đặt thư viện hệ thống
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Bước 1: Copy file requirements từ thư mục con vào
COPY app/backend/requirements.txt .

# Cài đặt torch bản CPU và các thư viện
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Bước 2: Copy toàn bộ nội dung trong app/backend vào thư mục hiện tại của Docker (/code)
COPY app/backend/ .

# Thiết lập User (Hugging Face yêu cầu)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user app/backend/ $HOME/app

# Chạy trên port 7860
CMD ["uvicorn", "unified_server:app", "--host", "0.0.0.0", "--port", "7860"]