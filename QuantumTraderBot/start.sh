#!/bin/bash
# Chạy ứng dụng web bằng Gunicorn
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app &

# Chạy bot trading
python main.py &

# Đợi cả hai tiến trình hoàn thành (hoặc giữ cho script không kết thúc)
wait