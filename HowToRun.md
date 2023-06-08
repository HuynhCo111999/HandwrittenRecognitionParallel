# HandwrittenRecognitionParallel

# Đặt vấn đề

# Hướng dẫn chạy

- Chạy code tuần tự (dùng hàm correlate2d của thư viện scipy):
  cd Sequential
  !python sequential1.py
- Chạy code tuần tự (dùng hàm correlate tự viết):
  cd Sequential
  !python sequential2.py
- Chạy code song song:
  cd Parallel
  !pythn parallel.py

# So sánh thời gian chạy

Với phiên bản tuần tự, do tốc độ chạy khá chậm nên nhóm chỉ tiến hành phân loại trên tập dữ liệu `mnist` với 3 label 0, 1, 2 với 20 epochs.

- Tuần tự 1: ~ 170s (tầm 2m30s - 3m)
- Tuần tự 2: ~ 1150s (tầm 17m - 19m)
  Ta thấy với phiên bản tuần tự chạy hàm correlate2d do nhóm tự viết thì thời gian chạy rất lâu (chậm hơn tầm 6-7 lần với hàm dùng thư viện của scipy) nên bài toán này cần song song hóa để cải thiện tốc độ.
