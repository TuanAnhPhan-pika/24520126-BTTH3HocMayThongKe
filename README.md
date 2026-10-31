# BÀI TẬP THỰC HÀNH 3 MÔN HỌC MÁY THỐNG KÊ

**SINH VIÊN:** PHAN TUẤN ANH  
**MSSV:** 24520126

## CÂU 1

**FILE:** `assignment1.py`, `linear_regression.py`

Xây dựng mô hình Linear Regression bằng NumPy. Huấn luyện và đánh giá trên bộ dữ liệu Forest Fires để dự đoán diện tích cháy rừng.

## CÂU 2

**FILE:** `assignment2.py`

Chuẩn hóa các biến liên tục về trung bình 0 và phương sai 1. So sánh RMSE của mô hình trên dữ liệu trước và sau chuẩn hóa.

## CÂU 3

**FILE:** `assignment3.py`  
**HÌNH KẾT QUẢ:** `heatmap.png`

Vẽ heatmap thể hiện tương quan giữa các biến. Loại bỏ cột `DMC` và so sánh RMSE trước và sau khi loại bỏ.

## CÂU 4

**FILE:** `assignment4.py`

Xây dựng Linear Regression bằng Scikit-learn. So sánh RMSE với mô hình tự xây dựng bằng NumPy trên cùng tập huấn luyện và kiểm tra.

## DỮ LIỆU

- `data/forestfires.csv`: bộ dữ liệu Forest Fires.
- `data/forestfires.names`: mô tả bộ dữ liệu.

Dữ liệu được chia thành 414 mẫu huấn luyện và 103 mẫu kiểm tra. Biến mục tiêu `area` được biến đổi bằng `log1p` khi học và chuyển ngược khi tính RMSE. RMSE càng nhỏ thì sai số dự đoán càng thấp.

## HƯỚNG DẪN CÀI ĐẶT

Cài Python 3.10 trở lên và Git, sau đó mở Terminal:

```powershell
git clone https://github.com/TuanAnhPhan-pika/24520126-BTTH3HocMayThongKe.git
cd 24520126-BTTH3HocMayThongKe
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## HƯỚNG DẪN CHẠY

Chạy từng câu từ thư mục dự án:

```powershell
.venv\Scripts\python.exe assignment1.py
.venv\Scripts\python.exe assignment2.py
.venv\Scripts\python.exe assignment3.py
.venv\Scripts\python.exe assignment4.py
```

Kết quả RMSE hiển thị trong Terminal. Câu 3 tạo file `heatmap.png` trong thư mục dự án.

Giữ các file Python và thư mục `data` theo cấu trúc của repo; câu 4 sử dụng các hàm trong `assignment2.py`.
