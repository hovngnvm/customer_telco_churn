# 📈 Telco Customer Churn Prediction

**Dự báo Khách hàng Rời bỏ Dịch vụ Viễn thông**

## 🧠 Giới thiệu Dự án

Đây là **dự án cuối kỳ môn Machine Learning**, tập trung vào việc phân tích dữ liệu khách hàng trong lĩnh vực viễn thông và xây dựng các mô hình học máy nhằm **dự đoán khả năng khách hàng rời bỏ dịch vụ (Customer Churn)**.

Bài toán churn prediction có ý nghĩa thực tiễn cao, giúp doanh nghiệp:

* Nhận diện sớm nhóm khách hàng có nguy cơ rời bỏ
* Chủ động triển khai các chiến lược giữ chân khách hàng
* Tối ưu chi phí marketing và nâng cao giá trị vòng đời khách hàng (Customer Lifetime Value)

## 🎯 Mục tiêu Dự án

* Phân tích hành vi và đặc điểm của khách hàng viễn thông
* Thực hiện **EDA (Exploratory Data Analysis)** để hiểu dữ liệu
* Tiền xử lý và chuẩn hóa dữ liệu theo quy trình Machine Learning chuẩn
* Xây dựng và so sánh nhiều mô hình dự báo Churn
* Lựa chọn mô hình tối ưu và triển khai dự báo cho khách hàng mới

## 📂 Cấu trúc Thư mục Dự án

```
Project_Cuoi_Ki/
│
├── data/  
│   └── Chứa dữ liệu gốc (Raw Data)
│
├── folder_clean_visual/  
│   └── Dữ liệu đã làm sạch sau EDA và các biểu đồ trực quan
│
├── folder_standardized/  
│   └── Dữ liệu đã chuẩn hóa (Train/Test)
│   └── Lưu Scaler và Encoder phục vụ dự đoán
│
├── models_and_results/  
│   └── Các mô hình đã huấn luyện (.pkl)
│   └── Kết quả đánh giá mô hình
│
├── EDA.ipynb  
│   └── Khám phá dữ liệu, xử lý missing values, trực quan hóa
│
├── processing.ipynb  
│   └── Feature Engineering, Encoding, Train/Test Split, Standardization
│
├── logisticRegression.ipynb  
│   └── Huấn luyện mô hình Logistic Regression
│
├── randomForest.ipynb  
│   └── Huấn luyện mô hình Random Forest
│
├── XGBoosting.ipynb  
│   └── Huấn luyện và tối ưu mô hình XGBoost
│
├── model_comparison.ipynb  
│   └── So sánh hiệu năng các mô hình
│
├── predict.ipynb  
│   └── Dự báo churn cho khách hàng mới
│
└── requirements.txt  
    └── Danh sách thư viện cần thiết
```

## ⚙️ Hướng dẫn Cài đặt & Chạy Dự án

### 1️⃣ Cài đặt môi trường

Khuyến nghị sử dụng **Anaconda** hoặc **Virtual Environment** để đảm bảo tính ổn định.

```bash
pip install -r requirements.txt
```

### 2️⃣ Quy trình Thực hiện

Để đảm bảo **tính nhất quán và tái lập kết quả**, các notebook cần được chạy theo đúng thứ tự:

1. **EDA.ipynb**

   * Đọc dữ liệu gốc
   * Xử lý missing values
   * Phân tích phân phối và mối quan hệ giữa các biến

2. **processing.ipynb**

   * Feature Engineering
   * One-Hot Encoding
   * Chia Train/Test
   * Chuẩn hóa dữ liệu

3. **Huấn luyện mô hình**

   * `logisticRegression.ipynb`
   * `randomForest.ipynb`
   * `XGBoosting.ipynb`

   👉 Áp dụng **SMOTE** để xử lý mất cân bằng dữ liệu (Imbalanced Dataset)

4. **model_comparison.ipynb**

   * Tổng hợp và so sánh các chỉ số đánh giá
   * Lựa chọn mô hình tốt nhất

5. **predict.ipynb**

   * Sử dụng mô hình tối ưu để dự báo churn cho khách hàng mới

## 🧪 Các Thuật toán Được Sử dụng

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

## 🛠️ Kỹ thuật & Phương pháp Nổi bật

* **SMOTE Pipeline** xử lý mất cân bằng dữ liệu
* **RandomizedSearchCV** tối ưu siêu tham số
* **Threshold Tuning** để cải thiện F1-Score
* Đánh giá bằng các metric: Precision, Recall, F1-Score

## 📊 Kết quả Nổi bật

* **Mô hình tốt nhất:** XGBoost Classifier
* **Ngưỡng phân loại tối ưu (Best Threshold):** ~0.54
* **Hiệu năng:**

  * F1-Score cao
  * Cân bằng tốt giữa Precision và Recall
* Mô hình phù hợp cho các bài toán churn prediction trong thực tế

## 📌 Kết luận

Dự án đã xây dựng thành công một **pipeline Machine Learning hoàn chỉnh**, từ phân tích dữ liệu, tiền xử lý, huấn luyện mô hình đến triển khai dự báo. Kết quả cho thấy **XGBoost kết hợp SMOTE và Threshold Tuning** là lựa chọn hiệu quả cho bài toán dự báo khách hàng rời bỏ dịch vụ viễn thông.
