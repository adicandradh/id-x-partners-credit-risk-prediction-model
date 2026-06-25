# ID/X Partners Credit Risk Prediction Model

## 📌 Project Type
Project-Based Internship – ID/X Partners: Data Scientist

## 📊 Overview
Project ini bertujuan untuk membangun model machine learning yang dapat memprediksi risiko gagal bayar (*credit default risk*) pada pengajuan pinjaman menggunakan dataset ID/X Partners. Model yang dihasilkan diharapkan dapat membantu proses *credit scoring* sehingga keputusan pemberian pinjaman menjadi lebih objektif, cepat, dan berbasis data.

Pipeline dikembangkan secara end-to-end mulai dari exploratory data analysis (EDA), data cleaning, feature engineering, preprocessing, hyperparameter tuning, evaluasi model, hingga analisis *feature importance*.

## 📂 Dataset Overview
Dataset yang digunakan merupakan data historis pengajuan pinjaman dari ID/X Partners yang terdiri dari **466.285 data** dengan **75 kolom** sebelum proses data cleaning dan feature engineering. Variabel target dibentuk dari kolom `loan_status` yang kemudian dikonversi menjadi `bad_flag`, di mana nilai `1` menunjukkan peminjam yang mengalami gagal bayar (*default*) dan nilai `0` menunjukkan peminjam yang tidak mengalami gagal bayar.

Hasil eksplorasi data menunjukkan bahwa dataset memiliki **missing values pada sejumlah fitur**, terutama pada variabel yang berkaitan dengan riwayat kredit dan informasi pinjaman, namun **tidak ditemukan data duplikat**. Oleh karena itu, proses preprocessing difokuskan pada penanganan *missing values*, *feature engineering*, *feature encoding*, serta penghapusan fitur yang berpotensi menyebabkan *data leakage* agar model dapat menghasilkan prediksi yang lebih objektif.

## 🎯 Objectives
- Memahami karakteristik data dan distribusi target melalui exploratory data analysis.
- Melakukan data cleaning dan preprocessing untuk meningkatkan kualitas data.
- Mengembangkan fitur-fitur baru yang dapat meningkatkan performa model.
- Membangun dan membandingkan model Logistic Regression, Random Forest, dan LightGBM.
- Melakukan hyperparameter tuning menggunakan RandomizedSearchCV.
- Mengevaluasi performa model menggunakan ROC-AUC dan Log Loss.
- Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap prediksi risiko gagal bayar melalui analisis *feature importance*.

## ⚙️ Machine Learning Pipeline
1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Feature Engineering
5. Missing Value Imputation
6. Ordinal Encoding
7. Label Encoding
8. Train–Validation Split
9. Feature Scaling
10. Hyperparameter Tuning using RandomizedSearchCV
11. Model Evaluation
12. Feature Importance Analysis

## 🛠️ Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM

## 🤖 Machine Learning Models
Project ini membandingkan tiga algoritma klasifikasi, yaitu:
- **Logistic Regression**
- **Random Forest**
- **LightGBM**

Hyperparameter tuning dilakukan menggunakan **RandomizedSearchCV** dengan **Stratified 3-Fold Cross Validation** untuk memperoleh kombinasi parameter terbaik pada masing-masing model.

## 📈 Model Performance

### Best Hyperparameters

| Model | Best Parameters |
|---------|----------------|
| Logistic Regression | `penalty = l2`, `C = 0.01`, `solver = saga`, `class_weight = balanced`, `max_iter = 1000` |
| Random Forest | `n_estimators = 200`, `max_depth = 10`, `min_samples_split = 5`, `min_samples_leaf = 1`, `class_weight = balanced` |
| LightGBM | `n_estimators = 100`, `learning_rate = 0.05`, `num_leaves = 63`, `max_depth = -1`, `class_weight = balanced` |

### Validation Data Performance

| Model | ROC-AUC | Log Loss |
|------------|--------:|---------:|
| Logistic Regression | 0.6943 | 0.6332 |
| Random Forest | 0.6988 | **0.6010** |
| **LightGBM** | **0.7057** | 0.6157 |

Berdasarkan hasil evaluasi pada **validation data**, **LightGBM** dipilih sebagai model terbaik karena menghasilkan **ROC-AUC** yang lebih tinggi dibandingkan Logistic Regression dan Random Forest. Meskipun Random Forest memperoleh nilai **Log Loss** yang sedikit lebih rendah, LightGBM memiliki kemampuan yang lebih baik dalam membedakan peminjam yang berisiko dan tidak berisiko. Oleh karena itu, model LightGBM dipilih sebagai model terbaik pada proyek ini.

## 🔍 Feature Importance Analysis

Analisis *feature importance* dilakukan untuk membandingkan faktor-faktor yang paling berpengaruh terhadap prediksi risiko gagal bayar pada ketiga model yang dikembangkan.

### Logistic Regression (Absolute Coefficient)

| Rank | Feature | Absolute Coefficient |
|------|---------|---------------------:|
| 1 | `int_rate` | 0.825222 |
| 2 | `sub_grade` | 0.334817 |
| 3 | `mths_since_issue_d` | 0.317693 |
| 4 | `funded_amnt_inv` | 0.235568 |
| 5 | `installment` | 0.154313 |
| 6 | `loan_income_ratio` | 0.113202 |
| 7 | `tot_coll_amt` | 0.108533 |
| 8 | `inq_last_6mths` | 0.108045 |
| 9 | `dti` | 0.097252 |
| 10 | `installment_income_ratio` | 0.089854 |

### Random Forest (Gini Importance)

| Rank | Feature | Gini Importance |
|------|---------|----------------:|
| 1 | `int_rate` | 0.237191 |
| 2 | `sub_grade` | 0.166193 |
| 3 | `mths_since_issue_d` | 0.109308 |
| 4 | `installment_income_ratio` | 0.043955 |
| 5 | `annual_inc` | 0.038263 |
| 6 | `loan_income_ratio` | 0.037947 |
| 7 | `tot_cur_bal` | 0.030194 |
| 8 | `avg_credit_per_account` | 0.028862 |
| 9 | `total_rev_hi_lim` | 0.027276 |
| 10 | `dti` | 0.025478 |

### LightGBM (Gain Importance)

| Rank | Feature | Gain Importance |
|------|---------|----------------:|
| 1 | `int_rate` | 319008 |
| 2 | `mths_since_issue_d` | 112308 |
| 3 | `annual_inc` | 38056 |
| 4 | `installment_income_ratio` | 21042 |
| 5 | `dti` | 18480.737125 |
| 6 | `mths_since_earliest_cr_line` | 13173 |
| 7 | `loan_income_ratio` | 13118 |
| 8 | `inq_last_6mths` | 11664 |
| 9 | `purpose` | 11588 |
| 10 | `sub_grade` | 10999 |

Hasil tersebut menunjukkan bahwa ketiga model menghasilkan pola yang relatif konsisten, di mana **tingkat suku bunga (`int_rate`)**, **tingkat risiko pinjaman (`sub_grade`)**, serta beberapa fitur hasil *feature engineering* seperti `loan_income_ratio` dan `installment_income_ratio` memberikan kontribusi yang signifikan dalam membedakan peminjam dengan risiko gagal bayar tinggi maupun rendah. Selain itu, fitur-fitur seperti **annual income**, **debt-to-income ratio (DTI)**, dan **riwayat kredit** juga menjadi faktor penting dalam proses prediksi pada beberapa model.

## 💡 Business Recommendations
- Mengintegrasikan model LightGBM ke dalam proses *credit scoring* sebagai alat bantu pengambilan keputusan pemberian pinjaman.
- Menggunakan probabilitas prediksi sebagai indikator tambahan dalam mengevaluasi pengajuan pinjaman yang memiliki risiko gagal bayar tinggi.
- Memprioritaskan kualitas dan validasi data pada fitur-fitur yang memiliki pengaruh besar terhadap hasil prediksi, terutama **interest rate**, **sub grade**, **annual income**, serta fitur-fitur hasil *feature engineering*.
- Melakukan *retraining* model secara berkala menggunakan data terbaru agar performa prediksi tetap optimal.
- Menyesuaikan strategi pemberian pinjaman, seperti limit kredit, tenor, atau kebijakan mitigasi risiko lainnya, berdasarkan tingkat risiko yang diprediksi model.
