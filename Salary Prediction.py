# Impor libraries yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Muat dataset
file_path = "Karyawan1.xlsx"
data = pd.read_excel(file_path)

# Hapus kolom yang tidak diperlukan jika ada
drop_columns = ['Unnamed: 10', 'Unnamed: 11', 'Z', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15']
data = data.drop(columns=[col for col in drop_columns if col in data.columns])

# Hapus kolom 'nomor' jika ada
if 'nomor' in data.columns:
    data = data.drop(columns=['nomor'])

# Periksa nilai yang hilang sebelum pemrosesan
print("Nilai yang hilang sebelum penanganan:\n", data.isnull().sum())

# Ubah kolom kategorikal menjadi numerik menggunakan one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Isi nilai yang hilang pada kolom numerik dengan rata-rata
data = data.fillna(data.mean())

# Pisahkan dataset menjadi fitur (X) dan variabel target (Y)
X = data.drop(columns=['gaji_per_bulan'])
Y = data['gaji_per_bulan']

# Pisahkan data menjadi data pelatihan dan data pengujian (pembagian 80-20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Periksa lagi untuk nilai yang hilang setelah pemisahan
print("Nilai yang hilang di X_train:", X_train.isnull().sum().sum())
print("Nilai yang hilang di X_test:", X_test.isnull().sum().sum())

# Hapus baris NaN yang tersisa (sebagai upaya terakhir)
X_train = X_train.dropna()
X_test = X_test.dropna()
Y_train = Y_train.loc[X_train.index]  # Sesuaikan Y_train dengan X_train
Y_test = Y_test.loc[X_test.index]     # Sesuaikan Y_test dengan X_test

# Buat dan latih model regresi linier
model = LinearRegression()
model.fit(X_train, Y_train)

# Prediksi data pengujian
Y_pred = model.predict(X_test)

# Hitung Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_pred)
print('Mean Squared Error (MSE):', mse)

# Hitung skor R^2
r2 = r2_score(Y_test, Y_pred)
print('Skor R^2:', r2)

# Visualisasi: Gaji Asli vs Gaji Prediksi
plt.figure(figsize=(12, 6))
sns.regplot(x=Y_test, y=Y_pred, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel('Gaji Asli')
plt.ylabel('Gaji Prediksi')
plt.title('Gaji Asli vs Gaji Prediksi')
plt.show()