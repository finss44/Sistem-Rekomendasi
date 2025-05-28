# Laporan Proyek Machine Learning - Elfin Darmawan

## Project Overview

Sistem rekomendasi telah menjadi bagian penting dalam berbagai platform digital untuk meningkatkan pengalaman pengguna. Proyek ini berfokus pada pengembangan sistem rekomendasi film dengan menggunakan dataset MovieLens Small Latest Dataset. Tujuan dari sistem ini adalah untuk menyarankan film yang relevan kepada pengguna berdasarkan preferensi dan riwayat rating mereka.

Dalam era digital saat ini, konsumen dihadapkan pada pilihan film yang sangat banyak, baik melalui platform streaming maupun layanan digital lainnya. Tanpa bantuan sistem rekomendasi, konsumen akan kesulitan menemukan film yang sesuai dengan preferensinya. Oleh karena itu, sistem rekomendasi menjadi solusi penting untuk membantu pengguna menemukan film yang relevan dan menarik.

### Mengapa Proyek Ini Penting?

- Membantu pengguna menemukan film yang sesuai selera mereka dari ribuan pilihan.
- Menyediakan pengalaman pengguna yang lebih baik dan personal.
- Menyarankan film-film berdasarkan preferensi pengguna

### Referensi:

Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook.

Movielens Dataset (GroupLens Research): https://grouplens.org/datasets/movielens/

## Business Understanding
### Problem Statements

Pengguna sering kali kesulitan memilih film yang sesuai dengan preferensi mereka karena banyaknya pilihan yang tersedia. Diperlukan sistem yang mampu merekomendasikan film secara akurat berdasarkan kemiripan isi film dan minat pengguna.

### Goals

Membangun sistem rekomendasi yang mampu memberikan Top-N rekomendasi film yang sesuai dengan minat pengguna secara otomatis dan personal.

### Solution statements

Proyek ini mengimplementasikan dua pendekatan:
- **Content-Based Filtering (TF-IDF + Cosine Similarity)**
Menggunakan informasi genre dari film untuk merekomendasikan film serupa berdasarkan kesukaan pengguna.
- **Collaborative Filtering (Neural Network-based Embedding Model)**
Menggunakan data rating untuk menemukan hubungan antar pengguna dan item untuk menampilkan rekomendasi film yang kemungkinan akan disukai oleh pengguna berdasarkan rating.


## Data Understanding

Dataset yang digunakan dalam proyek ini adalah MovieLens Small Latest Dataset, yang merupakan salah satu dataset benchmark populer untuk membangun dan mengevaluasi sistem rekomendasi. Dataset ini terdiri dari beberapa file utama, yaitu movies.csv, ratings.csv, links.csv, tags.csv. Tapi disini kita hanya menggunakan 2 dataset yaitu movies dan ratings.

Dataset: https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset

### Variabel-variabel pada `movies.csv` dataset adalah sebagai berikut:
**Terdiri dari `9.742 baris` dan `3 kolom`**

Berisi daftar film yang mencakup:
- `movieId`: ID unik untuk setiap film.
- `title`: Judul film beserta tahun rilis.
- `genres`: Daftar genre film yang dipisahkan oleh tanda pipe (|), seperti Action|Adventure|Sci-Fi.
       
### Variabel-variabel pada `ratings.csv` dataset adalah sebagai berikut:
**Terdiri dari `100.836 baris` dan `4 kolom`**

Berisi data penilaian yang diberikan pengguna terhadap film. Setiap entri menunjukkan interaksi pengguna dengan suatu film:
- `userId`: ID pengguna.
- `movieId`: ID film.
- `rating`: Nilai rating yang diberikan (skala 0.5 hingga 5.0).
- `timestamp`: Waktu rating diberikan (dalam UNIX time).

### Exploratory Data Analysis (EDA)
#### Movies Variabel
- Menampilkan 10 data awal dari dataframe movies menggunakan `.head(10)`
```
     movieId	        title	                                        genres
0	1	Toy Story (1995)	                Adventure|Animation|Children|Comedy|Fantasy
1	2	Jumanji (1995)	                        Adventure|Children|Fantasy
2	3	Grumpier Old Men (1995)	                Comedy|Romance
3	4	Waiting to Exhale (1995)	        Comedy|Drama|Romance
4	5	Father of the Bride Part II (1995)	Comedy
5	6	Heat (1995)	                        Action|Crime|Thriller
6	7	Sabrina (1995)	                        Comedy|Romance
7	8	Tom and Huck (1995)	                Adventure|Children
8	9	Sudden Death (1995)	                Action
9	10	GoldenEye (1995)	                Action|Adventure|Thriller
```
Tabel diatas merupakan isi dari dataset `movie.csv` dengan menampilkan 10 data awal dari dataset tersebut.

- Menampilkan dan mengecek informasi umum dataset menggunakan `.info`
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9742 entries, 0 to 9741
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   movieId  9742 non-null   int64 
 1   title    9742 non-null   object
 2   genres   9742 non-null   object
dtypes: int64(1), object(2)
memory usage: 228.5+ KB
```
Pada setiap kolom tertera 9742 non-null, yang menunjukan bahwa tidak ada nilai yang hilang atau mising value pada dataset. Kolom `movieId` memiliki tipe data integer, sedangkan `title` dan `genres` memiliki tipe data object.

- Melihat statistik deskriptif dari dataset menggunakan `.describe`
```bash
          movieId
count	9742.000000
mean	42200.353623
std	52160.494854
min	1.000000
25%	3248.250000
50%	7300.000000
75%	76232.000000
max	193609.000000
```
Karena dalam dataset `movies` hanya kolom `movieId` yang bersifat numerik, maka outputnya hanya menampilkan statistik untuk kolom tersebut. Terlihat nilai tengah, standar deviasi, minimal dan maximal untuk kolom `movieId`.

- Mengecek Missing Value dan data duplikat menggunakan `.isnull().sum()` dan `.duplicated().sum()`
```bash
        0
movieId	0
title	0
genres	0
dtype: int64 
```
Dari pengecekan tersebut terlihat bahwa tidak ada nilai null atau kosong dari dataframe movies

```bash
Jumlah data duplikat: 0
```
Mengecek data duplikat pada dataframe movies dan hasilnya adalah tidak ada data duplikat

- Distribusi Genre Terpopuler
[
## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
