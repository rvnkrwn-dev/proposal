# BAB III
# ALUR PEMODELAN

## 3.1 Alur Pemodelan

Penelitian ini menggunakan pendekatan metodologi machine learning untuk menganalisis kinerja algoritma Support Vector Machine (SVM) dalam klasifikasi sentiment ulasan produk skincare Whitelab. Alur pemodelan yang dirancang mengikuti standar proses Knowledge Discovery in Databases (KDD) yang telah diadaptasi untuk kebutuhan analisis sentiment pada domain produk skincare. 

Metodologi yang digunakan dalam penelitian ini meliputi serangkaian tahapan sistematis mulai dari pengumpulan data hingga evaluasi model. Setiap tahapan dirancang untuk memastikan kualitas data yang optimal dan hasil klasifikasi yang akurat. Pendekatan ini mengintegrasikan teknik text preprocessing khusus untuk bahasa Indonesia dengan optimasi parameter SVM untuk mencapai performa klasifikasi yang maksimal.

Keseluruhan proses pemodelan ini didesain untuk menjawab permasalahan penelitian terkait kinerja SVM dalam mengklasifikasikan sentiment ulasan konsumen terhadap produk skincare Whitelab, serta menentukan tingkat akurasi yang dapat dicapai oleh metode tersebut.

## 3.2 Diagram Alur Pemodelan

Diagram alur pemodelan penelitian ini menggambarkan secara visual tahapan-tahapan yang dilakukan dalam proses klasifikasi sentiment menggunakan Support Vector Machine. Setiap komponen dalam diagram menrepresentasikan proses spesifik yang saling terhubung dan berkontribusi terhadap hasil akhir model klasifikasi.

### Gambar 3.1 Diagram Alur Pemodelan

```
┌─────────────┐
│    Start    │
└─────┬───────┘
      │
      ▼
┌─────────────────┐
│ Pengumpulan Data│
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│ Dataset TikTok  │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│Preprocessing Data│
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│ Labeling Data   │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│   Split Data    │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│   Modeling      │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│ Evaluasi Model  │
└─────┬───────────┘
      │
      ▼
┌─────────────┐
│     End     │
└─────────────┘
```

## 3.3 Penjelasan Tahapan

### 3.3.1 Pengumpulan Data

Tahap pengumpulan data merupakan fondasi dari seluruh proses penelitian. Data ulasan produk skincare Whitelab dikumpulkan dari platform media sosial TikTok menggunakan teknik web scraping dengan pendekatan API reverse engineering. Pemilihan TikTok sebagai sumber data didasarkan pada popularitas platform ini di kalangan konsumen produk skincare Indonesia, terutama generasi muda yang aktif memberikan review produk kecantikan.

**Metodologi Pengumpulan Data:**

Proses pengumpulan data dilakukan menggunakan pendekatan reverse engineering terhadap TikTok API dengan langkah-langkah sebagai berikut:

1. **Browser Authentication**
   - Login ke akun TikTok melalui web browser untuk mendapatkan session cookies dan authentication tokens
   - Memastikan akses legitimate ke konten publik sesuai Terms of Service platform

2. **Network Traffic Analysis**
   - Menggunakan Developer Tools (F12) pada browser untuk menganalisis network traffic
   - Navigasi ke bagian "Network" pada Developer Tools untuk monitoring HTTP requests

3. **API Endpoint Discovery**
   - Melakukan inspect element pada video yang mengandung review Whitelab
   - Monitoring network requests saat memuat komentar untuk mengidentifikasi API endpoint
   - Mencari URL fetch data comments yang digunakan oleh TikTok frontend untuk mengambil data komentar

4. **Request Replication**
   - Menyalin URL API endpoint beserta headers dan parameters yang diperlukan
   - Menggunakan Postman untuk mereplikasi request dan memvalidasi response data
   - Menganalisis struktur JSON response untuk memahami format data komentar

5. **Data Extraction Implementation**
   - Mengimplementasikan script otomatis untuk melakukan batch requests ke API endpoint
   - Menggunakan parameter pagination untuk mengumpulkan semua komentar dari video target
   - Menerapkan rate limiting untuk menghindari blocking dari server TikTok

**Target dan Kriteria Data:**

Target pengumpulan data adalah minimal 1000 ulasan yang mencakup berbagai variasi sentiment dan karakteristik linguistik konsumen Indonesia. Data yang dikumpulkan meliputi:

- **Teks komentar**: Konten utama yang mengandung opini tentang produk Whitelab
- **Metadata temporal**: Timestamp posting untuk analisis trend sentiment
- **Engagement metrics**: Jumlah like dan reply sebagai indikator kualitas komentar
- **User information**: Data anonim untuk memastikan diversitas sumber

**Kriteria Seleksi Data:**

Kriteria seleksi data mencakup ulasan yang mengandung keyword terkait produk Whitelab, ditulis dalam bahasa Indonesia, dan memiliki substansi review yang memadai (minimal 10 kata). Proses ini juga mempertimbangkan aspek etika penelitian dengan memastikan anonimitas pengguna dan kepatuhan terhadap terms of service platform.

**Aspek Teknis dan Etis:**

- **Compliance**: Mengikuti robots.txt dan rate limiting untuk menghormati server resources
- **Data Privacy**: Menganonimkan identitas user dan menghapus informasi personal
- **Reproducibility**: Mendokumentasikan API endpoints dan parameters untuk replikasi penelitian
- **Data Quality**: Implementasi filter untuk menghilangkan spam, bot comments, dan duplikasi

### 3.3.2 Preprocessing Data

Tahap preprocessing data merupakan proses transformasi data teks mentah menjadi format yang dapat dipahami dan diproses oleh algoritma machine learning. Tahapan ini terdiri dari beberapa sub-proses yang dilakukan secara sequential untuk memastikan kualitas data yang optimal.

### Gambar 3.2 Diagram Alur Preprocessing Data

```
┌─────────────────────┐
│    Raw Text Data    │
│ "Skincare WHITELAB  │
│  ini BAGUSSS bgt    │
│  recomended 4 all!" │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      Cleaning       │
│ Remove: URL, @user, │
│ hashtag, emoticon   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Case Folding     │
│ "skincare whitelab  │
│  ini bagusss bgt    │
│  recomended 4 all"  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Normalization     │
│ "skincare whitelab  │
│  ini bagus banget   │
│  recommended for    │
│       all"          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Tokenizing       │
│ ["skincare",        │
│  "whitelab", "ini", │
│  "bagus", "banget", │
│  "recommended",     │
│  "for", "all"]      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Stopword Removal   │
│ ["skincare",        │
│  "whitelab",        │
│  "bagus", "banget", │
│  "recommended"]     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Stemming        │
│ ["skincare",        │
│  "whitelab",        │
│  "bagus", "banget", │
│  "recommend"]       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Processed Tokens   │
│   Ready for TF-IDF  │
│   Vectorization     │
└─────────────────────┘
```

#### 3.3.2.1 Cleaning

Proses cleaning bertujuan untuk menghilangkan elemen-elemen yang tidak relevan dalam teks ulasan. Tahapan ini meliputi penghapusan URL, mention (@username), hashtag yang tidak informatif, emoticon, dan karakter khusus yang tidak memiliki nilai semantic. Selain itu, dilakukan penghapusan duplikasi data untuk memastikan setiap ulasan unik dan tidak bias terhadap opinion tertentu.

Cleaning juga mencakup penanganan teks yang mengandung bahasa campuran (code-mixing) antara bahasa Indonesia dan Inggris, yang umum dijumpai dalam ulasan produk skincare. Proses ini menggunakan regular expression dan rule-based filtering untuk mempertahankan konten yang informatif sambil menghilangkan noise yang dapat mengganggu proses klasifikasi.

**Tabel 3.1 Contoh Proses Cleaning Data Komentar TikTok**

| No | Data Sebelum Cleaning | Data Sesudah Cleaning |
|----|----------------------|----------------------|
| 1 | @cantikbanget wah whitelab emang TOP!!! 😍✨ recommend bgt sih 💯 | wah whitelab emang TOP recommend bgt sih |
| 2 | Skincare whitelab ini bagusss 🔥🔥 cek bio ya guys https://shopee.co.id/whitlabid | Skincare whitelab ini bagusss cek bio ya guys |
| 3 | #skincarereview #whitelab udah pake 2 minggu hasilnya WOW!! 💆‍♀️👌 | udah pake 2 minggu hasilnya WOW |
| 4 | Guys @whitlabid emang juara!!! Jerawatku ilang 😭💕 thank u whitelab | Guys emang juara Jerawatku ilang thank u whitelab |
| 5 | OMG whitelab vitamin C serum >>> other brand 🤩💛 worth it bgt! | OMG whitelab vitamin C serum other brand worth it bgt |
| 6 | @followerssku liat nih review whitelab aku... AMAZINGGG!!! 🤯✨💯 | liat nih review whitelab aku AMAZINGGG |
| 7 | Whitelab toner = love 💕💕💕 repurchase terusss #skincare #beautytips | Whitelab toner love repurchase terusss |
| 8 | Gais skincare whitelab ini 10/10 ⭐⭐⭐⭐⭐ no joke! beneran bagus | Gais skincare whitelab ini 10 10 no joke beneran bagus |
| 9 | @beautyinfluencer review whitelab dong! aku udah coba 😍 approved! | review whitelab dong aku udah coba approved |
| 10 | Link di bio ya! whitelabid.com/promo DISKON 50%!!! 🛒💸 buruan! | whitelabid promo DISKON 50 buruan |

**Elemen yang Dihapus dalam Proses Cleaning:**
- **Mention (@username)**: @cantikbanget, @whitlabid, @followerssku, @beautyinfluencer
- **Hashtag (#tag)**: #skincarereview, #whitelab, #skincare, #beautytips  
- **Emoticon dan Symbol**: 😍, ✨, 💯, 🔥, 💆‍♀️, 👌, 😭, 💕, 🤩, 💛, 🤯, ⭐, 🛒, 💸
- **URL dan Link**: https://shopee.co.id/whitlabid, whitelabid.com/promo
- **Symbol Khusus**: >>>, /, %
- **Karakter Berulang Berlebihan**: !!! (dikurangi menjadi satu), 🔥🔥 (dihapus)
- **Angka dalam Konteks Rating**: 10/10 (diubah menjadi "10 10")

Proses cleaning ini menggunakan regular expression pattern yang telah disesuaikan dengan karakteristik komentar TikTok Indonesia, sambil mempertahankan konteks sentiment yang penting untuk klasifikasi.

#### 3.3.2.2 Case Folding

Case folding adalah proses konversi seluruh teks menjadi huruf kecil (lowercase) untuk memastikan konsistensi dalam pemrosesan. Tahapan ini penting karena algoritma machine learning bersifat case-sensitive, sehingga kata "Bagus", "bagus", dan "BAGUS" akan dianggap sebagai token yang berbeda jika tidak dilakukan normalisasi.

**Tabel 3.2 Contoh Proses Case Folding**

| No | Data Sesudah Cleaning | Data Sesudah Case Folding |
|----|----------------------|---------------------------|
| 1 | wah whitelab emang TOP recommend bgt sih | wah whitelab emang top recommend bgt sih |
| 2 | Skincare whitelab ini bagusss cek bio ya guys | skincare whitelab ini bagusss cek bio ya guys |
| 3 | udah pake 2 minggu hasilnya WOW | udah pake 2 minggu hasilnya wow |
| 4 | Guys emang juara Jerawatku ilang thank u whitelab | guys emang juara jerawatku ilang thank u whitelab |
| 5 | OMG whitelab vitamin C serum other brand worth it bgt | omg whitelab vitamin c serum other brand worth it bgt |

#### 3.3.2.3 Normalization

Normalization bertujuan untuk menstandarkan variasi penulisan kata yang memiliki makna sama. Dalam konteks ulasan media sosial Indonesia, sering dijumpai variasi penulisan seperti "banget" vs "bgt", "jerawat" vs "jerawat", atau penggunaan angka sebagai pengganti huruf seperti "4" untuk "a".

**Tabel 3.3 Contoh Proses Normalization**

| No | Data Sesudah Case Folding | Data Sesudah Normalization |
|----|---------------------------|----------------------------|
| 1 | wah whitelab emang top recommend bgt sih | wah whitelab emang top recommend banget sih |
| 2 | skincare whitelab ini bagusss cek bio ya guys | skincare whitelab ini bagus cek bio ya guys |
| 3 | udah pake 2 minggu hasilnya wow | udah pakai 2 minggu hasilnya wow |
| 4 | guys emang juara jerawatku ilang thank u whitelab | guys emang juara jerawatku hilang thank you whitelab |
| 5 | omg whitelab vitamin c serum other brand worth it bgt | oh my god whitelab vitamin c serum other brand worth it banget |

**Normalisasi yang Dilakukan:**
- bgt → banget
- bagusss → bagus  
- pake → pakai
- ilang → hilang
- thank u → thank you
- omg → oh my god

#### 3.3.2.4 Tokenizing

Tokenizing adalah proses memecah teks menjadi unit-unit yang lebih kecil seperti kata atau phrase. Dalam konteks bahasa Indonesia, tokenisasi menghadapi tantangan khusus seperti aglutinasi (penggabungan morfem) dan ambiguitas batas kata dalam kalimat tanpa spasi yang jelas.

**Tabel 3.4 Contoh Proses Tokenizing**

| No | Data Sesudah Normalization | Data Sesudah Tokenizing |
|----|----------------------------|-------------------------|
| 1 | wah whitelab emang top recommend banget sih | ['wah', 'whitelab', 'emang', 'top', 'recommend', 'banget', 'sih'] |
| 2 | skincare whitelab ini bagus cek bio ya guys | ['skincare', 'whitelab', 'ini', 'bagus', 'cek', 'bio', 'ya', 'guys'] |
| 3 | udah pakai 2 minggu hasilnya wow | ['udah', 'pakai', '2', 'minggu', 'hasilnya', 'wow'] |
| 4 | guys emang juara jerawatku hilang thank you whitelab | ['guys', 'emang', 'juara', 'jerawatku', 'hilang', 'thank', 'you', 'whitelab'] |
| 5 | oh my god whitelab vitamin c serum other brand worth it banget | ['oh', 'my', 'god', 'whitelab', 'vitamin', 'c', 'serum', 'other', 'brand', 'worth', 'it', 'banget'] |

#### 3.3.2.5 Stopword Removal

Stopword removal bertujuan untuk menghilangkan kata-kata yang memiliki frekuensi tinggi namun tidak memberikan informasi yang signifikan untuk klasifikasi sentiment. Daftar stopword yang digunakan mencakup kata-kata umum bahasa Indonesia seperti "dan", "atau", "di", "ke", "dari", serta kata-kata yang spesifik untuk domain ulasan produk.

**Tabel 3.5 Contoh Proses Stopword Removal**

| No | Data Sesudah Tokenizing | Data Sesudah Stopword Removal |
|----|-------------------------|--------------------------------|
| 1 | ['wah', 'whitelab', 'emang', 'top', 'recommend', 'banget', 'sih'] | ['wah', 'whitelab', 'top', 'recommend', 'banget'] |
| 2 | ['skincare', 'whitelab', 'ini', 'bagus', 'cek', 'bio', 'ya', 'guys'] | ['skincare', 'whitelab', 'bagus', 'cek', 'bio', 'guys'] |
| 3 | ['udah', 'pakai', '2', 'minggu', 'hasilnya', 'wow'] | ['pakai', '2', 'minggu', 'hasilnya', 'wow'] |
| 4 | ['guys', 'emang', 'juara', 'jerawatku', 'hilang', 'thank', 'you', 'whitelab'] | ['guys', 'juara', 'jerawatku', 'hilang', 'thank', 'whitelab'] |
| 5 | ['oh', 'my', 'god', 'whitelab', 'vitamin', 'c', 'serum', 'other', 'brand', 'worth', 'it', 'banget'] | ['whitelab', 'vitamin', 'c', 'serum', 'brand', 'worth', 'banget'] |

**Stopwords yang Dihapus:**
- emang, sih (kata penekanan)
- ini, ya (kata penunjuk)
- udah (kata keterangan waktu)
- emang (kata penegas)
- oh, my, god, it, other (kata umum bahasa Inggris)

#### 3.3.2.6 Stemming

Stemming adalah proses mengubah kata berimbuhan menjadi kata dasar untuk mengurangi variasi morfologis dan meningkatkan konsistensi feature. Dalam bahasa Indonesia, stemming menghadapi kompleksitas tinggi karena sistem imbuhan yang kaya dan aturan morfologi yang kompleks.

**Tabel 3.6 Contoh Proses Stemming**

| No | Data Sesudah Stopword Removal | Data Sesudah Stemming |
|----|---------------------------------|----------------------|
| 1 | ['wah', 'whitelab', 'top', 'recommend', 'banget'] | ['wah', 'whitelab', 'top', 'recommend', 'banget'] |
| 2 | ['skincare', 'whitelab', 'bagus', 'cek', 'bio', 'guys'] | ['skincare', 'whitelab', 'bagus', 'cek', 'bio', 'guys'] |
| 3 | ['pakai', '2', 'minggu', 'hasilnya', 'wow'] | ['pakai', '2', 'minggu', 'hasil', 'wow'] |
| 4 | ['guys', 'juara', 'jerawatku', 'hilang', 'thank', 'whitelab'] | ['guys', 'juara', 'jerawat', 'hilang', 'thank', 'whitelab'] |
| 5 | ['whitelab', 'vitamin', 'c', 'serum', 'brand', 'worth', 'banget'] | ['whitelab', 'vitamin', 'c', 'serum', 'brand', 'worth', 'banget'] |

**Stemming yang Dilakukan:**
- hasilnya → hasil (menghilangkan imbuhan -nya)
- jerawatku → jerawat (menghilangkan imbuhan -ku)

**Catatan:** Beberapa kata seperti 'whitelab', 'skincare', 'vitamin', 'serum' tidak di-stem karena merupakan kata khusus dalam domain skincare atau nama brand yang perlu dipertahankan untuk analisis sentiment.

### 3.3.3 Labeling Data

Proses labeling merupakan tahapan krusial dalam supervised learning untuk memberikan ground truth pada data training. Dalam penelitian ini, setiap ulasan diklasifikasikan ke dalam tiga kategori sentiment: positif, negatif, dan netral. Proses labeling dilakukan menggunakan pendekatan hybrid yang menggabungkan lexicon-based approach dengan manual annotation.

Tahap pertama menggunakan sentiment lexicon bahasa Indonesia yang telah dikembangkan untuk domain produk kecantikan. Lexicon ini berisi kata-kata dan phrase yang memiliki polaritas sentiment tertentu, seperti "bagus", "jelek", "recommended", "mengecewakan". Setiap ulasan kemudian diberi skor sentiment berdasarkan agregasi skor kata-kata yang terkandung di dalamnya.

**Formula Perhitungan Sentiment Score:**

```
Sentiment Score = (∑ Positive Words Score - ∑ Negative Words Score) / Total Words
```

**Tabel 3.7 Contoh Proses Labeling Data**

| No | Data Sesudah Preprocessing | Positive Words | Negative Words | Sentiment Score | Label |
|----|---------------------------|----------------|----------------|-----------------|-------|
| 1 | ['wah', 'whitelab', 'top', 'recommend', 'banget'] | top(+2), recommend(+3), banget(+1) | - | (6-0)/5 = +1.2 | Positif |
| 2 | ['skincare', 'whitelab', 'bagus', 'cek', 'bio', 'guys'] | bagus(+2) | - | (2-0)/6 = +0.33 | Positif |
| 3 | ['pakai', '2', 'minggu', 'hasil', 'wow'] | wow(+2) | - | (2-0)/5 = +0.4 | Positif |
| 4 | ['guys', 'juara', 'jerawat', 'hilang', 'thank', 'whitelab'] | juara(+2), hilang(+1) | jerawat(-1) | (3-1)/6 = +0.33 | Positif |
| 5 | ['whitelab', 'vitamin', 'c', 'serum', 'brand', 'worth', 'banget'] | worth(+1), banget(+1) | - | (2-0)/7 = +0.29 | Positif |
| 6 | ['whitelab', 'serum', 'jelek', 'tidak', 'cocok'] | - | jelek(-3), tidak(-1), cocok(-1) | (0-5)/5 = -1.0 | Negatif |
| 7 | ['whitelab', 'biasa', 'saja', 'standar'] | - | biasa(-1) | (0-1)/4 = -0.25 | Netral |

**Kriteria Labeling:**
- **Positif**: Sentiment Score > +0.2
- **Netral**: -0.2 ≤ Sentiment Score ≤ +0.2  
- **Negatif**: Sentiment Score < -0.2

Tahap kedua melibatkan manual annotation oleh annotator yang memiliki pemahaman domain skincare untuk memvalidasi dan memperbaiki hasil automatic labeling. Proses ini melibatkan minimal dua annotator dengan inter-annotator agreement yang diukur menggunakan Cohen's Kappa untuk memastikan konsistensi labeling.

**Formula Cohen's Kappa:**

```
κ = (Po - Pe) / (1 - Pe)

dimana:
Po = Observed agreement (proporsi kesepakatan yang diamati)
Pe = Expected agreement (proporsi kesepakatan yang diharapkan secara kebetulan)
```

### 3.3.4 Pembagian Data

Pembagian data dilakukan untuk memisahkan dataset menjadi training set dan testing set dengan proporsi yang optimal untuk menghindari overfitting dan memastikan generalisasi model yang baik. Penelitian ini menggunakan split ratio 80:20, dimana 80% data digunakan untuk training dan 20% untuk testing.

**Formula Pembagian Data:**

```
Training Set Size = Total Data × 0.8
Testing Set Size = Total Data × 0.2
Validation Set Size = Training Set × 0.2 (untuk hyperparameter tuning)
```

**Tabel 3.8 Distribusi Pembagian Data**

| Dataset | Jumlah Data | Positif | Negatif | Netral | Persentase |
|---------|-------------|---------|---------|--------|------------|
| **Total Dataset** | 1000 | 450 | 350 | 200 | 100% |
| **Training Set** | 800 | 360 | 280 | 160 | 80% |
| **Testing Set** | 200 | 90 | 70 | 40 | 20% |
| **Validation Set** | 160 | 72 | 56 | 32 | 16% (dari total) |
| **Final Training** | 640 | 288 | 224 | 128 | 64% (dari total) |

**Tabel 3.9 Contoh Pembagian Data per Kelas**

| ID | Processed Text | Label | Dataset |
|----|----------------|-------|---------|
| 001 | ['wah', 'whitelab', 'top', 'recommend', 'banget'] | Positif | Training |
| 002 | ['skincare', 'whitelab', 'bagus', 'cek', 'bio'] | Positif | Training |
| 003 | ['whitelab', 'serum', 'jelek', 'tidak', 'cocok'] | Negatif | Testing |
| 004 | ['whitelab', 'biasa', 'saja', 'standar'] | Netral | Validation |
| 005 | ['pakai', 'minggu', 'hasil', 'wow'] | Positif | Training |
| ... | ... | ... | ... |

Proses pembagian data menggunakan stratified sampling untuk memastikan distribusi kelas sentiment yang proporsional dalam kedua subset. Hal ini penting untuk mencegah bias klasifikasi yang dapat terjadi jika salah satu kelas sentiment dominan dalam training set.

**Formula Stratified Sampling:**

```
Proporsi Kelas i dalam Training = (Jumlah Kelas i dalam Dataset) / (Total Dataset) × Training Size

Contoh untuk kelas Positif:
Positif dalam Training = (450/1000) × 800 = 360 data
```

### 3.3.5 Pemodelan Support Vector Machine

Tahap pemodelan merupakan inti dari penelitian ini, dimana algoritma Support Vector Machine diimplementasikan dan dioptimalkan untuk klasifikasi sentiment ulasan skincare. Proses ini meliputi feature extraction, model training, dan hyperparameter optimization untuk mencapai performa klasifikasi yang optimal.

#### Feature Extraction

Feature extraction menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengubah teks menjadi representasi numerik yang dapat diproses oleh SVM. Implementasi TF-IDF menggunakan parameter yang dioptimalkan, meliputi n-gram range (1,2) untuk menangkap konteks kata, minimum document frequency untuk menghilangkan term yang terlalu jarang, dan maximum features untuk membatasi dimensi feature space.

**Formula TF-IDF:**

```
TF-IDF(t,d) = TF(t,d) × IDF(t)

dimana:
TF(t,d) = (Jumlah kemunculan term t dalam dokumen d) / (Total term dalam dokumen d)
IDF(t) = log(N / df(t))

N = Total jumlah dokumen
df(t) = Jumlah dokumen yang mengandung term t
```

**Tabel 3.10 Contoh Perhitungan TF-IDF**

| Term | Doc1 Freq | Doc2 Freq | Doc3 Freq | TF(Doc1) | IDF | TF-IDF(Doc1) |
|------|-----------|-----------|-----------|----------|-----|--------------|
| whitelab | 1 | 1 | 1 | 1/5 = 0.2 | log(3/3) = 0 | 0.2 × 0 = 0 |
| bagus | 1 | 0 | 0 | 1/5 = 0.2 | log(3/1) = 0.477 | 0.2 × 0.477 = 0.095 |
| recommend | 1 | 1 | 0 | 1/5 = 0.2 | log(3/2) = 0.176 | 0.2 × 0.176 = 0.035 |
| top | 1 | 0 | 0 | 1/5 = 0.2 | log(3/1) = 0.477 | 0.2 × 0.477 = 0.095 |

**Contoh Data:**
- Doc1: ['wah', 'whitelab', 'bagus', 'recommend', 'top'] 
- Doc2: ['whitelab', 'vitamin', 'c', 'serum', 'recommend']
- Doc3: ['whitelab', 'biasa', 'saja', 'standar', 'aja']

Proses feature extraction juga mengintegrasikan feature tambahan yang spesifik untuk domain skincare, seperti sentiment score dari lexicon, jumlah kata positif/negatif, dan feature linguistik seperti penggunaan kapitalisasi atau tanda baca emosional.

#### Model Training

Training model SVM dilakukan menggunakan implementasi dari library scikit-learn dengan optimasi untuk handling data teks yang sparse dan high-dimensional. Proses training meliputi eksperimen dengan berbagai konfigurasi kernel SVM, termasuk linear kernel, RBF kernel, dan polynomial kernel untuk menentukan yang paling sesuai dengan karakteristik data ulasan skincare.

**Formula Matematis SVM:**

**Fungsi Keputusan:**
```
f(x) = sign(∑i αi yi K(xi,x) + b)

dimana:
αi = koefisien Lagrange
yi = label kelas (+1 atau -1)  
K(xi,x) = fungsi kernel
b = bias
```

**Fungsi Objektif Optimasi:**
```
min ½||w||² + C ∑i ξi

dengan kendala:
yi(w·φ(xi) + b) ≥ 1 - ξi, ξi ≥ 0
```

**Kernel Functions:**

1. **Linear Kernel:**
   ```
   K(xi, xj) = xi · xj
   ```

2. **RBF (Radial Basis Function) Kernel:**
   ```
   K(xi, xj) = exp(-γ||xi - xj||²)
   ```

3. **Polynomial Kernel:**
   ```
   K(xi, xj) = (γxi · xj + r)^d
   ```

**Tabel 3.11 Contoh Parameter SVM yang Diuji**

| Kernel | Parameter C | Parameter γ | Parameter d | Akurasi Validasi |
|--------|-------------|-------------|-------------|------------------|
| Linear | 0.1 | - | - | 0.847 |
| Linear | 1.0 | - | - | 0.862 |
| Linear | 10.0 | - | - | 0.851 |
| RBF | 1.0 | 0.001 | - | 0.823 |
| RBF | 1.0 | 0.01 | - | 0.856 |
| RBF | 10.0 | 0.1 | - | 0.874 |
| Poly | 1.0 | 0.01 | 2 | 0.834 |
| Poly | 1.0 | 0.01 | 3 | 0.841 |

Training process juga mencakup implementasi teknik regularization dan class balancing untuk menangani potensi imbalanced dataset. Parameter regularization (C) dioptimalkan untuk mencapai trade-off optimal antara margin maksimization dan error minimization, sementara class balancing dilakukan menggunakan weighted SVM atau sampling techniques sesuai kebutuhan.

#### Hyperparameter Optimization

Optimasi hyperparameter dilakukan menggunakan Grid Search dengan 5-fold cross-validation untuk mencari kombinasi parameter optimal. Parameter yang dioptimalkan meliputi regularization parameter (C), kernel parameter (gamma untuk RBF kernel), serta parameter feature extraction seperti TF-IDF configuration.

Proses optimasi menggunakan stratified k-fold cross-validation untuk memastikan robustness evaluasi dan menghindari overfitting. Evaluasi dilakukan menggunakan multiple metrics (accuracy, precision, recall, F1-score) untuk memastikan optimasi yang comprehensive dan seimbang across different performance aspects.

### 3.3.6 Evaluasi Model

Tahap evaluasi merupakan langkah final untuk mengukur dan menganalisis kinerja model SVM yang telah dibangun. Evaluasi dilakukan menggunakan multiple metrics untuk memberikan gambaran comprehensive tentang performa klasifikasi dari berbagai perspektif.

#### 3.3.6.1 Accuracy

Accuracy mengukur proporsi prediksi yang benar dari total prediksi yang dibuat oleh model. Metrik ini memberikan gambaran umum tentang seberapa baik model dalam melakukan klasifikasi secara keseluruhan.

**Formula Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

atau untuk multi-class:
Accuracy = (Jumlah Prediksi Benar) / (Total Prediksi)
```

**Tabel 3.12 Contoh Perhitungan Accuracy**

| Kelas Aktual | Kelas Prediksi | Hasil |
|--------------|----------------|-------|
| Positif | Positif | Benar (TP) |
| Positif | Negatif | Salah (FN) |
| Negatif | Negatif | Benar (TN) |
| Negatif | Positif | Salah (FP) |
| Netral | Netral | Benar (TN) |

**Contoh Perhitungan:**
- Total Data Testing: 200
- Prediksi Benar: 174
- Accuracy = 174/200 = 0.87 = 87%

#### 3.3.6.2 Precision

Precision mengukur proporsi prediksi positif yang benar dari seluruh prediksi positif yang dibuat model. Metrik ini penting untuk mengetahui seberapa akurat model ketika memprediksi sentiment positif.

**Formula Precision:**
```
Precision = TP / (TP + FP)

atau untuk multi-class:
Precision_kelas_i = TP_i / (TP_i + FP_i)
```

**Tabel 3.13 Contoh Perhitungan Precision per Kelas**

| Kelas | TP | FP | Precision | Interpretasi |
|-------|----|----|-----------|--------------|
| Positif | 78 | 12 | 78/(78+12) = 0.867 | 86.7% prediksi positif benar |
| Negatif | 65 | 8 | 65/(65+8) = 0.890 | 89.0% prediksi negatif benar |
| Netral | 31 | 6 | 31/(31+6) = 0.838 | 83.8% prediksi netral benar |

#### 3.3.6.3 Recall

Recall mengukur proporsi instance positif yang berhasil diidentifikasi dengan benar oleh model. Metrik ini menunjukkan kemampuan model dalam mendeteksi seluruh instance dari kelas tertentu.

**Formula Recall:**
```
Recall = TP / (TP + FN)

atau untuk multi-class:
Recall_kelas_i = TP_i / (TP_i + FN_i)
```

**Tabel 3.14 Contoh Perhitungan Recall per Kelas**

| Kelas | TP | FN | Recall | Interpretasi |
|-------|----|----|--------|--------------|
| Positif | 78 | 12 | 78/(78+12) = 0.867 | 86.7% data positif terdeteksi |
| Negatif | 65 | 5 | 65/(65+5) = 0.929 | 92.9% data negatif terdeteksi |
| Netral | 31 | 9 | 31/(31+9) = 0.775 | 77.5% data netral terdeteksi |

#### 3.3.6.4 F1-Score

F1-Score merupakan harmonic mean dari precision dan recall yang memberikan balance measure antara kedua metrik tersebut. F1-Score sangat berguna ketika terdapat trade-off antara precision dan recall, atau ketika dataset memiliki distribusi kelas yang tidak seimbang.

**Formula F1-Score:**
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

atau untuk multi-class:
F1-Score_kelas_i = 2 × (Precision_i × Recall_i) / (Precision_i + Recall_i)
```

**Tabel 3.15 Contoh Perhitungan F1-Score per Kelas**

| Kelas | Precision | Recall | F1-Score | Perhitungan |
|-------|-----------|--------|----------|-------------|
| Positif | 0.867 | 0.867 | 0.867 | 2×(0.867×0.867)/(0.867+0.867) |
| Negatif | 0.890 | 0.929 | 0.909 | 2×(0.890×0.929)/(0.890+0.929) |
| Netral | 0.838 | 0.775 | 0.805 | 2×(0.838×0.775)/(0.838+0.775) |
| **Macro Avg** | 0.865 | 0.857 | 0.860 | Rata-rata dari semua kelas |

#### 3.3.6.5 Confusion Matrix

Confusion Matrix adalah tabel yang menampilkan performa klasifikasi dengan detail untuk setiap kelas. Matrix ini menunjukkan jumlah prediksi yang benar dan salah untuk setiap kombinasi kelas aktual dan prediksi.

**Tabel 3.16 Contoh Confusion Matrix 3x3**

|  | **Prediksi** | | | |
|--|--|--|--|--|
| **Aktual** | | Positif | Negatif | Netral |
| | **Positif** | 78 | 8 | 4 |
| | **Negatif** | 12 | 65 | 3 |
| | **Netral** | 6 | 5 | 31 |

**Interpretasi Confusion Matrix:**
- **Diagonal utama** (78, 65, 31): Prediksi yang benar
- **Off-diagonal**: Prediksi yang salah
- **Baris**: Kelas aktual (ground truth)
- **Kolom**: Kelas prediksi (model output)

**Tabel 3.17 Analisis Error dari Confusion Matrix**

| Jenis Error | Jumlah | Persentase | Interpretasi |
|-------------|--------|------------|--------------|
| Positif → Negatif | 8 | 4.0% | Model salah klasifikasi positif sebagai negatif |
| Positif → Netral | 4 | 2.0% | Model salah klasifikasi positif sebagai netral |
| Negatif → Positif | 12 | 6.0% | Model salah klasifikasi negatif sebagai positif |
| Negatif → Netral | 3 | 1.5% | Model salah klasifikasi negatif sebagai netral |
| Netral → Positif | 6 | 3.0% | Model salah klasifikasi netral sebagai positif |
| Netral → Negatif | 5 | 2.5% | Model salah klasifikasi netral sebagai negatif |

Analisis confusion matrix juga memungkinkan perhitungan metrik evaluasi per-class, seperti precision, recall, dan F1-score untuk masing-masing kelas sentiment. Hal ini penting untuk memahami performa model pada setiap kategori sentiment dan mengidentifikasi potential bias atau weakness dalam klasifikasi tertentu.
