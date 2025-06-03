# Price Comparison

## Deskripsi Proyek
Price Comparison adalah sebuah aplikasi yang memungkinkan pengguna untuk membandingkan harga suatu produk di platform e-commerce seperti Tokopedia. Aplikasi ini mendukung input baik dalam bentuk gambar maupun teks, melakukan klasifikasi produk, mencari produk serupa di e-commerce, dan memberikan hasil berupa produk dengan harga termurah dan terjual terbanyak beserta link pembelian.

## Fitur Utama
- **Klasifikasi Produk**: Mendukung input dalam bentuk gambar (image classification) maupun teks.
- **Pencarian di Marketplace**: Mencari produk yang sesuai di Tokopedia.
- **Perbandingan Harga dan Jumlah Terjual**: Mengidentifikasi produk dengan harga paling murah dan jumlah terjual terbanyak.
- **Penyimpanan Data**: Menyimpan hasil pencarian produk dalam bentuk csv untuk referensi di masa mendatang.

## Cara Kerja
1. Pengguna mengunggah gambar atau memasukkan teks produk yang ingin dicari.
2. Sistem mengklasifikasikan produk untuk mendapatkan kategori yang sesuai.
3. Sistem mencari produk serupa di Tokopedia.
4. Sistem membandingkan harga dan menampilkan hasil dengan harga termurah serta terjual terbanyak beserta link pembelian.
5. Hasil pencarian dapat disimpan ke file csv.

## Instalasi dan Penggunaa

### Instalasi
```bash
git clone https://github.com/PriscilliaCandra/comparing_price.git
cd compare_price_online_shop
```

### Menjalankan Aplikasi
```bash
streamlit run final.py
```

## Teknologi yang Digunakan
- Python
- TensorFlow/Keras untuk image classification
- Scraping/API untuk mengambil data dari Tokopedia
- Streamlit untuk tampilan website

## Kontribusi
Jika ingin berkontribusi, silakan lakukan fork repository ini dan buat pull request dengan perubahan yang diusulkan.

---

**Link Repository:** [GitHub - compare_price_online_shop](https://github.com/PriscilliaCandra/compare_price_online_shop)

