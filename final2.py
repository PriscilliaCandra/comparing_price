import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
import time

# Load model
model = load_model("image_classify.h5", compile=False)

data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon'
]

img_height = 180
img_width = 180

def parse_price(price_str):
    try:
        price_num = price_str.replace('Rp', '').replace('.', '').replace(',', '').strip()
        return int(price_num)
    except:
        return None

def parse_sold(sold_str):
    try:
        num = ''.join(filter(str.isdigit, sold_str))
        return int(num) if num else 0
    except:
        return 0

def scrape_tokopedia_with_sales(product_name):
    search_url = f'https://www.tokopedia.com/search?st=product&q={product_name}'

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--headless")

    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
    product_info = []

    try:
        driver.get(search_url)
        time.sleep(5)

        products = driver.find_elements(By.CLASS_NAME, 'css-5wh65g')

        for product in products[:10]:
            try:
                name = product.find_element(By.XPATH, ".//span[contains(text(), '')]").text
            except:
                name = 'Nama tidak ditemukan'

            try:
                price = product.find_element(By.XPATH, ".//div[contains(text(), 'Rp')]").text
            except:
                price = 'Harga tidak ditemukan'

            try:
                sold = product.find_element(By.XPATH, ".//span[contains(text(), 'terjual')]").text
            except:
                sold = '0 terjual'

            try:
                link = product.find_element(By.TAG_NAME, "a").get_attribute("href")
            except:
                link = "#"

            product_info.append({
                'name': name,
                'price': price,
                'price_num': parse_price(price),
                'sold': sold,
                'sold_num': parse_sold(sold),
                'link': link
            })

    except Exception as e:
        st.error(f"Gagal ambil data Tokopedia: {e}")
    finally:
        driver.quit()

    return product_info

st.title("Klasifikasi Gambar + Cari Harga Termurah & Terjual Tokopedia")

with st.form("search_form"):
    keyword_input = st.text_input("Masukkan keyword pencarian Tokopedia (opsional)")
    uploaded_file = st.file_uploader('Upload gambar', type=['jpg', 'jpeg', 'png'])
    submitted = st.form_submit_button("Cari")

if submitted:
    # Jika gambar belum diupload, beri peringatan
    if uploaded_file is None and keyword_input.strip() == '':
        st.warning("Silakan upload gambar atau masukkan keyword pencarian!")
    else:
        # Proses prediksi hanya jika gambar ada
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            img = image.resize((img_width, img_height))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            st.write('Sedang melakukan klasifikasi...')
            predictions = model.predict(img_array)

            if model.output_shape[-1] == len(data_cat):
                score = predictions[0]
            else:
                score = tf.nn.softmax(predictions[0])

            predicted_class = data_cat[np.argmax(score)]
            confidence = np.max(score) * 100

            st.write(f'Prediksi: **{predicted_class}**')
            st.write(f'Keyakinan: **{confidence:.2f}%**')
        else:
            predicted_class = None

        # Tentukan keyword pencarian
        keyword_to_search = keyword_input.strip() if keyword_input.strip() != '' else (predicted_class if predicted_class else '')

        if keyword_to_search == '':
            st.warning("Tidak ada keyword untuk pencarian Tokopedia")
        else:
            st.write(f'Mencari harga dan jumlah terjual untuk: **{keyword_to_search}** di Tokopedia...')
            products = scrape_tokopedia_with_sales(keyword_to_search)

            if products:
                for p in products:
                    st.markdown(f"- **{p['name']}**\n  - Harga: {p['price']}\n  - Terjual: {p['sold']}\n  - [Lihat Produk]({p['link']})")

                harga_terendah = min(
                    [p for p in products if p['price_num'] is not None],
                    key=lambda x: x['price_num']
                )
                terjual_terbanyak = max(
                    products,
                    key=lambda x: x['sold_num']
                )

                st.markdown("---")
                st.markdown(f"### Harga Termurah:\n- **{harga_terendah['name']}** - {harga_terendah['price']} - [Link]({harga_terendah['link']})")

                st.markdown(f"### Terjual Terbanyak:\n- **{terjual_terbanyak['name']}** - Terjual: {terjual_terbanyak['sold']} - [Link]({terjual_terbanyak['link']})")
            else:
                st.write('Tidak ada hasil ditemukan di Tokopedia')
