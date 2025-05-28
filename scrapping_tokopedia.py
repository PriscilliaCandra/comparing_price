import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
import time

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

def scrape_tokopedia(product_name):
    search_url = f'https://www.tokopedia.com/search?st=product&q={product_name}'

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
    product_info = []

    try:
        driver.get(search_url)
        time.sleep(5)  # Tunggu sampai halaman dimuat

        products = driver.find_elements(By.CLASS_NAME, 'css-5wh65g')  # Produk container

        for product in products[:5]:
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
                sold = 'Jumlah terjual tidak ditemukan'

            try:
                link = product.find_element(By.TAG_NAME, "a").get_attribute("href")
            except:
                link = "#"

            product_info.append((name, price, sold, link))

    except Exception as e:
        print(f"[X] Gagal ambil data Tokopedia: {e}")
    finally:
        driver.quit()

    return product_info

# Streamlit UI
st.title("Scrapping Tokopedia")

uploaded_file = st.file_uploader('Upload gambar', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image untuk prediksi
    img = image.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    st.write('Sedang melakukan klasifikasi...')
    predictions = model.predict(img_array)

    # Output handling
    if model.output_shape[-1] == len(data_cat):
        score = predictions[0]
    else:
        score = tf.nn.softmax(predictions[0])

    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100

    st.write(f'Prediksi: **{predicted_class}**')
    st.write(f'Keyakinan: **{confidence:.2f}%**')

    st.write(f'Mencari harga untuk: **{predicted_class}** di Tokopedia...')
    product_prices = scrape_tokopedia(predicted_class)

    if product_prices:
        for name, price, sold, link in product_prices:
            st.markdown(f"- **{name}**\n  - Harga: {price}\n  - Terjual: {sold}\n  - [Lihat Produk]({link})")
    else:
        st.write('Tidak ada hasil ditemukan di Tokopedia')
