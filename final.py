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
import re
import time
import pandas as pd

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
        # Contoh "123 terjual" ambil angka 123
        number = re.findall(r'\d+', sold_str.replace('.', ''))
        if number:
            return int(number[0])
        else:
            return 0
    except:
        return 0

def scrape_tokopedia_with_sales(product_name):
    search_url = f'https://www.tokopedia.com/search?st=product&q={product_name}'

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
    product_info = []

    try:
        driver.get(search_url)
        time.sleep(5)  

        products = driver.find_elements(By.CLASS_NAME, 'css-5wh65g')

        for product in products[:10]:
            try:
                name = product.find_element(By.XPATH, ".//span[contains(@class, '_0T8-iGxMpV6NEsYEhwkqEg==')]").text
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


st.title("Comparing Price in Tokopedia")

uploaded_file = st.file_uploader('Upload image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image untuk prediksi
    img = image.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    if model.output_shape[-1] == len(data_cat):
        score = predictions[0]
    else:
        score = tf.nn.softmax(predictions[0])

    predicted_class = data_cat[np.argmax(score)]

    st.write(f'Prediction: {predicted_class}')

    products = scrape_tokopedia_with_sales(predicted_class)

    if products:
        st.markdown("### All Products:")
        
        for idx, p in enumerate(products, start=1):
            with st.expander(f"{idx}. {p['name']}"):
                cols = st.columns([2, 2, 1])
                with cols[0]:
                    st.markdown(f"Price: {p['price']}")
                    st.markdown(f"Sold: {p['sold']}")
                with cols[1]:
                    st.markdown(f"[Product Detail]({p['link']})")

    
        harga_terendah = min(
            [p for p in products if p['price_num'] is not None],
            key=lambda x: x['price_num']
        )

        terjual_terbanyak = max(
            products,
            key=lambda x: x['sold_num']
        )

        st.markdown("---")
        st.markdown("### The Best Choice")

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Cheapest Price \n\n[{harga_terendah['name']}]({harga_terendah['link']})\n\n**{harga_terendah['price']}** - {harga_terendah['sold']}")
        with col2:
            st.info(f"Sold the Most \n\n[{terjual_terbanyak['name']}]({terjual_terbanyak['link']})\n\n**{terjual_terbanyak['price']}** - {terjual_terbanyak['sold']}")
        
        produk_terpilih = pd.DataFrame([
            {
                'Type': 'Harga Termurah',
                'Name': harga_terendah['name'],
                'Price': harga_terendah['price'],
                'Sold': harga_terendah['sold'],
                'Link': harga_terendah['link']
            },
            {
                'Type': 'Terjual Terbanyak',
                'Name': terjual_terbanyak['name'],
                'Price': terjual_terbanyak['price'],
                'Sold': terjual_terbanyak['sold'],
                'Link': terjual_terbanyak['link']
            }
        ])
        
        st.markdown("Saved Product")
        st.dataframe(produk_terpilih)

        csv_file = "saved.csv"
        produk_terpilih.to_csv(csv_file, index=False)

        with open(csv_file, "rb") as f:
            st.download_button(
                label="Download Saved Product",
                data=f,
                file_name=csv_file,
                mime="text/csv"
            )

    else:
        st.write('Tidak ada hasil ditemukan di Tokopedia')
