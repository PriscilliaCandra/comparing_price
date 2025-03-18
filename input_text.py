import streamlit as st
import tensorflow as tf
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from bs4 import BeautifulSoup
import re

# Load the pre-trained model
model = load_model('image_classify.keras')

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
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    product_info = []
    products = soup.find_all('div', class_='css-5wh65g') 

    for product in products[:5]:  
        try:
            name = product.find('span', class_='_0T8-iGxMpV6NEsYEhwkqEg==').text
            price = product.find('div', class_='_67d6E1xDKIzw+i2D2L0tjw==').text
            link = product.find('a', href=True)['href']
            product_info.append((name, price, link))
        except Exception:
            pass

    return product_info 

def scrape_bukalapak(product_name):
    options = webdriver.EdgeOptions()
    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
    search_url = f"https://www.bukalapak.com/products?search%5Bkeywords%5D={product_name}"
    driver.get(search_url)
    
    product_info = []
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'bl-product-card-new__description'))
        )
        
        products = driver.find_elements(By.CLASS_NAME, 'bl-product-card-new__description')
        
        for product in products[:5]:
            try:
                name = product.find_element(By.CLASS_NAME, 'bl-product-card-new__name').find_element(By.TAG_NAME, 'a').text.strip()
                price = product.find_element(By.CLASS_NAME, 'bl-product-card-new__price').text.strip()
                link = product.find_element(By.CLASS_NAME, 'bl-product-card-new__name').find_element(By.TAG_NAME, 'a').get_attribute('href')
                product_info.append((name, price, link))
            except Exception:
                pass
    except Exception:
        pass
    finally:
        driver.quit()
    
    return product_info

def clean_price(price):
    price = re.sub(r'[^\d]', '', price)
    return int(price) if price else 0

st.title("Image Classification and Price Scraping")

search_query = st.text_input("Masukkan nama produk:")

uploaded_file = st.file_uploader('Upload an image...', type=['jpg', 'png', 'jpeg'])

def find_cheapest_product(product_name):
    tokopedia_prices = scrape_tokopedia(product_name)
    bukalapak_prices = scrape_bukalapak(product_name)
    
    all_products = tokopedia_prices + bukalapak_prices
    cleaned_products = [(name, clean_price(price), link) for name, price, link in all_products if clean_price(price) > 0]
    
    if cleaned_products:
        cleaned_products.sort(key=lambda x: x[1])
        return cleaned_products[0]
    return None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Classifying...')
    
    image = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(image)
    img_batch = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_batch)
    score = tf.nn.softmax(predictions[0]) if model.output_shape[-1] != len(data_cat) else predictions[0]
    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100
    
    st.write(f'Prediction: **{predicted_class}**')
    st.write(f'Confidence: **{confidence:.2f}%**')
    
    cheapest_product = find_cheapest_product(predicted_class)
    
    if cheapest_product:
        name, price, link = cheapest_product
        st.write(f'Produk Termurah: **{name}** - Rp {price:,}')
        st.write(f'[Lihat Produk Termurah]({link})')
    else:
        st.write('Tidak ada produk ditemukan.')

if search_query:
    st.write(f'Mencari harga untuk: **{search_query}**')
    cheapest_product = find_cheapest_product(search_query)
    
    if cheapest_product:
        name, price, link = cheapest_product
        st.write(f'Produk Termurah: **{name}** - Rp {price:,}')
        st.write(f'[Lihat Produk Termurah]({link})')
    else:
        st.write('Tidak ada produk ditemukan.')
