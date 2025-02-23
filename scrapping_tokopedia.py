import streamlit as st
import tensorflow as tf
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from bs4 import BeautifulSoup

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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
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
        except Exception as e:
            print(f"Error scraping: {e}")

    return product_info 

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Classifying')
    
    image = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(image) 
    img_batch = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_batch)
    
    if model.output_shape[-1] == len(data_cat):
        score = predictions[0]
    else:
        score = tf.nn.softmax(predictions[0])
        
    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100
    
    st.write(f'Prediction: **{predicted_class}*')
    st.write(f'Confidence: **{confidence:.2f}%**')
    
    st.write(f'Mencari harga untuk: **{predicted_class}** di Tokopedia...')
    product_prices = scrape_tokopedia(predicted_class)
    
    if product_prices:
        for name, price, link in product_prices:
            st.write(f'- **{name}**: {price} - [Lihat di Tokopedia]({link})')
            
    else:
        st.write('Tidak ada hasil ditemukan di Tokopedia')  