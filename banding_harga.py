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

# List of categories
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon'
]

# Image dimensions
img_height = 180
img_width = 180

# Function to scrape Tokopedia
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
            print(f"Error scraping Tokopedia: {e}")

    return product_info 

def scrape_bukalapak(product_name):
    options = webdriver.EdgeOptions()
    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
    search_url = f"https://www.bukalapak.com/products?search%5Bkeywords%5D={product_name}"
    print(f'Membuka url: {search_url}')
    driver.get(search_url)
    
    product_info = []
    
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'bl-product-card-new__description'))
        )
        print('Product ditemukan')

        products = driver.find_elements(By.CLASS_NAME, 'bl-product-card-new__description')
        print(f'Jumlah produk: {len(products)}')
        
        for idx, product in enumerate(products[:5]):
            try:
                try:
                    name_elem = product.find_element(By.CLASS_NAME, 'bl-product-card-new__name')
                    name = name_elem.find_element(By.TAG_NAME, 'a').text.strip()
                
                except Exception as e:
                    print('Error: ', e)
                    
                try:
                    price_elem = product.find_element(By.CLASS_NAME, 'bl-product-card-new__price')
                    price = price_elem.text.strip()
                    
                except Exception as e:
                    print('Error: ', e)
                    
                try:
                    link_elem = name_elem.find_element(By.TAG_NAME, 'a')
                    link = link_elem.get_attribute('href')
                    
                except Exception as e:
                    print('Error: ', e)
                
                product_info.append((name, price, link))
                print(f'Produk #{idx + 1}: {name} - {price} - {link}')
                    
            except Exception as e:
                print('Error: ', e)
                
    except Exception as e:
        print('Error: ', e)
        
    finally:
        driver.quit()
        
    return product_info

def clean_price(price):
    price = re.sub(r'[^\d]', '', price)
    return int(price)

# Streamlit app
st.title("Image Classification and Price Scraping")

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Classifying...')
    
    # Preprocess the image
    image = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(image) 
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Make predictions
    predictions = model.predict(img_batch)
    
    if model.output_shape[-1] == len(data_cat):
        score = predictions[0]
    else:
        score = tf.nn.softmax(predictions[0])
        
    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100
    
    st.write(f'Prediction: **{predicted_class}**')
    st.write(f'Confidence: **{confidence:.2f}%**')
    
    # Scrape Tokopedia
    st.write(f'Mencari harga untuk: **{predicted_class}** di Tokopedia...')
    tokopedia_prices = scrape_tokopedia(predicted_class)
    
    if tokopedia_prices:
        for name, price, link in tokopedia_prices:
            st.write(f'- **{name}**: {price} - [Lihat di Tokopedia]({link})')
    else:
        st.write('Tidak ada hasil ditemukan di Tokopedia')  
          
    # Scrape Bukalapak
    st.write(f'Mencari harga untuk: **{predicted_class}** di Bukalapak...')
    bukalapak_prices = scrape_bukalapak(predicted_class)
    
    if bukalapak_prices:
        for name, price, link in bukalapak_prices:
            st.write(f'- **{name}**: {price} - [Lihat di Bukalapak]({link})')  
    else:
        st.write('Tidak ada hasil ditemukan di Bukalapak')
        
    all_products = tokopedia_prices + bukalapak_prices
    
    cleaned_products = []
    
    for item in all_products:
        name, price, link = item
        cleaned_price = clean_price(price)
        cleaned_products.append((name, cleaned_price, link))
        
    n = len(cleaned_products)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if cleaned_products[j][1] > cleaned_products[j + 1][1]:
                cleaned_products[j], cleaned_products[j + 1] = cleaned_products[j + 1], cleaned_products[j]
                
                
    all_products = cleaned_products
    
    cheapest_name, cheapest_price, cheapest_link = all_products[0]
    
    st.write('Produk Termurah')
    st.write(f'{cheapest_name} - Rp {cheapest_price:,}')
    st.write(f'[Lihat Produk Termurah]({cheapest_link})')