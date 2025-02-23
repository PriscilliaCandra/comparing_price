import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

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

st.title("Image Classification with Deep Learning")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

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

    st.write(f"This image is most likely a **{predicted_class}**.")
    st.write(f"With confidence: **{confidence:.2f}%**")
