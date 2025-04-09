import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load trained generator
generator = tf.keras.models.load_model('your_saved_generator_model.h5')  # Save model after training

def generate_clothing():
    noise = tf.random.normal([1, 100])
    prediction = generator(noise, training=False)
    img = prediction[0, :, :, 0] * 127.5 + 127.5
    return img.numpy()

st.title("👗 AI-Based Fashion Designer")
st.write("Generate clothing designs using DCGAN!")

if st.button("Generate Design"):
    img = generate_clothing()
    st.image(img, caption="AI-generated clothing", use_column_width=True)
