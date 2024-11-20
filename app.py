import os
from PIL import Image
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras

classes = ["Real", "Fake"]

st.title("Deepfake Image Classification")

# @st.cache_resource  
def load_model():
    return keras.models.load_model("transfer_model.keras")

def main():
    model = load_model()
    
    user_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if user_image:
        image = Image.open(user_image)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        target_size = (224, 224) 

        img = image.resize(target_size)

        img_matrix = img_to_array(img) / 255.0 

        img_matrix = np.expand_dims(img_matrix, axis=0)  
        
        st.write("Running detection...")

        predictions = model.predict(img_matrix)
        
        if predictions.shape[-1] == 2: 
            probabilities = tf.nn.softmax(predictions).numpy()[0]
            confidence = np.max(probabilities)
            predicted_class = classes[np.argmax(probabilities)]
        else:
            confidence = predictions[0][0] if predictions[0][0] > 0.5 else 1 - predictions[0][0]
            predicted_class = classes[0] if predictions[0][0] > 0.5 else classes[1]

        st.write(f"Prediction: {predicted_class} with confidence {confidence:.2f}")

if __name__ == "__main__":
    main()
