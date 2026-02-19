
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("cat_dog_model.h5")
print(model.input_shape)


# Page title
st.title("🐶 Cat vs Dog Image Classifier")

st.write("Upload an image and the model will predict whether it is a Cat or Dog.")

# Image upload option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Open image
    img = Image.open(uploaded_file)
    
    # Resize image (same size as training)
    img_resized = img.resize((224, 224))
    
    # Show image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to array
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Display result
    if prediction[0][0] > 0.5:
        st.success("Prediction: 🐶 Dog")
    else:
        st.success("Prediction: 🐱 Cat")
