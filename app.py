import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model once and cache it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/home/developer/Coding/Kaggle/Emotion_detection/emotion_detection_model.h5")

model = load_model()


CLASS_NAMES = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Image preprocessing
def preprocess_image(image, target_size=(128, 128)):
    image = image.convert('RGB')  # make sure 3 channels
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # normalize
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

# Streamlit app
st.title("Emotion Detection with AlexNet 😃")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)

    st.write("Detecting emotion...")
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.success(f"Predicted Emotion: **{predicted_class}** ({confidence:.2f} confidence)")

    # Optional: show confidence for all classes
    st.subheader("Confidence Scores:")
    for label, prob in zip(CLASS_NAMES, predictions):
        st.write(f"{label}: {prob:.2f}")

