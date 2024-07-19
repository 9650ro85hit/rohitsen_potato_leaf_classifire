import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from PIL import Image

from tensorflow.keras.losses import SparseCategoricalCrossentropy

model = load_model('D:/DeepLearning/rohitsen_potato_leaf_classifire/leaf_classifire.h5', compile=False)

model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

custom_objects = {
    'mse': MeanSquaredError(),
}

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def predict_f(image, model):
    target_size = (256, 256)
    img = image.resize(target_size, Image.Resampling.BILINEAR)
    num_img = np.array(img)
    num_img = num_img / 255.0
    img_array = tf.expand_dims(num_img, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

st.title("Leaf Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        pred_cls, confi = predict_f(image, model)
        st.write(f"**Predicted Class:** {pred_cls}")
        st.write(f"**Confidence:** {confi}%")
