import streamlit as st
import numpy as np
import cv2
import pickle
from keras_facenet import FaceNet

# load model
model = pickle.load(open('face_recognition_model.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))

embedder = FaceNet()

st.title("Face Recognition App")

uploaded_file = st.file_uploader("Upload Image", type=['jpg','png'])

def get_embedding(face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    samples = np.expand_dims(face, axis=0)
    return embedder.embeddings(samples)[0]

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR")

    face = cv2.resize(img, (160,160))

    embedding = get_embedding(face)
    embedding = np.expand_dims(embedding, axis=0)

    pred = model.predict(embedding)
    name = encoder.inverse_transform(pred)

    st.success(f"Prediction: {name[0]}")
