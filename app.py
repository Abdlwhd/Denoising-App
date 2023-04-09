import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
header=st.container()
with header:
    st.title('Document Image Denoising App')
    st.subheader('Upload an Image Document that has noise.')
    st.text('The App will denoise the image and an output will be provided in a form of image.')

def get_model():
    model=tf.keras.models.load_model('denoise_model.h5')
    return model

def prediction(img):
    model = get_model()
    IMG_SIZE = 224  # 50 in txt-based
    img_array = cv2.resize(img,(1024,1024))
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.astype("float32") / 255
    pred_img = new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 1)
    pred_img = model.predict(pred_img)
    return pred_img
    


model=st.container()
with model:
    image=st.file_uploader('Upload an Image')
    sel_col, disp_col= st.columns(2)
    if image is None:
        st.text('Waiting for an Image')
    else:
        sel_col.image(image,output_format="auto")
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        nsy_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img=prediction(nsy_img)
        disp_col.image(img,width=340)
   