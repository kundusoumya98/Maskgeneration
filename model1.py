
import numpy as np # linear algebra
import os
import sys
import cv2
from tqdm import tqdm_notebook, tnrange,notebook
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import create_model as cm
import tensorflow as tf

st.markdown("<small>by Soumya Kundu ,Jadavpur University</small>",unsafe_allow_html=True)
st.title("Mask image Generator")

st.markdown("\nThis app will generate Mask of an image report.")



col1,col2 = st.columns(2)
image_1 = col1.file_uploader("X-ray 1",type=['png','jpg','jpeg'])


col1,col2 = st.columns(2)
predict_button = col1.button('Predict on uploaded files')


#@st.cache
def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer


def predict(image_1,predict_button = predict_button):
    start = time.process_time()
    if predict_button:
        if (image_1 is not None):
            start = time.process_time()  
            image_1 = Image.open(image_1).convert("RGB")
            x = img_to_array(image_1)[:,:,1]
            x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
            x=np.expand_dims(x, 0) 
            preds_test = model_tokenizer.predict(x, verbose=1)
            preds_test_t = (preds_test > 0.5).astype(np.uint8)
            tmp = np.squeeze(preds_test_t).astype(np.float32)
            opencv_image=(np.dstack((tmp,tmp,tmp)))
            st.image([image_1],width=300)
            st.image([opencv_image],width=300)
            del image_1,opencv_image
            
            


    

model_tokenizer = create_model()


predict(image_1,model_tokenizer)



