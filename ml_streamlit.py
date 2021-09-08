# Importing libraries, 
import streamlit as st 
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import cv2 as cv
import torch
import pickle
import joblib
import wget
import os
from nlpend.parse_desc import Textparser
MODEL_FILE_EXTENSION = ['.pt', '.pth', '.pkl', '.joblib', '.h5']


def main():
    st.title("Here you can upload your model and turn it into ios apk or desktop exe")
    st.info("First, specify several stuff about your app")
    parser = Textparser(0.1,1)
    sentence_input = st.text_input("Please describe how you want to use machine learning")
    if sentence_input is not None:
        out_text = parser.extract(sentence_input)
        out_keywords = parser.keywording(sentence_input)
        #st.write(out_text)
        st.markdown(out_keywords)
       

    col_upload, col_link = st.columns(2)
    with col_upload:
        f = st.file_uploader("Upload your model")
    with col_link:
        dlink = st.text_input("or you may paste the link for the model file")
        if dlink is not None:
            f = wget.download(dlink)
    st.sidebar.title("Choose options here")
    app_mode = st.sidebar.selectbox('Mobile or Desktop', ['Mobile', 'Desktop'])
    if f is not None:
        if app_mode == 'Desktop':
            file_ext = get_file_extension(f)
            ind = MODEL_FILE_EXTENSION.index(file_ext)
            if ind == 0 or ind == 1:
                model = torch.load(f)
            elif ind==2:
                with open(f,'rb') as file:
                    model = pickle.load(file)
            elif ind==3:
                model = joblib.load(f)
            elif ind==4:
                model = load_model(f)
            
            application_tp = st.selectbox("Please specify the application scenario", 
            ['Segmentation', 'Classification', 'Detection'])




def get_file_extension(file_in):
    split_f = os.path.splitext(file_in)
    return split_f[1] # returns file extinsion

if __name__ == "__main__":
  main()