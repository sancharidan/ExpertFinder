import streamlit as st
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
import pandas as pd

from transformers import BertTokenizer,BertForSequenceClassification
import os
import gdown
import zipfile
import time

url = "https://drive.google.com/uc?export=download&id=1mXlgM-gEswtMBIJE6aC9hewDek446HEK"
output = './Model/model.zip'
@st.cache
def download_model():
     files = [file for file in os.listdir('./Model/')]
     if 'scibert_6_epochs_3105_pub_yes_distinctsegid_yes_entvocab_no' in files:
          with st.spinner("Using cached model... " + str(files[0])):
               time.sleep(3)
               return
     else:
          with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):          
               gdown.download(url, output, quiet=False)
          with zipfile.ZipFile(output, 'r') as zip_ref:
               zip_ref.extractall('./Model/')
          os.remove(output)


# @st.cache
# def download_model():
    
#     path1 = './Model'
    
#     # Local
#     # path1 = './data/LastModelResnet50_v2_16.pth.tar'
#     # path2 = './data/resnet50_captioning.pt'
#     # print("I am here.")
    
#     if not os.path.exists(path1):
#         decoder_url = 'wget -O ./Model/ https://www.dropbox.com/s/cf2ox65vi7c2fou/Flickr30k_Decoder_10.pth.tar?dl=0'
        
#         with st.spinner('done!\nmodel weights were not found, downloading them...'):
#             os.system(decoder_url)
#     else:
#         print("Model 1 is here.")

#     if not os.path.exists(path2):
#         encoder_url = 'wget -O ./resnet5010.pt https://www.dropbox.com/s/v0ikcdbh8w2rqii/resnet5010.pt?dl=0'
#         with st.spinner('Downloading model weights for resnet50'):
#             os.system(encoder_url)
#     else:
#         print("Model 2 is here.")

# App title
download_model()
import retrieve_experts

st.title("SMU Expert Finder")
query = st.text_input("Please enter research area for which you seek experts", key="topic_textbox")
expert_school = st.selectbox('Please select School from which you wish to retrieve experts for above research area',\
     ('SCIS', 'Business', 'All'),index = 2, key = 'school_select')
num_experts = st.slider('Please choose number of experts you wish to retrieve',1,50, key = 'num_experts_slider')
if query:
    retrieve_experts.main(QUERY = query, EXPERT_SCHOOL = expert_school, NUM_EXPERTS = num_experts)
    st.write('Displaying top {} experts in the field of {} from {} school'.format(num_experts,query.upper(),expert_school.upper()))
    df = pd.read_csv('./Output/results.csv',index_col=False)
    st.dataframe(df)
