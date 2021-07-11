import streamlit as st
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
import pandas as pd

from transformers import BertTokenizer,BertForSequenceClassification
import os
import retrieve_experts

# App title
st.title("SMU Expert Finder")
query = st.text_input("Please enter research area for which you seek experts", key="topic_textbox")
expert_school = st.selectbox('Please select School from which you wish to retrieve experts for above research area',\
     ('SCIS', 'Business', 'All'),index = 2, key = 'school_select')
num_experts = st.slider('Please choose number of experts you wish to retrieve',1,15, key = 'num_experts_slider')
if query:
    retrieve_experts.main(QUERY = query, EXPERT_SCHOOL = expert_school, NUM_EXPERTS = num_experts)
st.write('Displaying top {} experts in the field of {} from {} school'.format(num_experts,query.upper(),expert_school.upper()))
df = pd.read_csv('./Output/results.csv',index_col=False)
st.dataframe(df)