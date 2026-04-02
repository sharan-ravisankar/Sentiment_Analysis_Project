#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import streamlit as st
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from text import TextPreprocessor
warnings.filterwarnings('ignore')


# In[2]:


model = pickle.load(open('nlp.pkl','rb'))


# In[3]:


st.title("🧠 Sentiment Analysis Web App")
st.write("Enter any text or customer review below to analyze its **sentiment** using your trained NLP model.")

user_input = st.text_area("💬 Enter text here:", height=150, placeholder="Type a review or feedback...")

print(user_input)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:       
        prediction = model.predict(pd.Series([user_input]))[0]

        if prediction.lower() == "positive":
            st.success(f"✅ Predicted Sentiment: {prediction}")
        elif prediction.lower() == "negative":
            st.error(f"❌ Predicted Sentiment: {prediction}")
        else:
            st.info(f"🤔 Predicted Sentiment: {prediction}")


# In[ ]:




