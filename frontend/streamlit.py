import streamlit as st
from agent import agent
import requests

API_URL='https://ai-aptitude-master.c-321a6c0.stage.kyma.ondemand.com/solve'

st.set_page_config(page_title='Ai aptitude master')

st.title('Ai aptitude master')

question=st.text_input('Ask me any aptitude question')

if st.button('Answer'):
    if question:
        with st.spinner('thinking ...'):
            
            response=requests.post(
                API_URL,
                {'question': question 
            })
   