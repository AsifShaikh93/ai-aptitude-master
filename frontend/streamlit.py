import streamlit as st
import requests

API_URL = 'https://ai-aptitude-master.c-321a6c0.stage.kyma.ondemand.com/solve'

st.set_page_config(page_title='AI Aptitude Master')
st.title('AI Aptitude Master')

question = st.text_input('Ask me any aptitude question')

if st.button('Answer'):
    if question:
       
        answer_container = st.empty()
        full_response = ""
        
        try:
            with requests.post(API_URL, json={'question': question}, stream=True) as response:
                if response.status_code == 200:
                    
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            
                            answer_container.markdown(full_response)
                else:
                    st.error(f"Error: Received status code {response.status_code}")
                    st.write(response.text)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question first.")