FROM python:3.10

WORKDIR /app

COPY streamlit.py .

RUN pip install streamlit requests

EXPOSE 8502

CMD ["streamlit", "run", "streamlit.py", "--server.address=0.0.0.0", "--server.port=8502","--server.enableCORS=false"]