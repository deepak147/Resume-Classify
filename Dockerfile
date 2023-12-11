FROM python:3.10.11
WORKDIR app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data'); nltk.download('punkt', download_dir='/usr/local/nltk_data');" ]
COPY . . 
EXPOSE 8080
CMD [ "python", "main.py" ]