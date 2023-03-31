FROM python:3.7
COPY requirements.txt /requirements.txt
RUN chmod 1777 /tmp

# install dependencies
RUN pip3 install -r requirements.txt

# download spacy model
RUN python3 -m spacy download en_core_web_sm

# Build directory architecture
RUN mkdir /input_files
RUN mkdir /output_files
WORKDIR /app
COPY . /app

ENTRYPOINT ["python3"]

CMD ["inference.py"]