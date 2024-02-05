FROM ultralytics/ultralytics

RUN apt-get update || true

COPY requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

COPY . /app
WORKDIR /app

EXPOSE 8501

ENTRYPOINT [ "python3" ]
CMD [ "server.py" ]
