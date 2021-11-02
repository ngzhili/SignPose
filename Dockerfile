# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
#FROM tensorflow/tensorflow

WORKDIR /app

# solves "ImportError: libGL.so.1: cannot open shared object file: No such file or directory"
#https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install tensorflow==2.4.1
RUN pip3 install tensorflow-gpu cudatoolkit==11.0

COPY . .


# required by heroku documentation
# exposes a dynamic port to the outside world 
# port is determined by heroku
# CMD gunicorn --bind 0.0.0.0:$PORT wsgi


CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]