FROM pytorch/pytorch:latest

# Set bash as default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /app/

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# update the local files
RUN apt-get update 

#install some dependancies for CV2
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

RUN pip install jupyterlab

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888