FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
MAINTAINER Thiago PX

RUN apt-get update && \
    apt-get install python3-dev python3-pip -y

# new user
ARG UNAME=testuser
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

# installing virtualenv
RUN pip3 install -U virtualenv

# switch to user (local installation)
USER $UNAME
WORKDIR /home/$UNAME

ENV PROJECT deeprec-sib18

# create virtual environment
RUN virtualenv --system-site-packages -p python3 envs/$PROJECT
RUN . envs/$PROJECT/bin/activate && pip install --upgrade pip

# tensorflow
COPY tensorflow-1.10.1-cp35-cp35m-linux_x86_64.whl .
RUN . envs/$PROJECT/bin/activate && pip install tensorflow-1.10.1-cp35-cp35m-linux_x86_64.whl
RUN rm tensorflow-1.10.1-cp35-cp35m-linux_x86_64.whl

# install additional requirements
COPY requirements.txt .
RUN . envs/$PROJECT/bin/activate && pip install -r requirements.txt
RUN rm requirements.txt

# MobileNetV2
COPY mobilenet_v2.py envs/$PROJECT/lib/python3.6/site-packages/keras_applications

ENV PROJECTDIR /home/$UNAME/$PROJECT
ENV LD_LIBRARY_PATH "$PROJECTDIR/docrec/libs:$LD_LIBRARY_PATH"
ENV PYTHONPATH "$PROJECTDIR/docrec/libs:$PYTHONPATH"

# Latex (for graphs plotting)
#USER root
#RUN apt install texlive-base -y

# OpenCV deps
USER root
RUN apt-get install -y libsm6 libglib2.0-0 libxrender-dev libxext6


# Andalo's software deps
#RUN apt-get install -y libgl1-mesa-glx qt4-dev-tools libgmp-dev

