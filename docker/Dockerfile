FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
MAINTAINER Thiago PX

RUN apt-get update && apt-get install -y build-essential curl

# OpenCV deps
RUN apt-get install -y libsm6 libglib2.0-0 libxrender-dev libxext6

# Andalo's software deps
RUN apt-get install -y libgl1-mesa-glx qt4-dev-tools libgmp-dev

# Home and project paths in the VM
ENV PROJECTDIR /home/$UNAME/sib18
ENV LD_LIBRARY_PATH "$PROJECTDIR/docrec/libs:$LD_LIBRARY_PATH"
ENV PYTHONPATH "$PROJECTDIR/docrec/libs:$PYTHONPATH"

# new user
ARG UNAME=testuser
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

# switch to user (local installation)
USER $UNAME
WORKDIR /home/$UNAME

# Anaconda
RUN curl -o installer.sh https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh && \
    bash installer.sh -b -f && \
    rm installer.sh

ENV PATH /home/$UNAME/anaconda3/bin:$PATH

# tensorflow env
RUN conda install pip
#RUN pip3 install --upgrade pip

# Tensorflow
RUN pip install --ignore-installed --upgrade http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp36-cp36m-linux_x86_64.whl

# Keras
RUN pip install keras==2.2.1

# MobileNetV2
COPY mobilenet_v2.py anaconda3/lib/python3.6/site-packages/keras_applications

# OpenCV
RUN pip install opencv-python

# numba
RUN conda install numba

# seaborn (v 0.9.0)
RUN conda install seaborn==0.9.0

# Scikit-image
RUN conda install scikit-image

# Latex (for graphs plotting)
USER root
RUN apt install texlive-full -y