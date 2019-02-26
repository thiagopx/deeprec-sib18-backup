FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
MAINTAINER Thiago PX

RUN apt-get update && \
    apt-get install wget -y && \
    apt-get install python3 -y

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# new user
ARG UNAME=testuser
ARG UID=1000
ARG GID=1000
ARG GPUID=0
RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

# switch to user (local installation)
USER $UNAME
WORKDIR /home/$UNAME

# tensorflow (amx disabled)
#COPY tensorflow-1.10.1-cp35-cp35m-linux_x86_64.whl .
#RUN pip install --user tensorflow-1.10.1-cp35-cp35m-linux_x86_64.whl
#RUN rm tensorflow-1.10.1-cp35-cp35m-linux_x86_64.whl

# install additional requirements
COPY requirements.txt .
RUN pip install --user -r requirements.txt
RUN rm requirements.txt

# MobileNetV2
COPY mobilenet_v2.py /home/$UNAME/.local/lib/python3.5/site-packages/keras_applications

ENV PROJECT deeprec-sib18
ENV NVIDIA_VISIBLE_DEVICES $GPUID
ENV PROJECTDIR /home/$UNAME/$PROJECT
ENV LD_LIBRARY_PATH "$PROJECTDIR/docrec/libs:/opt/localsolver/bin:$LD_LIBRARY_PATH"
ENV PYTHONPATH "$PROJECTDIR/docrec/libs:/opt/localsolver/bin/python37:$PYTHONPATH"

# installing as root
USER root

# Latex (for graphs plotting)
RUN apt install texlive-base -y

# OpenCV deps
RUN apt-get install libsm6 libglib2.0-0 libxrender-dev libxext6 -y

# Keras deps
RUN apt-get install libgomp1

# Andalo's software deps
RUN apt-get install libgl1-mesa-glx qt4-dev-tools libgmp-dev -y

RUN apt install sudo

# Localsolver
COPY LocalSolver_8_5_20190219_Linux64.run .
USER root
RUN bash /home/$UNAME/LocalSolver_8_5_20190219_Linux64.run
RUN rm LocalSolver_8_5_20190219_Linux64.run
