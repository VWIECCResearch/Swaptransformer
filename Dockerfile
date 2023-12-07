FROM nvidia/vulkan:1.1.121-cuda-10.1-beta.1-ubuntu18.04
# Defining some variables used at build time to install Python3
ARG PYTHON=python3.8
ARG PYTHON_PIP=python3-pip
ARG PIP=pip
ARG PYTHON_VERSION=3.8.16

# These 3 lines are for fixing the package management issue
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install some handful libraries like curl, wget, git, build-essential, zlib
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y zlib1g-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt update && apt install gdebi-core -y

RUN add-apt-repository ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y sumo sumo-tools sumo-doc

RUN apt-get install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev
# Installing python3.8
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
        tar -xvf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION && \
        ./configure && make && make install && \
        apt-get update && apt-get install -y --no-install-recommends liblzma-dev libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev git && \
        make && make install && rm -rf ../Python-$PYTHON_VERSION* && \
        ln -s /usr/local/bin/pip3 /usr/bin/pip

# Upgrading pip and creating symbolic link for python3
RUN ${PIP} --no-cache-dir install --upgrade pip
RUN ${PIP} install setuptools==65.5.0 pip==21

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN apt-get install -y libssl-dev

WORKDIR /

# Installing numpy, pandas, scikit-learn, scipy
RUN ${PIP} install "numpy==1.23.1" \
                   pandas \
                   tqdm \
                   "opencv-python-headless<4.3" \
                   pynput \
                   paramiko

# Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN ${PIP} install --no-cache-dir \
    gputil \
    lz4 \
    PyOpenGL==3.1.0 \
    python-xz \
    wandb


RUN ${PIP} install --upgrade --no-cache-dir \
    gym-unity \
    mlagents-envs tensorboard \
    matplotlib \
    gitpython \
    torchviz \
    torch \
    torchvision \ 
    torchaudio

RUN ${PIP} install --upgrade protobuf==3.20.0
RUN ${PIP} install efficientnet_pytorch \
    plotly \
    lightning
# Keep in mind that proxies are not supported by Apollo anymore
# ENV HTTPS_PROXY="http://usmiah1usrlocalproxy.vwoa.na.vwg:3128"
# ENV http_proxy="http://usmiah1usrlocalproxy.vwoa.na.vwg:3128"
# ENV http_proxy=http://usmiah1usrlocalproxy.vwoa.na.vwg:3128

RUN export SUMO_HOME=/usr/share/sumo
RUN PATH=$PATH:/usr/bin/sumo:/usr/share/sumo

RUN mkdir /opt/ml
RUN mkdir /opt/data
WORKDIR /opt/ml