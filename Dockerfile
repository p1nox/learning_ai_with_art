FROM python:3.5
MAINTAINER Raul Pino <raul@pino.com>

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y curl wget unzip vim htop
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH $PATH:/root/.local/bin

RUN mkdir /home/code
WORKDIR /home/code
COPY . /home/code

RUN wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O assets/imagenet-vgg-verydeep-19.mat

RUN pip install -r /home/code/requirements.txt

CMD bash
