FROM ubuntu:22.04

RUN apt-get update && apt-get upgrade -y

# Install packages
RUN apt-get install -y python3 python3-pip git wget
RUN pip3 install numpy matplotlib kafka-python dropbox msgpack simplejpeg pandas scipy timm tk
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install python3-tk -y 
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
RUN git clone https://github.com/Spyridonidis98/Cross-View-Localization.git
WORKDIR /app/Cross-View-Localization

#RUN wget https://archive.apache.org/dist/kafka/3.5.2/kafka_2.13-3.5.2.tgz
#RUN tar -xzf kafka_2.13-3.5.2.tgz && rm kafka_2.13-3.5.2.tgz && cd kafka_2.13-3.5.2

# Use the correct format for CMD
# CMD ["python3", "-c", "import numpy as np; print('done')"]

# to build docker image run: docker build -t ubuntu-test .
# to run image run: docker run -it ubuntu-test /bin/bash