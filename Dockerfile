FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
      apt-get update && \
      apt-get upgrade -y && \
      apt-get install -y python3.6 python3.6-dev build-essential cmake libgtk2.0-dev python3.6-tk && \
      curl https://bootstrap.pypa.io/get-pip.py | python3.6

ADD mask_rcnn/requirements.txt /mask_rcnn_requirements.txt
RUN pip3.6 install -r mask_rcnn_requirements.txt

ADD requirements.txt /service_requirements.txt
RUN pip3.6 install -r service_requirements.txt

# Install snet daemon
RUN mkdir -p /tmp/snetd && cd /tmp/snetd && \
      curl -OL https://github.com/singnet/snet-daemon/releases/download/v0.1.4/snetd-0.1.4.tar.gz && \
      tar -xvf snetd-0.1.4.tar.gz && \
      mv snetd-0.1.4/snetd-linux-amd64 /usr/bin/snetd && \
      cd / && rm -r /tmp/snetd

ADD . /semantic-segmentation
WORKDIR /semantic-segmentation
RUN ./buildproto.sh

CMD ["python3.6", "run_service.py"]