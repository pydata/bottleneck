FROM ubuntu:devel
RUN apt-get update
RUN apt-get install -y gcc python3-dev python3-pip
RUN pip3 install --upgrade pip
WORKDIR /tmp
CMD ["pip3", "install", "/bottleneck_src"]