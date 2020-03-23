FROM centos:7
RUN yum update -y
RUN yum install -y gcc python3-devel python3-pip
RUN pip3 install --upgrade pip
WORKDIR /tmp
CMD ["pip3", "install", "/bottleneck_src"]