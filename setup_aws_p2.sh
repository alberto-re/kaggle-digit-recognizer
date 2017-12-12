#!/bin/sh
#
# Provisioning script for AWS P2 instances.
#
# Tested on instance p2.xlarge with Amazon Linux AMI 2017.09.1 (HVM).
#
# cuDNN (v6.0 for CUDA 8.0) should be downloaded manually from developer.nvidia.com.

yum update -y
yum install -y git python36 python36-pip gcc kernel-devel
python36 -m pip install pandas scipy sklearn matplotlib keras tensorflow-gpu h5py

wget http://us.download.nvidia.com/XFree86/Linux-x86_64/367.106/NVIDIA-Linux-x86_64-367.106.run

/bin/bash ./NVIDIA-Linux-x86_64-367.106.run -a -s

wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run

sh cuda_8.0.61_375.26_linux-run --silent --toolkit

cd /usr/local \
    && tar xvzf ~/cudnn-8.0-linux-x64-v6.0.tgz \
    && cd -

export PATH="$PATH:/usr/local/cuda/bin"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
