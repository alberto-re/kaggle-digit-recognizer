#!/bin/sh
#
# Provisioning script for AWS P2 instances.
#
# Tested on instance p2.xlarge with Amazon Linux AMI 2017.09.1 (HVM).
#
# cuDNN (v6.0 for CUDA 8.0) should be downloaded manually from developer.nvidia.com.
# and uploaded to a S3 bucket.
S3_BUCKET="<BUCKET_NAME>"

yum update -y
yum install -y python36-pip gcc kernel-devel
python36 -m pip install -r requirements.txt
python36 -m pip install tensorflow-gpu

test -f NVIDIA-Linux-x86_64-367.106.run \
	|| wget http://us.download.nvidia.com/XFree86/Linux-x86_64/367.106/NVIDIA-Linux-x86_64-367.106.run

/bin/bash ./NVIDIA-Linux-x86_64-367.106.run -a -s

test -f cuda_8.0.61_375.26_linux-run \
	|| wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run

sh cuda_8.0.61_375.26_linux-run --silent --toolkit

test -f cudnn-8.0-linux-x64-v6.0.tgz \
	|| aws s3 cp s3://$S3_BUCKET/cudnn-8.0-linux-x64-v6.0.tgz /tmp/.

cd /usr/local \
    && sudo tar xvzf /tmp/cudnn-8.0-linux-x64-v6.0.tgz \
    && cd -

echo export PATH="$PATH:/usr/local/cuda/bin:/usr/local/bin/" > /etc/profile.d/cuda.sh
echo export LD_LIBRARY_PATH="/usr/local/cuda/lib64" >> /etc/profile.d/cuda.sh
