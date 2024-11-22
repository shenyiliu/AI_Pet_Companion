FROM nvcr.io/nvidia/pytorch:22.08-py3
USER root
WORKDIR /workspace

# update timezone
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# 更新软件源，安装基础软件 \
RUN sed -i "s@http://ftp.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://security.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://archive.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://security.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    export vimrc=~/.vimrc && \
    echo "set colorcolumn=80" >> $vimrc && \
    echo "set textwidth=80" >> $vimrc && \
    echo "set ts=4" >> $vimrc && \
    echo "set tabstop=4" >> $vimrc && \
    echo "set expandtab" >> $vimrc && \
    echo "%retab!"  >> $vimrc && \
    echo "syntax enable" >> $vimrc && \
    apt update && apt upgrade -y &&\
    apt install gcc lsof vim -y && \
    apt install curl sudo musl-dev -y && \
    apt install tzdata -y
RUN dpkg-reconfigure --frontend noninteractive tzdata

# Chinese font suport
RUN apt-get install language-pack-zh-hans -y && \
    echo LANG="zh_CN.UTF-8" >> /etc/environment && \
    echo LANGUAGE="zh_CN:zh:en_US:en" >> /etc/environment && \
    echo en_US.UTF-8 UTF-8 >> /var/lib/locales/supported.d/local && \
    echo zh_CN.UTF-8 UTF-8 >> /var/lib/locales/supported.d/local && \
    echo zh_CN.GBK GBK >> /var/lib/locales/supported.d/local && \
    echo zh_CN GB2312 >> /var/lib/locales/supported.d/local && \
    locale-gen && \
    echo LANG="zh_CN.UTF-8" >>  /etc/default/locale && \
    echo LANGUAGE="zh_CN:zh:en_US:en" >> /etc/default/locale


# install kineto
RUN echo "begin install kineto" && \
    git clone --recursive https://github.com/pytorch/kineto.git && \
    cd kineto/libkineto && \
    mkdir build && cd build && \
    cmake .. &&\
    make && make install && \
    cd /workspace

# install cusparse 0.1.0
RUN echo "begin install cusparse" && \
    apt install software-properties-common -y && \
  	apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
  	add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
  	apt-get update && \
  	apt install libcusparselt0=0.1.0.2-1 libcusparselt-dev=0.1.0.2-1 -y

# install FastTransFormers
RUN echo "begin to build FasterTransformer" && \
	  git clone https://github.com/NVIDIA/FasterTransformer.git && \
    cd FasterTransformer && \
    git checkout -b v5.1 \
    mkdir build && cd build && \
    cmake -DSM=86 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DSPARSITY_SUPPORT=ON .. && \
    export cpu_num=`cat /proc/cpuinfo| grep "processor"| wc -l` && \
    make -j${cpu_num} && \
    cd /workspace

# install python project
WORKDIR /workspace/get_diet
COPY . .
RUN chmod +x ./start.sh

EXPOSE 5518
RUN /opt/conda/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    /opt/conda/bin/pip install -r requirements.txt

# start
CMD ["./start.sh"]


