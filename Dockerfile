FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG CUDA_ARCH

RUN if [ -z "${CUDA_ARCH}" ]; \
    then \
        echo "CUDA_ARCH not set, run build with e.g. \`--build-arg CUDA_ARCH=\"6.1\" based on your GPU\`" 1>&2; \
        exit 1; \
    else \
        echo "CUDA_ARCH set to ${CUDA_ARCH}"; \
    fi

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get -yqq update \
 && apt-get install -yqq --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gdb \
        clang \
        python \
        wget \
        vim \
        git \
        libblas-dev \
        liblapack-dev \
        xorg-dev \
        libglvnd-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        zsh 1> /dev/null \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && unset DEBIAN_FRONTEND

# Install newer CMake
RUN wget https://cmake.org/files/v3.19/cmake-3.19.8-Linux-x86_64.sh -q -O /tmp/cmake-install.sh \
 && sh /tmp/cmake-install.sh --skip-license --prefix=/usr/local/  \
 && rm /tmp/cmake-install.sh

# Install more advanced dotfiles for execking into the container
RUN curl https://gist.githubusercontent.com/Auratons/2db1bd338f79bc0e7edc26ee2cf07d48/raw/dotfiles-install.sh | bash

RUN export DIR=/tmp/glfw3 \
 && git clone --depth 1 --branch 3.3.6 https://github.com/glfw/glfw.git ${DIR} \
 && cmake -S "${DIR}" -B "${DIR}/build" -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF \
 && cmake --build "${DIR}/build" --target install --config Debug --parallel $(nproc) \
 && rm -rf ${DIR} \
 && unset DIR

COPY ./docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
