FROM nvidia/cudagl:11.2.2-devel-ubuntu20.04

ARG CUDA_ARCH

RUN if [ -z "${CUDA_ARCH}" ]; \
    then \
        echo "CUDA_ARCH not set, run build with e.g. \`--build-arg CUDA_ARCH=\"6.1\" based on your GPU\`" 1>&2; \
        exit 1; \
    else \
        echo "CUDA_ARCH set to ${CUDA_ARCH}"; \
    fi

RUN export DEBIAN_FRONTEND=noninteractive \
 && ln -fs /usr/share/zoneinfo/Europe/Prague /etc/localtime \
 && apt-get -yqq update \
 && apt-get install -yqq --no-install-recommends \
        ssh \
        build-essential \
        gcc \
        g++ \
        gdb \
        clang \
        rsync \
        tar \
        python \
        wget \
        vim \
        git \
        libblas-dev \
        liblapack-dev \
        xorg-dev \
        libglvnd-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglew-dev \
        libopenblas-dev \
        libboost-all-dev \
        zsh \
        tmux 1> /dev/null \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && unset DEBIAN_FRONTEND

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
    echo 'Port 2222'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN echo 'root:root_pass' | chpasswd

# Install newer CMake
RUN wget https://cmake.org/files/v3.23/cmake-3.23.1-linux-x86_64.sh -q -O /tmp/cmake-install.sh \
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

RUN export DIR=/tmp/cli11 \
 && git clone --depth 1 --branch v2.1.2 https://github.com/CLIUtils/CLI11.git ${DIR} \
 && cmake -S "${DIR}" -B "${DIR}/build" \
 && cmake --build "${DIR}/build" --target install --config Debug -j $(nproc) \
 && rm -rf ${DIR} \
 && unset DIR

RUN export DIR=/tmp/nlohmann \
 && git clone --depth 1 --branch v3.10.5 https://github.com/nlohmann/json.git ${DIR} \
 && cmake -S "${DIR}" -B "${DIR}/build" \
 && cmake --build "${DIR}/build" --target install --config Debug -j $(nproc) \
 && rm -rf ${DIR} \
 && unset DIR

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

COPY ./docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]

#docker run --gpus all --rm -it \
#    -e DISPLAY=$DISPLAY \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -v $(pwd):$(pwd) \
#    -w $(pwd) \
#    --cap-add sys_ptrace \
#    -v $HOME/.Xauthority:/root/.Xauthority \
#    --net=host \
#    --ipc=host \
#    -e QT_X11_NO_MITSHM=1 \
#    renderer:latest zsh
