FROM python:3.9

ARG DOCKER_UID
ARG DOCKER_GID
RUN groupadd -r docker-user -g $DOCKER_GID && useradd -r -u $DOCKER_UID -g $DOCKER_GID -m -s /bin/false -g docker-user docker-user

RUN apt update && apt install -y less nano jq git zstd

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
WORKDIR $DOCKER_WORKSPACE_PATH

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
                torch==1.12.1+cu113 \
                torchvision==0.13.1+cu113 \
                torchaudio==0.12.1+cu113 

RUN pip install transformers==4.21.0 \
                pytorch-lightning==1.7.4 \
                einops==0.6.0 \
                pytest==7.2.1 \
                sentence-transformers==2.2.2 \
                faiss-gpu==1.7.2 \
                matplotlib==3.6.3 \ 
                seaborn==0.12.2

USER $DOCKER_UID:$DOCKER_GID