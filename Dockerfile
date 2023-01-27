FROM python:3.9

ARG DOCKER_UID
ARG DOCKER_GID
RUN groupadd -r docker-user -g $DOCKER_GID && useradd -r -u $DOCKER_UID -g $DOCKER_GID -m -s /bin/false -g docker-user docker-user

RUN apt update && apt install -y less nano jq git

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
WORKDIR $DOCKER_WORKSPACE_PATH

# TODO: List project dependencies here
# RUN pip install ...

USER $DOCKER_UID:$DOCKER_GID