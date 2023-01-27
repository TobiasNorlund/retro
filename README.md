# Data Science Project Repository Template

This repo constitutes a skeleton for a typical ML/DS project. Docker is a first class citizen and can be customized by editing the provided `Dockerfile`.

When starting a new project, please do the following:

1. On GitHub, create your own repository from this template by clicking the "Use this template" button
2. Update `DOCKER_IMAGE_NAME` in `start.sh`
3. Build and start a docker container:
```bash
./start.sh [--gpu] [--notebook] [--tensorboard] [-v|--mount /host/path:/container/path] [--detach]
```
5. Start a development container in VS Code:
   There are two ways this can be done. 
   - Attach to the already running container (preferred when container is running on remote host)
      - In VS Code, install the `Remote-Containers` extention
      - Run `Remote-Containers: Attach to Running Container...` (F1). Select the newly created container
      - In the Explorer pane, click `Open Folder` and type the workspace directory (by default mounted to `/workspace`)
   - Let VS Code manage the container (preferred for local development)
      - In VS Code, install the `Remote-Containers` extention
      - Update `name` in `.devcontainer/devcontainer.json` to the value of `DOCKER_IMAGE_NAME`
      - Run `Remote-Containers: Reopen in container` (F1). Select the newly created container
