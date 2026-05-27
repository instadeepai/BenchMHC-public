# Installation

The project defines a Python 3.12 package `bench-mhc`. The latter can be used with CLI
commands created thanks to [`click`](https://click.palletsprojects.com/en/stable/). The external
python dependencies are defined in `pyproject.toml` and locked in `uv.lock`(see
[this documentation](./development/python_dependencies.md)).

## Requirements

- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Either [uv](https://docs.astral.sh/uv/getting-started/installation/) or
  [docker](https://docs.docker.com/get-docker/)
- [BLAST+](https://www.ncbi.nlm.nih.gov/books/NBK569861/) (optional) if using the uv virtual environment
   to launch  the `generate-decoys` command line.

!!! note "Minimal versions"

    The project installation has been tested with the following versions: `git==2.34.1`, `uv==0.7.8`,
    `docker==27.3.1` and `BLAST+=2.16.0`. In case of installation issues, try to upgrade to at least those minimal
    versions.

To use the `bench-mhc` package, you will execute its source code either within a virtual
environment created by `uv` or within a `docker` image.

## Virtual environment with `uv`

> [!TIP]
> When you are on macOS with Apple Silicon, using the virtual environment allows you to access the
> GPU with MPS backend thanks to the [Metal](https://developer.apple.com/metal/pytorch/)
> integration.

1. Clone the repository:

   ```bash
   git clone git@github.com:instadeepai/BenchMHC-public.git && cd BenchMHC-public
   ```

1. Copy the `.env.template` to a file named `.env`.

1. Create the virtual environment (named `bench-mhc` and located in `.venv/`) and install the
   required libraries (including `bench_mhc` in editable mode, such that any changes into the code
   affects the runs) with `uv`:

   ```bash
   uv sync --locked
   ```

1. Then activate the virtual environment to get access to `bench-mhc` CLI commands:

   ```bash
   source .venv/bin/activate
   ```

1. You should be able to use the CLI commands, e.g.

   ```bash
   bench-mhc --help
   ```

## Docker

> [!TIP]
> Building and running a Docker container allows you to run the package with access to the GPU
> (CUDA) on Ubuntu.

<!-- markdownlint-disable MD028 -->
> [!WARNING]
> You can use the package within the Docker image on macOS with Apple Silicon but you won't have
> access to the GPU with MPS backend. To do so, you should use it within the
> [virtual environment](#virtual-environment-with-uv).

### Standard Installation

Please install Docker following the [official guidelines](https://docs.docker.com/engine/install/).
In particular:

- For macOS, please install the [Docker Desktop](https://docs.docker.com/docker-for-mac/install/).
- For Ubuntu, please
  [set up Docker’s repositories and install from them](https://docs.docker.com/engine/install/ubuntu/).

### Post installation

As Docker is installed by default under `root`, to avoid prefacing the Docker command with `sudo`,
please create a Unix group named `docker` and add users to it. If you are a Mac user, please skip
this section.

1. Create the Docker group.

   ```bash
   sudo groupadd docker
   ```

1. Add your user to the Docker group.

   ```bash
   sudo usermod -aG docker $USER
   ```

1. Log out and log back in so that your group membership is re-evaluated. If testing on a virtual
   machine, it may be necessary to restart the virtual machine for changes to take effect.

1. Activate the changes to groups:

   ```bash
   newgrp docker
   ```

1. Verify that you can run Docker commands without sudo.

   ```bash
   docker run hello-world
   ```

   This command downloads a test image and runs it in a container. When the container runs, it
   prints an informational message and exits.

1. (Optional) If you initially ran Docker CLI commands using `sudo` before adding your user to the
   Docker group, you may see the following error:

   ```bash
   WARNING: Error loading config file: /home/user/.docker/config.json -
   stat /home/user/.docker/config.json: permission denied
   ```

   This indicates that your `~/.docker/` directory was created with incorrect permissions due to the
   `sudo` commands. To fix this problem, you can change ownership and permissions of the
   `~/.docker/` with the following:

   ```bash
   sudo chown "$USER":"$USER" /home/"$USER"/.docker -R
   sudo chmod g+rwx "$HOME/.docker" -R
   ```

### Enable GPU support with CUDA on Ubuntu

In order to use GPU with the Docker, you should install the
[nvidia-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Docker environment

Follow these steps to start a Docker container with the right environment:

1. Clone the repository:

   ```bash
   git clone git@github.com:instadeepai/BenchMHC-public.git && cd BenchMHC-public
   ```

1. Copy the `.env.template` to a file named `.env`.

1. Run your Docker:

   ```bash
   make bash
   ```

- The previous command will create the Docker container, install the package under **editable
mode** and launch a shell session within. If you want to launch a jupyter notebook instance, use
`make notebook` instead.
