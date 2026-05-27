# Define default values across multi-stages
ARG UV_DIR=/usr/local/bin/uv
ARG UV_PYTHON_INSTALL_DIR=/python
ARG UV_PROJECT_ENVIRONMENT=/app
ARG UV_VERSION=0.7.8
ARG UV_PYTHON=python3.12.7
ARG UV_COMPILE_BYTECODE=1

# Stage: 'uv'
# It is used to define the uv Docker image
FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv

# Stage: 'env'
# It is used to define python version and install all the Python dependencies
FROM ubuntu:noble AS env

# Copy uv with proper version
ARG UV_DIR
COPY --from=uv /uv $UV_DIR

# Set location for python installed by uv
ARG UV_PYTHON_INSTALL_DIR
ENV UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR

# Set /app for the virtual environment created by uv
ARG UV_PROJECT_ENVIRONMENT
ENV UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT

# Define Python version
ARG UV_PYTHON
ENV UV_PYTHON=$UV_PYTHON

# Byte-compile the Python files for faster application startup
ARG UV_COMPILE_BYTECODE
ENV UV_COMPILE_BYTECODE=$UV_COMPILE_BYTECODE

# Copy the files with locked dependencies
COPY pyproject.toml /tmp/pyproject.toml
COPY uv.lock /tmp/uv.lock

# Install all the dependencies with uv
RUN cd /tmp && uv sync --locked --no-dev --no-cache --no-install-project

# Stage: 'runtime'
# It is used to define the image for the runtime with:
# - linux packages
# - non root user
# - python packages
# - env variables
# - BLAST+ 2.16.0
# Fix the hash of the base cuda/cudnn image to avoid issues

# Build for AMD64 (linux)
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6 AS build-amd64

# Build for ARM64 (macos)
FROM --platform=linux/arm64 ubuntu:22.04 AS build-arm64

FROM build-${BUILDARCH} AS runtime

ARG BUILDARCH
RUN echo "Building for architecture: ${BUILDARCH}"

# Use default values for the GitHub CI
ARG HOST_UID=1001
ARG HOST_GID=123

ENV USER=appuser
ENV HOME_DIRECTORY=/home/$USER/bench-mhc

# Do not create .pyc file cf https://stackoverflow.com/a/60797635/8056572
ENV PYTHONDONTWRITEBYTECODE=1

# Allow to have log in real time
ENV PYTHONUNBUFFERED=1

# Use same GPU IDs between nvidia-smi and torch
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Install the required system dependencies
# Clean after packages' install
# libgomp1 is required by BLAST+ 2.16.0
RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install ca-certificates curl git gnupg libgomp1 sudo -y && \
    rm -rf /var/lib/apt/lists/*

# Install NCBI BLAST+
ENV BLAST_VERSION=2.16.0
ARG BUILDARCH
RUN BLAST_ARCH=$([ "$BUILDARCH" = "arm64" ] && echo "aarch64" || echo "x64") && \
    BLAST_URL="https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/${BLAST_VERSION}/ncbi-blast-${BLAST_VERSION}+-${BLAST_ARCH}-linux.tar.gz" && \
    echo "Downloading BLAST+ from: ${BLAST_URL}" && \
    curl -fL --retry 3 "${BLAST_URL}" -o /tmp/blast.tar.gz && \
    tar -xzf /tmp/blast.tar.gz -C /opt && \
    rm /tmp/blast.tar.gz
ENV PATH="/opt/ncbi-blast-${BLAST_VERSION}+/bin:${PATH}"

# Install Google Cloud SDK
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Create group and user, add -f to skip the command without error if it exists already,
# --no-log-init avoids initializing /var/log/lastlog and /var/log/faillog.
# With very large UIDs (e.g. from host UID mapping), these files are indexed by UID
# and can create huge sparse files that Docker materializes into massive layers.
RUN groupadd --force --gid $HOST_GID $USER && \
        useradd --no-log-init -m --uid $HOST_UID --gid $HOST_GID $USER

# Ensure there is no prompt for password when running sudo
RUN echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the default user
USER $USER

# Set the PATH to have access to python executable
ARG UV_PROJECT_ENVIRONMENT
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

# Set /app for the virtual environment created by uv
ENV UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT

# Set HOME_DIRECTORY as default
WORKDIR $HOME_DIRECTORY

# Set PYTHONPATH to access the 'bench-mhc' Python package in the current directory
ENV PYTHONPATH="."

# Set the terminal color
ENV TERM=xterm-256color

# Copy uv, python and the installed packages
ARG UV_DIR
ARG UV_PYTHON_INSTALL_DIR
ARG UV_PROJECT_ENVIRONMENT
COPY --chown=$USER:$USER --from=env $UV_DIR $UV_DIR
COPY --chown=$USER:$USER --from=env $UV_PYTHON_INSTALL_DIR $UV_PYTHON_INSTALL_DIR
COPY --chown=$USER:$USER --from=env $UV_PROJECT_ENVIRONMENT $UV_PROJECT_ENVIRONMENT

# Stage: 'dev'
# It contains all the dependencies to run the Python package and its associated tests with pytest
FROM runtime AS dev

# Prevent uv from downloading isolated Python builds as Python is already available
ENV UV_PYTHON_DOWNLOADS=never

# Byte-compile the python files for faster application startup
ARG UV_COMPILE_BYTECODE
ENV UV_COMPILE_BYTECODE=$UV_COMPILE_BYTECODE

# Copy the files with locked dependencies
COPY --chown=$USER:$USER pyproject.toml /tmp/pyproject.toml
COPY --chown=$USER:$USER uv.lock /tmp/uv.lock

# Synchronize dependencies to also include dev-specific dependencies
RUN cd /tmp && uv sync --locked --no-cache --no-install-project && rm /tmp/pyproject.toml /tmp/uv.lock

# Stage: lint
# It additionally contains all the linting dependencies to run pre-commit hooks
FROM dev AS lint

# Install libatomic1 since it is missing from ubuntu base image and is needed for the pre-commit
# hook environment (markdownlint-cli uses Node.js which requires libatomic1)
RUN sudo apt update && \
    sudo apt install -y --no-install-recommends libatomic1 && \
    sudo rm -rf /var/lib/apt/lists/*

# Copy the files with locked dependencies
COPY --chown=$USER:$USER pyproject.toml /tmp/pyproject.toml
COPY --chown=$USER:$USER uv.lock /tmp/uv.lock

# Install lint-specific dependencies only
RUN cd /tmp && uv sync --locked --group lint

# Install pre-commit hooks
COPY --chown=$USER:$USER .pre-commit-config.yaml /tmp/pre-commit-config.yaml
RUN git init && pre-commit install-hooks -c /tmp/pre-commit-config.yaml

# Stage: 'docs'
# It contains all the dependencies required to generate the documentation
# Build for AMD64 (linux)
FROM --platform=linux/amd64 ubuntu:22.04 AS build-docs-amd64

# Build for ARM64 (macos)
FROM --platform=linux/arm64 ubuntu:22.04 AS build-docs-arm64

FROM build-docs-${BUILDARCH} AS docs

# Install ca-certificates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ca-certificates --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Provide default values for the CI
ARG HOST_UID=1001
ARG HOST_GID=123
ARG HOME_DIRECTORY=/builds/instadeep/bench-mhc

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Set the PATH to have access to python executable
ARG UV_PROJECT_ENVIRONMENT
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

# Install git
RUN apt update && apt install -y --no-install-recommends git \
    && apt clean autoclean \
    && apt autoremove -y \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Create group and user, add -f to skip the command without error if it exists already,
# --no-log-init avoids initializing /var/log/lastlog and /var/log/faillog.
ENV USER=appuser
RUN groupadd --force --gid $HOST_GID $USER && \
        useradd --no-log-init -m --uid $HOST_UID --gid $HOST_GID $USER

# Set the default user
USER $USER

# Prevent uv from downloading isolated Python builds as Python is already available
ENV UV_PYTHON_DOWNLOADS=never

# Set /app for the virtual environment created by uv
ENV UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT

# Byte-compile the python files for faster application startup
ARG UV_COMPILE_BYTECODE
ENV UV_COMPILE_BYTECODE=$UV_COMPILE_BYTECODE

# Copy uv, python and the installed packages
# It is required by mkdocs-click
ARG UV_DIR
ARG UV_PYTHON_INSTALL_DIR
COPY --chown=$USER:$USER --from=env $UV_DIR $UV_DIR
COPY --chown=$USER:$USER --from=env $UV_PYTHON_INSTALL_DIR $UV_PYTHON_INSTALL_DIR
COPY --chown=$USER:$USER --from=env $UV_PROJECT_ENVIRONMENT $UV_PROJECT_ENVIRONMENT

# Copy the files with locked dependencies
COPY --chown=$USER:$USER pyproject.toml /tmp/pyproject.toml
COPY --chown=$USER:$USER uv.lock /tmp/uv.lock

# Synchronize dependencies to also include doc-specific dependencies
RUN cd /tmp \
    && uv sync --locked --no-dev --group doc --no-cache --no-install-project \
    && rm /tmp/pyproject.toml /tmp/uv.lock

RUN git config --global --add safe.directory $HOME_DIRECTORY

# Set HOME_DIRECTORY as default
WORKDIR $HOME_DIRECTORY

# Set PYTHONPATH to access the 'bench-mhc' Python package in the current directory
ENV PYTHONPATH="."
