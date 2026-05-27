# Define the shell to use
SHELL := /bin/bash

# Docker related variables
PROJECT_NAME = bench-mhc
DOCKER_IMAGE_NAME = $(PROJECT_NAME)
DOCKER_IMAGE_TAG = $(shell git rev-parse --short HEAD)
DOCKER_IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
DOCKER_IMAGE_DOCS = $(DOCKER_IMAGE_NAME)/docs:$(DOCKER_IMAGE_TAG)
DOCKER_HOME_DIRECTORY = /home/appuser/$(PROJECT_NAME)
DOCKER_RUN_FLAGS = -it --rm --volume $(PWD):$(DOCKER_HOME_DIRECTORY)
IS_GPU_AVAILABLE := $(shell command -v nvidia-smi 2> /dev/null)
IS_GPUS_FLAG_AVAILABLE := $(shell docker run --help | grep gpus)

ifdef IS_GPU_AVAILABLE
	ifdef IS_GPUS_FLAG_AVAILABLE
		DOCKER_GPU_COMMAND = --gpus all
	else
		DOCKER_GPU_COMMAND = --runtime nvidia
	endif
endif

DOCKER_BUILD_FLAGS = --build-arg HOST_UID=$$(id -u) --build-arg HOST_GID=$$(id -g) --build-arg HOME_DIRECTORY=$(DOCKER_HOME_DIRECTORY)

# Default port to use for notebook or docs
port = 8888

.PHONY: help build-image build-docs-image bash notebook tests docs build-docs
.DEFAULT_GOAL := help

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "------------------------------------"
	@echo "build-image:             Build your container."
	@echo "build-docs-image:        Build the doc container."
	@echo "bash:                    Build and run your container."
	@echo "notebook:                Launch jupyter notebook. You can specify a custom port with 'make notebook port={my_port}'."
	@echo "tests:                   Launch the python tests with pytest and generate a coverage report."
	@echo "build-docs:              Build the documentation locally."
	@echo "docs:                    Serve the documentation locally. You can specify a custom port with 'make docs port={my_port}'."

build-image:
	DOCKER_BUILDKIT=1 docker build $(DOCKER_BUILD_FLAGS) -t $(DOCKER_IMAGE) --target dev .

bash: build-image
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_GPU_COMMAND) $(DOCKER_IMAGE) sh -c "uv sync --locked && /bin/bash"

build-docs-image:
	DOCKER_BUILDKIT=1 docker build $(DOCKER_BUILD_FLAGS) -t $(DOCKER_IMAGE_DOCS) --target docs .

build-docs: build-docs-image
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_DOCS) sh -c "mkdocs build"

docs: build-docs-image
	docker run $(DOCKER_RUN_FLAGS) -p $(port):8000 $(DOCKER_IMAGE_DOCS) sh -c "mkdocs serve"

notebook: build-image
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_GPU_COMMAND) -p $(port):8888 $(DOCKER_IMAGE) sh -c \
		"uv sync --locked && jupyter notebook --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''"

tests: build-image
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) sh -c \
		"uv sync --locked && pytest tests bench_mhc --doctest-modules --durations=20 --cov=bench_mhc --junitxml=pytest.xml"
