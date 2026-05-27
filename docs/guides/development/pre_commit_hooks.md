# Pre-commit hooks

Before contributing to the repository, you will need to install our pre-commit hooks in the virtual
environment (outside the Docker environment):

1. If not already done, [create and activate the virtual environment named `bench-mhc`](
   ../installation.md#virtual-environment-with-uv) and install the linting requirements and the
   package's dependencies (to properly run the `mypy` pre-commit hook):

   ```bash
   uv sync --locked --all-groups
   ```

1. Install the pre-commit hooks within the virtual environment created by `uv`:

   ```bash
   source .venv/bin/activate && pre-commit install -t pre-commit -t commit-msg
   ```

To enable in your IDE code completion code navigation and inspection of the installed libraries,
you need to set the Python interpreter of your IDE to the environment defined in the Docker image
created above. To set it up:

- If you are a VSCode user, you can attach to the running Docker container (created above with
  `make bash`) following
  [this documentation](https://code.visualstudio.com/docs/devcontainers/attach-container).
- If you are a PyCharm user, you cannot yet directly attach to a running container (cf.
  [this thread](https://youtrack.jetbrains.com/issue/PY-53854/Use-python-interpreter-from-running-Docker-container)).
  As recommended in
  [their documentation](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html),
  you hence have to manually specify the `./Dockerfile` and the `docker build` options and arguments
  (that you will find in `./Makefile` at the `build` command), so that PyCharm runs its own
  container that you will use as your IDE environment.
