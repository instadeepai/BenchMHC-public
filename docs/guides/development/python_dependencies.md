# Python dependencies

Python dependencies are installed thanks to [uv](https://github.com/astral-sh/uv):

- it locks the dependencies (in a lockfile called `uv.lock`) to make the environment fully
  reproducible,
- it replaces Python dependencies / virtual environments managers tools such as
  [`pip`](https://pip.pypa.io/en/stable/), [`pip-tools`](https://github.com/jazzband/pip-tools),
  [`poetry`](https://python-poetry.org/) or [`virtualenv`](https://virtualenv.pypa.io/en/latest/).

---

The dependencies of the environment are defined by the users in `pyproject.toml`:

- the "first level" dependencies of the project are defined in the `project.dependencies` section,
- the lint-specific "first level" dependencies, e.g. linting-related dependencies, are defined in
  the `dependency-groups.lint` section,
- the dev-specific "first level" dependencies, e.g. pytest-related dependencies, are defined in
  the `dependency-groups.dev` section,
- the doc-specific "first level" dependencies, e.g. mkdocs-related dependencies, are defined in
  the `dependency-groups.doc` section,

These dependencies are locked in the
[`uv.lock` file](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile) with the
`uv lock` command. `uv.lock` is a universal or cross-platform lockfile that captures the packages
that would be installed across all possible Python markers such as operating system, architecture,
and Python version. Unlike the `pyproject.toml`, which is used to specify the broad requirements of
your project, the lockfile contains the exact resolved versions that are installed in the project
environment. `uv.lock` is a human-readable TOML file but is managed by `uv` and should not be edited
manually. This file is checked into version control, allowing consistent and reproducible
installation of the repo across machines.

- The following command is used to generate `uv.lock`:

  ```bash
  uv lock
  ```

??? tip "Exporting `uv.lock` dependencies in `requirements.txt` files"

    There is no Python standard for lockfiles at this time, so the format of this file
    is specific to `uv`. To convert the locked requirements in `requirements-txt`-formatted files,
    you can use `uv export`, e.g.:

    - The following command is used to generate `requirements.txt`:

      ```bash
      uv export --format requirements-txt --no-emit-package bench_mhc --frozen --no-dev -o requirements.txt
      ```

    - The following command is used to generate `requirements-dev.txt`:

      ```bash
      uv export --format requirements-txt --no-emit-package bench_mhc --frozen -o requirements-dev.txt
      ```

    - The following command is used to generate `requirements-lint.txt`:

      ```bash
      uv export --format requirements-txt --no-emit-package bench_mhc --frozen --no-dev --group lint -o requirements-lint.txt
      ```

    - The following command is used to generate `requirements-doc.txt`:

      ```bash
      uv export --format requirements-txt --no-emit-package bench_mhc --frozen --no-dev --group doc -o requirements-doc.txt
      ```

!!! info

    The dependencies defined in `pyproject.toml` do not necessarily have a pinned version.
    If no version is provided, it will be defined at lock time.

The locked dependencies are then installed in the `docker` image or in a virtual environment
thanks to the `uv sync` command.

!!! info "`uv sync`"

    - `uv sync` allows to completely synchronize an environment: it will remove dependencies not
      specified in the `uv.lock` file.
    - The `--locked` option in `uv sync --locked` makes the command fail if the `uv.lock` file is
      outdated with respect to the broad dependencies.
    - The `--no-install-project` option in `uv sync --no-install-project` installs the project
      dependencies without installing the repo python project itself (`bench_mhc`).
    - The `--no-dev` option in `uv sync --no-dev` installs the project dependencies without
      installing the dev-specific dependencies. Similarly, specifying `--group lint`/`--group doc`
      in the command makes `uv sync` additionally install the lint/doc-specific dependencies.

## How to add a new dependency?

1. Add the dependency with the following command. This will update both `pyproject.toml` and
   `uv.lock` with the new dependency.

   ```bash
   uv add {pkg}
   ```

1. Commit `pyproject.toml` and `uv.lock`.

Similarly, to add a {dev|lint|doc}-specific dependency, you should use `uv add --dev {pkg}` or
`uv add --group lint {pkg}` or `uv add --group doc {pkg}`. Additional details about adding a
dependency are available in the [`uv` documentation](
https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies).

!!! tip

    It is not necessary to pin a specific version when adding a dependency: the dependency added
    by `uv` will include a constraint, e.g., `>=1.0.0`, for the most recent, compatible version of
    the package. An alternative constraint can be provided: `uv add "polars>=1.0.0"`

## How to upgrade a dependency?

1. Upgrade the dependency with the following command. This will update both `pyproject.toml` and
   `uv.lock` with the new dependency.

   ```bash
   uv add {pkg} --upgrade-package {pkg}
   ```

1. Commit `pyproject.toml` and `uv.lock`.

!!! tip

    If you update the provided constraint e.g. `polars>=1.16.0` without `--upgrade-package`, the
    locked version of the dependency will only change if necessary to satisfy the new constraints.
    Hence, to force the package version to update to the latest within the constraints, you need to
    specify `--upgrade-package`.

Similarly, to update a {dev|lint}-specific dependency, you should use
`uv add --dev {pkg} --upgrade-package {pkg}` or `uv add --group lint {pkg} --upgrade-package {pkg}`.
Additional details about upgrading a dependency are available in the [`uv` documentation](
https://docs.astral.sh/uv/concepts/projects/dependencies/#changing-dependencies).

## Upgrade python or uv version

### How to upgrade `python`?

1. In `Dockerfile`, modify the env variable `UV_PYTHON` with the desired version.
1. In `pyproject.toml`:
   1. update the field `project.requires-python`,
   1. update the field `tool.ruff.target-version`.
   1. update the field `tool.mypy.python_version`.
1. Remove `uv.lock`.
1. Compile the requirements according to `pyproject.toml`:

   ```bash
   uv lock
   ```

1. If you have any issue during the installation of the dependencies, use `>=` in the
   `project.dependencies` section of `pyproject.toml` instead of `==` and restart from step 3.
   Same in the `dependency-groups.{dev|lint}` section of the `pyproject.toml` file for a
   {dev|lint}-specific dependency.
1. Update the `--python-version` for `mypy` hook in `.pre-commit-config.yaml`.
1. Commit the changes: it will trigger the hook to lock the environment if not done before.
1. Run the pre-commit on all the files:

   ```bash
   pre-commit run --all-files
   ```

1. Fix any issue linked to the upgrade (due to pre-commit hooks or tests).
1. Update in the documentation any reference to the old python version e.g. `README.md`.

### How to upgrade `uv`?

1. In `Dockerfile` modify the env variable `UV_VERSION` with the desired version.
1. Update the `rev` field of the hook `uv-pre-commit` in `.pre-commit-config.yaml` to match the
   version in `Dockerfile`.
1. Update the `with.version` field in `.github/workflows/post-release.yaml` to match the version in
   `Dockerfile`.
1. Commit both files.
