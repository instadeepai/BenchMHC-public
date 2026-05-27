# Track experiments with MLflow

Experiment tracking can be done with [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html).
Runs record parameters, metrics, and artifacts (e.g. configuration, training metadata).

MLflow logging is enabled by default through the `MLflowLogger` configured in the training
config (e.g. `configuration/nnalign_ba_sa.yml`).

!!! note "Skip MLflow logging"

    If you don't need MLflow logging, you can store logs in CSV by setting the
    `training.logger.class_name` to `CSVLogger` in your training configuration.

## 1. Local setup

Set the tracking URI and experiment name as environment variables (this can be in your `.env`):

```text
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_WORKSPACE=my-workspace
MLFLOW_EXPERIMENT_NAME=my-experiment
```

Start a local MLflow server backed by a SQLite database:

```bash
MLFLOW_VERSION=$(python3 -c "import tomllib; print(next(p['version'] for p in tomllib.load(open('uv.lock', 'rb'))['package'] if p['name'] == 'mlflow-skinny'))")
uv tool install "mlflow==${MLFLOW_VERSION}"
uv tool run mlflow server --port 5000 --enable-workspaces
```

Create the workspace manually (only needed once per local server):

```bash
uv run python -c "
import os
import dotenv
import mlflow

dotenv.load_dotenv()
workspace_name = os.environ['MLFLOW_WORKSPACE']
workspace = mlflow.create_workspace(workspace_name, description='Local BenchMHC workspace')
print('Created workspace:', workspace.name)
print('All workspaces:', [w.name for w in mlflow.list_workspaces()])
"
```

Open [http://localhost:5000](http://localhost:5000) in your browser to view runs.

You can then launch your trainings.

!!! tip "File-based logging (no server needed)"

    If you don't need the MLflow UI (e.g. when debugging locally), you can point the tracking
    URI to a local directory instead of a server. MLflow will write all metrics and artifacts
    as files:

    ```text
    MLFLOW_TRACKING_URI=data/mlflow_logs
    ```

    This avoids having to start the MLflow server and works out of the box.

## 2. Remote setup

If you have a remote server, you can set up MLflow tracking by setting the tracking URI in your
`.env`:

```text
MLFLOW_TRACKING_URI=https://<your-remote-server>/
```

as well as credentials, cf. the [MLflow documentation](https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth/#using-environment-variables):

```text
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=<your-password>
```

!!! note "Shared MLflow server for InstaDeep users"

    InstaDeep users can use the shared MLflow server, more details [here](https://www.notion.so/instadeep/Access-MLflow-through-VPN-33ced6e1dfc48062be41fff63ddcbe25?source=copy_link).
