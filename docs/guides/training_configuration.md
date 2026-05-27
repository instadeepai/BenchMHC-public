# Training configuration

Predefined training configuration files are saved in `./configuration`.

## Experiment logging

Training uses [MLflow](development/mlflow.md) for experiment tracking by default (`MLflowLogger`).
You can set `training.logger.class_name` to `MLflowLogger` or `CSVLogger`.
The MLflow experiment name should be set with the `MLFLOW_EXPERIMENT_NAME` environment variable in
`.env`. Optionally, set `MLFLOW_WORKSPACE` to bind runs to a specific MLflow workspace; if omitted,
the default workspace is used. Both env vars are picked up automatically by the MLflow client.

## Custom loss and metrics

Loss and metrics, as well as preprocessing steps, are defined at the variable level. By default, we
implement specific loss and metrics for each output variable (e.g. the `BCELoss` for
a `BinaryOutput` or `MeanAbsoluteError` loss for the `NumericOutput`).

We can implement different loss and metrics through the training configuration, with two different methods:

- We can call a method already implemented in Pytorch, with different parameters. For the loss, we
  screen the `torch.nn` module. For the metrics we screen the `torchmetrics` module, as well as
  the `torchmetrics.classification` submodule.
- We can implement our own loss and metrics in `bench_mhc/custom_objects/losses.py` and
  `bench_mhc/custom_objects/metrics.py` respectively. The custom loss should be inherited from
  the `nn.Module`, and the custom metrics from the `torchmetrics.Metric` module.

The following example calls a custom loss for `hit` and two custom metrics for
`binding_affinity`. Note that these arguments are optional in the training configuration, and the
metrics should be called as a list: we can have one loss per variable, but multiple metrics.

```yaml
outputs:
  hit:
    class_name: BinaryOutput
    default_value: -1
    loss:
      class_name: CustomLoss
      param1: "sum"
      param2: 0.6
  binding_affinity:
    class_name: NumericOutput
    default_value: -1
    metrics:
      - class_name: MeanAbsoluteError
      - class_name: CustomMetric
        name: my_custom_metric
```
