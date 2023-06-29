MLflow is an experiment tracking, model registry and serving tool under the Apache License 2.0 https://mlflow.org/docs/latest/what-is-mlflow.html. While other MLOps tools like Weights & Biases offer more features, MLflow is free for commercial use.

# Definitions
Tracking uri: URI of the backend store where MLflow logs params, metrics and tags of a run.<br>
Artifact uri: URI of the artifact store where MLflow logs artifacts (files and directories). By default, MLflow logs artifacts to the <experiment_id>/<run_id>/artifacts directory under the tracking uri.<br>
Experiments are created under the tracking uri, and runs are created under an experiment.<br>

# Why this decorator?
Some of the clunkier or missing features of MLflow include:
- Runs are started under a context manager, so all your code needs an extra tab.
- Conda envs and repo info (repoURL, branch name, commit hash, diff patch) aren't saved like they are in Wandb.
- If you're launching experiments from a notebook, the source tag gets populated with `ipykernel_launcher.py` rather than the path to your actual notebook.

The `mlflow_tracking` decorator fixes all of the above:
1. Add a `**kwargs` argument to your function so the decorator args can be passed at call time
2. Decorate your function
3. Call your function with decorator args passed as additional kwargs


And it adds a few missing and QOL features:

- Detects if you're inside a repo, and sets the repoURL, branch and commit hash as tags of the Run. If your working tree is dirty, it generates a `diff.patch` and logs it as an artifact.
- Runs conda-pack https://conda.github.io/conda-pack/ on your current conda environment. If that fails, it resorts to `conda env export`. Environment yamls are notoriously bad at recreating large environments, by bad I mean `conda create` takes forever. conda-pack runs a bit slower but when it's time to recreate an environment, it's fast. However, conda-pack fails when the environment contains pip _and_ conda installations of the same packages, so if conda-pack fails, consider whether this is the case.
- If you're running from a Jupyter notebook, it sets the source as the path to your notebook and your notebook name.
- Creates a new MLflow Experiment if the experiment name you provide doesn't currently exist.

```
from mlflow_deco.decorator import mlflow_tracking
from mlflow import log_metric, log_param, log_params, log_artifact, log_artifacts, set_tags
import os
import numpy as np

@mlflow_tracking(experiment='tutorial')
def run_experiment(a, b, **kwargs):

    # Your code here

    # log params
    log_params({'param1': 1, 'param2': 'hello world'})

    # log metrics
    for step, mse in enumerate([0.8, 0.85, 0.9]):
        log_metric('mse', mse, step)

    # Log artifacts - larger objects like .pth, .npy, or whole directories.
    # Copies of the object and/or directory are created into the artifact store.

    # define an output directory and save large objects
    output_path = "/tmp/mlflow"  # save in tmp, since mlflow copies artifacts to mlruns
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "big_array.npy"), "wb") as f:
        np.save(f, np.array([1, 2]))

    # log a single file
    log_artifact(os.path.join(output_path, "big_array.npy"))

    # log all contents of a directory
    log_artifacts(output_path)

    return 'hello world'

# decorator args passed as kwargs at call time takes precedence
run_experiment(a, b, experiment='tesla coil')  # 'tesla coil' overrides 'tutorial'
```




# Installation
pip install git+https://github.com/litanli/mlflow-deco.git




