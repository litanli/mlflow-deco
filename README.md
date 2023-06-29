### MLflow Decorator

[MLflow](https://mlflow.org/docs/latest/what-is-mlflow.html) is an experiment tracking tool under the Apache License 2.0. While other MLOps tools like Weights & Biases offer more features, MLflow is free for commercial use.<br><br>
June 2023


---
#### Definitions
_Tracking uri_: backend store location where MLflow logs the params, metrics and tags of a run.<br>
_Artifact uri_: artifact store location where MLflow logs artifacts (files and directories).<br><br>
The hierarchy goes:
* backend and artifact store
* experiment
* run<br>
---
#### Why this decorator? 
Some of the MLflow's missing or clunkier aspects:
- does not log your conda environment and repository info
- populates the source tag with `ipykernel_launcher.py` when prototyping experiments from a notebook
- starts runs under a context manager, so all your code needs an extra tab

The `mlflow_tracking` decorator fixes the above:
- logs the repo URL, branch name, and commit hash as run tags
- if your working tree is dirty, logs a `diff.patch` artifact
- runs [conda-pack](https://conda.github.io/conda-pack/) on your current conda environment; if that fails, resort to `conda env export`<sup>*</sup> 
- populates the source tag with `path/to/notebook_name` when launching from a notebook
- creates a new MLflow experiment if the provided experiment name doesn't exist

<sup>*</sup> Environment yamls are notoriously bad at recreating large environments, by bad I mean `conda create` takes forever. `conda pack` runs a bit slower but when it's time to recreate an environment, it's fast. <br><br>
*Remark*: `conda pack` fails when the environment contains both pip _and_ conda installations of the same packages, so if it fails, check whether this is the case.

---
1. add a `**kwargs` argument to your function so decorator args can be passed at call time
2. decorate your function
3. call your function with decorator args passed as kwargs





```python
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

    return "hello world"

# decorator args passed as kwargs at call time takes precedence
run_experiment(a, b, experiment='tesla coil')  # 'tesla coil' overrides 'tutorial'
```
---
#### Installation 
pip install git+https://github.com/litanli/mlflow-deco.git
