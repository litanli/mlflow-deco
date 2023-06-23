MLFlow is a popular open-sourced experiment tracking, model registry and serving tool under the Apache License 2.0 https://mlflow.org/docs/latest/what-is-mlflow.html. While other MLOps tools like Weights & Biases offer more features, the basic logging, tracking, and querying capabilities are more than enough and is free for commerical use.

Specifically the MLFlow Tracking component allows you to define experiments and launch runs, loging parameters, metrics, tags and artifacts to one centralized backend. It has a front-end UI where you can query runs by any combination of metrics, parameters and tags. Of course, there's a Python API for programmatic querying as well. 

# Definitions
Backend store - the location where MlFlow logs params, metrics and tags of a run.
Artifact store - the location where MlFlow logs artifacts (files and directories).
Tracking uri - the URI of the backend store.
Artifact uri - the URI of the artifact store. By default, MlFlow logs artifacts to the <experiment_id>/<run_id>/artifacts directory under the tracking uri.
Experiments are created under the tracking uri, and runs are created under an experiment.

# Why this Repo
Some of the clunkier/missing features of MLFlow include
- Runs are started under a context manager, so all your code needs an extra tab.
- Conda envs and repository info (repoURL, branch name, commit hash, diff patch) aren't saved like they are in Wandb.
- If you're launching experiments from a notebook, the source tag gets populated with ipykernel_launcher.py rather than the path to your actual notebook.

The mlflow_tracking decorator fixes all of those things:
1. Add a **kwargs argument to your function so the decorator args can be passed at call time.
2. Decorate your function
3. Call your function with decorator args passed as additional kwargs

```
from mlflow_decorator.decorator import mlflow_tracking
    from mlflow import log_metric, log_param, log_params, log_artifact, log_artifacts, set_tags
    import os
    import numpy as np

    @mlflow_tracking(experiment='tutorial')
    def my_function(a, b, **kwargs):

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

        return 'aloha'

    # decorator args passed as kwargs to my_function takes precedence
    my_function(a, b, experiment='tesla coil')  # 'tesla coil' overrides 'default'
```

And adds a few missing and QOL features:

- Detects if you're inside a repo, and sets the repoURL, branch and commit hash as tags of the Run. If you're working tree is dirty, it generates a diff.patch and logs it as an artifact.
- Runs conda-pack https://conda.github.io/conda-pack/ on your current Conda environment. If that fails, it resorts to conda env export. Environment yamls are notoriously bad at recreating complex environments, by bad I mean conda create takes forever. conda-pack works much better but fails when said conda environment contains pip _and_ conda installations of the same package.
- If you're running from a Jupyter notebook, it sets the source as the path of your notebook and your notebook name.
- Creates a new MlFlow Experiment if the experiment name you provide doesn't currently exist.



# Installation
pip install git+https://github.com/litanli/mlflow-deco.git




