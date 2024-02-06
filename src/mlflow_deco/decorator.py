import inspect
import os
import re
import subprocess
from functools import wraps

import git
import mlflow
from mlflow import log_artifact, log_param, log_params, set_tag, set_tags

# set your tracking uri here
MLFLOW_TRACKING_URI = "file:/path/to/mlruns"

def ran_from_jupyter():
    """Infer if you're running from a notebook by looking 
    for ipykernel_launcher.py in the call stack.
    """
    traceback = inspect.stack()
    for frame in traceback:
        if os.path.basename(frame.filename) == "ipykernel_launcher.py":
            return True
    return False


def get_source(notebook_name):

    # No clean way to programmatically get the notebook name
    # if there are multiple notebooks.

    jupyter = ran_from_jupyter()

    if jupyter:
        # If the Run was ran from a notebook,
        # cwd containings the notebook
        dirname = os.getcwd()

        # try to infer the notebook name
        file_list = [file for file in os.listdir(dirname) if file.endswith(".ipynb")]
        if len(file_list) == 0:
            raise ValueError(f"Expected to find at least one notebook under {dirname}")
        elif len(file_list) == 1:
            source = os.path.join(dirname, file_list[0])
        elif notebook_name in file_list:
            source = os.path.join(dirname, notebook_name)
        else:
            raise ValueError("Please provide a valid Jupyter notebook_name.")

    else:  # ran from .py

        traceback = inspect.stack()
        root_frame = traceback[-1]
        source = root_frame.filename

    return source, jupyter


def get_env():
    result = subprocess.run(["conda", "info"], capture_output=True, text=True)
    pat = "active environment : (.+)"
    r = re.search(pat, result.stdout)
    return r.group(1)


def run_conda_pack(env_name: str):
    
    print(f"Running conda pack on env {env_name}.")

    os.makedirs("/tmp/mlflow", exist_ok=True)
    fname = f"/tmp/mlflow/{env_name}.tar.gz"
    if os.path.exists(fname):
        os.remove(fname)  # otherwise conda pack complains

    return subprocess.run(
        [
            "conda",
            "pack",
            "-n",
            env_name,
            "-o",
            fname,
            "--ignore-editable-packages",
        ],
        capture_output=True,
        text=True,
    ), fname


def run_conda_export(env_name: str):

    os.makedirs("/tmp/mlflow", exist_ok=True)
    fname = f"/tmp/mlflow/{env_name}.yaml"
    return subprocess.run(
        [f"conda env export > {fname}"],
        capture_output=True,
        text=True,
        shell=True,
    ), fname


def log_environ():

    # conda-pack the environment. if fail, export a yaml
    # to at least record the packages and verions (conda create
    # using a yaml can get stuck for large environments)

    env_name = get_env()
    result, fname = run_conda_pack(env_name)

    if result.returncode == 0:
        log_artifact(fname)
    else:
        print("Warning: conda pack failed, resorting to conda env export.")
        
        result, fname = run_conda_export(env_name)
        if result.returncode == 0:
            log_artifact(fname)
        else:
            print("Warning: conda env export also failed. Your env will not be logged.")
           

def log_code_version(notebook_name: str, repo_path: str | None):

    source, jupyter = get_source(notebook_name)
    try:
        # override if repo_path is given
        repo = (
            git.Repo(repo_path, search_parent_directories=True)
            if repo_path
            else git.Repo(source, search_parent_directories=True)
        )

        # log git diff patch as an artifact
        tmp = "/tmp/mlflow"
        os.makedirs(tmp, exist_ok=True)
        diff_path = os.path.join(tmp, "diff.patch")
        with open(diff_path, "w") as f:
            f.write(repo.git.diff())
        log_artifact(diff_path)

        # override existing system tags
        repoURL = repo.remotes.origin.url
        branch = repo.active_branch.name
        commit = repo.head.commit.hexsha

        set_tags(
            {
                "mlflow.source.git.repoURL": repoURL,
                "mlflow.source.git.branch": branch,
                "mlflow.source.git.commit": commit,
            }
        )

        # system tags are used internally, so some of them don't show up
        # in mlflow ui - set them as non-system tags as well so they show
        # up in the Tags section.
        set_tags(
            {
                "source.git.repo": repo.working_dir,
                "source.git.repoURL": repoURL,
                "source.git.branch": branch,
                "source.git.commit": commit,
                "source.git.diff_path": os.path.join(
                    mlflow.get_artifact_uri(), "diff.patch"
                ),
            }
        )

    except:
        print(
            "You are using mlflow tracking outside of a repository. Run inside a repo if you want code reproducibility."
        )

    # override existing system tags
    set_tag("mlflow.source.name", source)
    if jupyter:
        set_tag("mlflow.source.type", "NOTEBOOK")


def mlflow_tracking(
    experiment: str = "default",
    run_name: str | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    repo_path: str | None = None,
    notebook_name: str = None,
    log_env: bool = True,
    note: str = "",
):

    """Decorator used to start a new mlflow Run. To override mlflow_tracking args
    (experiment, run_name, etc.) via kwargs to the decorated function, your function
    needs to have **kwargs in its signature (my_function below).

    e.g.

    from mlflow_decorator.decorator import mlflow_tracking
    from mlflow import log_metric, log_param, log_params, log_artifact, log_artifacts, set_tags
    import os
    import numpy as np

    @mlflow_tracking(experiment="tutorial", tracking_uri="path/to/mlruns")
    def my_function(a, b, **kwargs):

        # Your code here

        # log params
        log_params({"param1": 1, "param2": "hello world"})

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

    # decorator args passed as kwargs to my_function takes precedence
    my_function(a, b, experiment="llm_tune")  # "llm_tune" overrides "tutorial"


    Parameters
    ----------

    experiment   str
        The experiment name. If doesn't exist under the tracking_uri, will be created.

    run_name    str, default=None
        Name of the run. If None, defaults to a unique name.

    tracking_uri   str, default=MLFLOW_TRACKING_URI
        The backend that Run params, tags and metrics are logged to. By default artifacts are logged to
        this place as well.

    repo_path    str, default=None
        The absolute path to a repo. If given, the repo name, branch name, commit hash, and diff patch
        of the given repo will be logged. This is useful if you're running code outside of a repo
        but are using imports from a repo you'd like to remember. If None, log_code_version
        looks for the repo that your executing code is under, if any.

    notebook_name    str, default=None
        When running from a notebook, provide the name of the notebook.

    log_env    bool, default=True
        Should the current conda environment be logged for reproducibility.
        Set False when you are debugging saves a lot of time.

    note    str, default=""
        A descriptive note about this Run. The content is displayed on the run's page under the
        Notes sections on mlflow ui. You can set it there as well.
    """

    def decorator(f):

        @wraps(f)
        def inner(*args, **kwargs):

            # kwarg values provided from function call override
            # arg values provided to the decorator
            deco_args = dict(
                experiment=experiment,
                run_name=run_name,
                tracking_uri=tracking_uri,
                repo_path=repo_path,
                notebook_name=notebook_name,
                log_env=log_env,
                note=note,
            )
            deco_args.update(
                {
                    # pop() to remove decorator args from being
                    # passed as kwargs to the decorated function itself
                    k: kwargs.pop(k)  
                    for k in deco_args
                    if k in kwargs
                }
            )

            mlflow.set_tracking_uri(deco_args["tracking_uri"])

            # retrieve the experiment under the tracking_uri
            exp_obj = mlflow.get_experiment_by_name(deco_args['experiment'])
            if exp_obj is None:
                print(f"Experiment {deco_args['experiment']} not found, creating it.")
                experiment_id = mlflow.create_experiment(deco_args['experiment'])

                from urllib.parse import urlparse

                experiment_path = os.path.join(
                    # removes uri type prefix, e.g. "file:"
                    urlparse(mlflow.get_tracking_uri()).path,
                    experiment_id,
                )
                # give group complete access to experiment id folder - required by mlflow
                # so you don't run into permission issues when multiple people
                # share one tracking uri.
                subprocess.run(
                    [f"chmod -R 770 {experiment_path}"], text=True, shell=True
                )
            else:
                experiment_id = exp_obj.experiment_id

            # start the run
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=deco_args["run_name"]
            ) as run:

                run_id = run.info.run_id
                run_path = os.path.join(
                    mlflow.get_tracking_uri(), experiment_id, run_id
                )

                print("Logging mlflow Run params, metrics and tags to", run_path)
                print("Logging mlflow Run artifacts to", mlflow.get_artifact_uri())
                print(f"Experiment name and id: {deco_args['experiment']} {experiment_id}")
                print(f"Run name and id: {run.info.run_name} {run_id}\n")

                log_code_version(deco_args["notebook_name"], deco_args["repo_path"])

                if deco_args["log_env"]:
                    log_environ()

                if deco_args["note"]:
                    # override the mlflow.note.content system tag
                    set_tag("mlflow.note.content", deco_args["note"])

                return f(*args, **kwargs)

        return inner

    return decorator
