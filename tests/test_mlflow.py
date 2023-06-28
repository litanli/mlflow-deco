import os
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import yaml

from mlflow_deco.decorator import (
    get_env,
    get_source,
    log_code_version,
    log_environ,
    mlflow_tracking,
)


class TestGetSource(unittest.TestCase):

    # Mocking ran_from_jupyter as True
    @patch("mlflow_deco.decorator.ran_from_jupyter", return_value=True)
    def test_get_source_from_jupyter(self, mock_ran_from_jupyter):

        cwd = "/path/to/notebook_directory"

        with patch.object(os, "getcwd", return_value=cwd):

            with patch.object(os, "listdir", return_value=[]):
                with self.assertRaises(ValueError):
                    get_source(None)

            # single notebook under cwd
            with patch.object(os, "listdir", return_value=["notebook.ipynb"]):
                self.assertEqual(
                    get_source(None), (os.path.join(cwd, "notebook.ipynb"), True)
                )

                # ignores given notebook name when only one notebook exists
                self.assertEqual(
                    get_source("false_name.ipynb"), (os.path.join(cwd, "notebook.ipynb"), True)
                )

            # multiple notebooks
            file_list = ["notebook1.ipynb", "notebook2.ipynb"]
            with patch.object(os, "listdir", return_value=file_list):

                # given notebook name exists
                self.assertEqual(
                    get_source("notebook1.ipynb"), (os.path.join(cwd, "notebook1.ipynb"), True)
                )

                # incorrect notebook name or user did not provide
                with self.assertRaises(ValueError):
                    get_source("false_name.ipynb")
                    get_source(None)


class TestGetEnv(unittest.TestCase):
    @patch("subprocess.run")
    def test_get_env(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="\n     active environment : env_name\n"
        )
        self.assertEqual(get_env(), "env_name")


@contextmanager
def tmp_dir(path="/tmp/mlflow"):
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def check_obj_logged(fname, path):
    for root, dirs, files in os.walk(path):
        if fname in files:
            return True
    return False


class TestLogEnviron(unittest.TestCase):
    def setUp(self):
        self.tracking_uri = "/tmp/mlflow/mlruns"
        self.env_name = get_env()

    @patch("mlflow_deco.decorator.run_conda_pack")
    def test_conda_export(self, mock_run_conda_pack):
        fname = f"/tmp/mlflow/{self.env_name}.tar.gz"
        mock_run_conda_pack.return_value = (MagicMock(returncode=1, stderr=""), fname)
        with tmp_dir():
            with mlflow.start_run(experiment_id=0) as run:
                log_environ()
                self.assertTrue(
                    check_obj_logged(
                        f"{self.env_name}.yaml",
                        os.path.join(
                            self.tracking_uri, f"0/{run.info.run_id}/artifacts"
                        ),
                    )
                )
        self.assertTrue(os.path.exists(f"/tmp/mlflow/{self.env_name}.yaml"))


def mock_git_repo():
    repo = MagicMock()
    repo.remotes.origin.url.split.return_value = [
        "https://github.com/username/repo.git"
    ]
    repo.active_branch.name = "dev"
    repo.head.commit.hexsha = "1234567890abcdef"
    repo.git.diff.return_value = "diff content"
    return repo


class TestLogCodeVersion(unittest.TestCase):
    def setUp(self):
        self.tracking_uri = "/tmp/mlflow/mlruns"
        self.repo = mock_git_repo()

    @patch("git.Repo")
    def test_log_code_version_inside_repo(self, mock_repo):
        mock_repo.return_value = self.repo

        with tmp_dir():
            with mlflow.start_run(experiment_id=0) as run:
                log_code_version(None, None)
                self.assertTrue(
                    check_obj_logged(
                        "diff.patch",
                        f"/tmp/mlflow/mlruns/0/{run.info.run_id}/artifacts",
                    )
                )
                self.assertTrue(os.path.exists("/tmp/mlflow/diff.patch"))
                for tag in [
                    "mlflow.source.git.repoURL",
                    "mlflow.source.git.branch",
                    "mlflow.source.git.commit",
                    "source.git.repo",
                    "source.git.repoURL",
                    "source.git.branch",
                    "source.git.commit",
                    "source.git.diff_path",
                    "mlflow.source.name"
                ]:
                    self.assertTrue(
                        check_obj_logged(
                            tag, f"/tmp/mlflow/mlruns/0/{run.info.run_id}/tags"
                        )
                    )

    @patch("git.Repo")
    def test_log_code_version_outside_repo(self, mock_repo):
        mock_repo.return_value = self.repo
        mock_repo.side_effect = ValueError("Repo path not found.")

        with tmp_dir():
            with mlflow.start_run(experiment_id=0) as run:
                log_code_version(None, None)
                self.assertTrue(
                    check_obj_logged(
                        "mlflow.source.name", f"/tmp/mlflow/mlruns/0/{run.info.run_id}/tags"
                    )
                )


def get_latest_directory(path):
    directories = [
        os.path.join(path, directory)
        for directory in os.listdir(path)
        if os.path.isdir(os.path.join(path, directory))
    ]
    latest_directory = max(directories, key=os.path.getctime)
    return latest_directory


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


class TestMlflowTracking(unittest.TestCase):
    def setUp(self):

        self.tracking_uri = "/tmp/mlflow/mlruns"
        self.experiment = "test"
        os.makedirs(self.tracking_uri, exist_ok=True)
        self.repo = mock_git_repo()

        @mlflow_tracking(experiment=self.experiment, tracking_uri=self.tracking_uri)
        def f(a=1, b=2, **kwargs):
            return a, b

        self.f = f

    def test_function_returns_are_unaffected(self):
        a, b = self.f()
        self.assertTrue(a, 1)
        self.assertTrue(b, 2)

    def assert_structure(self, tracking_uri, experiment, run_name):
        self.assertTrue(os.path.exists(tracking_uri))
        self.assertTrue(mlflow.get_tracking_uri(), tracking_uri)
        experiment_id = mlflow.get_experiment_by_name(experiment).experiment_id
        experiment_uri = os.path.join(tracking_uri, experiment_id)
        self.assertTrue(os.path.exists(experiment_uri))

        run_uri = get_latest_directory(experiment_uri)
        meta = load_yaml(os.path.join(run_uri, "meta.yaml"))
        self.assertEqual(meta["experiment_id"], experiment_id)
        self.assertEqual(meta["run_name"], run_name)

    def test_decorator_args_are_used(self):

        tracking_uri = "/tmp/mlflow/mlruns_from_decor"
        experiment = "from_decor"
        run_name = "from_decor_run"

        @mlflow_tracking(
            experiment=experiment, run_name=run_name, tracking_uri=tracking_uri
        )
        def f(**kwargs):
            pass

        f()
        self.assert_structure(tracking_uri, experiment, run_name)

    def test_artifact_uri_set_properly(self):

        self.f()

        # Check artifact_uri wasn't set at the tracking_uri level.
        # When this happens, a run_id (alphanumeric) folder is created
        # at the tracking_uri level and contains the logged artifacts,
        # while <experiment_id>/<run_id>/artifacts is empty.

        # only experiment_ids at the tracking_uri level
        self.assertTrue(
            all(
                d.isdigit()
                for d in os.listdir(self.tracking_uri)
                if not d.startswith(".")
            )
        )

        # artifact directory not empty
        self.assertTrue(mlflow.get_tracking_uri(), self.tracking_uri)
        experiment_id = mlflow.get_experiment_by_name(self.experiment).experiment_id

        run_uri = get_latest_directory(os.path.join(self.tracking_uri, experiment_id))
        self.assertTrue(len(os.listdir(os.path.join(run_uri, "artifacts"))))

    def test_func_kwargs_override_decorator_args(self):
        tracking_uri = "/tmp/mlflow/mlruns_override"
        experiment = "override"
        run_name = "override_run"
        self.f(
            experiment=experiment,
            run_name=run_name,
            tracking_uri=tracking_uri,
        )
        self.assert_structure(tracking_uri, experiment, run_name)

    def test_creates_new_experiment_if_nonexistent(self):

        self.f(experiment="new")
        self.assertTrue(mlflow.get_tracking_uri(), self.tracking_uri)
        experiment_id = mlflow.get_experiment_by_name("new").experiment_id

        meta = load_yaml(os.path.join(self.tracking_uri, experiment_id, "meta.yaml"))
        self.assertEqual(meta["name"], "new")
