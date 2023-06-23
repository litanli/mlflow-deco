from setuptools import setup

setup(
    name='mlflow_deco',
    version='1.0.0',
    description='A mlflow decorator with QOL and handy features missing from native MlFlow.',
    author='Litan Li',
    author_email='litan.li3@gmail.com',
    packages=['mlflow_deco'],
    install_requires=[
        'mlflow',
        'gitpython',
        'conda-pack'
    ],
    license='MIT',
)