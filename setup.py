from setuptools import setup, find_packages

setup(
    name='mlflow_deco',
    version='1.0.0',
    description='A mlflow decorator with QOL and handy features missing from native MlFlow.',
    author='Litan Li',
    author_email='litan.li3@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'mlflow',
        'gitpython',
        'conda-pack',
        'pyyaml',
        'numpy'
    ],
    license='MIT',
)