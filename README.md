# Jacobian Descent Experiments

This repository contains the code to reproduce the experiments of
[Jacobian Descent For Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232).
It uses an early version of the library [torchjd](https://github.com/TorchJD/torchjd), contained in `src/torchjd`.

> [!WARNING]
> Since this repo uses an outdated version of torchjd, it is not a good resource to start learning how to
> use this library. For the same reason, it is not the best way to learn how to train neural networks
> with Jacobian descent. Instead, we recommend reading the documentation of
> [torchjd](https://github.com/TorchJD/torchjd).

# Installation
The installation steps are given for Linux (more specifically, they have been tested on recent
versions of Ubuntu, Fedora and Debian) but they should be easily adaptable to work on other
operating systems.
1) Clone the repository
2) Install Python 3.10.13. We use pyenv to manage python versions:
    - Install pyenv: https://github.com/pyenv/pyenv#installation
    - Install libraries that are required to install python with pyenv:
      ```bash
      sudo apt-get install build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev
      ```
    - Install a python 3.10.13 version using pyenv:
      ```bash
      pyenv install 3.10.13
      ```
    - Automatically activate this python version when you are inside this repo (command to run from
    the root of jde):
      ```bash
      pyenv local 3.10.13
      ```

3) Install `pdm`:
   - Download and install pdm:
     ```bash
     curl -sSL https://pdm.fming.dev/install-pdm.py | python3 -
     ```
   - Make pdm accessible from your `PATH`. In your `.bashrc` file, add:
     ```bash
     export PATH="$PATH:$HOME/.local/bin"
     ```

4) Inside the project root folder, install the dependencies:
   ```bash
   pdm install --frozen-lockfile
   ```
   It should create a virtual environment in a `.venv` folder.
   > ⚠️ If it does not create this `.venv` folder, you can try to run `pdm venv create`, followed by
   `pdm use .venv/bin/python`, and install the project by re-running `pdm install
   --frozen-lockfile`.

   > 💡 The python version that you should specify in your IDE is `path_to_jde/.venv/bin/python`.

5) We use Weights and Biases to monitor the runs (losses, metrics and more). In order to reproduce
our experiments, you have to create an account at https://wandb.ai/home. You should then create a team
that you can name however you want, and a project that you should name "jde".
In `src/scripts/download_study`, you have to replace "team-name" by the team name that you have chosen.

6) To quickly test the install, run:
   ```bash
   pdm run iwrm_study mnist tiny 1 --wandb-mode=online --name="test install"
   ```
   You might see some warnings and wandb should ask for some access credentials. If it works, at
   the end, you should see the message "11/11 successful experiments".

# Explanations

- A `problem` is defined mainly as a supervised dataset for which we want to train a model,
  together with a number of epochs for which we want to train. It is thus generally specified with a
  `dataset_key`, `size_key` tuple. For instance `mnist tiny` specifies the problem corresponding to
  a version of [MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
  with a very small number of samples and epochs, used for testing purposes.
- A `solution` defines how training is performed. It encompasses many hyper-parameters, such as the
  loss function(s) to use, the model architecture, the batch sizes, the optimizer, and, more
  interestingly for us, the model wrapper which defines how the backward pass should be performed in
  the context of jacobian descent.
- An `experiment` is a `(problem, solution)` pair.
- A `study` is a collection of `experiments` whose performance are comparable.

# Usage

The scripts to reproduce our experiments generally take 3 parameters: 
- The dataset key (`svhn`, `cifar10`, `euro_sat`, `mnist`, `fashion_mnist`, `kmnist`).
- An int indicating the seed and the subset on which the experiment should run
(we used `1`, `2`, `3`, `4`, `5`, `6`, `7`, and `8`).
- A study version, which is used to separate the results on wandb into a different group, in case of
problem (just use `v1`).

All the commands mentioned below should be run from the root of `jde`.
Be warned that the time required to reproduce all of our experiments on all seeds is in hundreds of
hours.

The script for the main experiments is the following:
```bash
   ./src/scripts/study_pipeline.sh dataset_key seed version
```
For instance, to obtain the results for `cifar10` on the first seed / subset, you have to run:
```bash
   ./src/scripts/study_pipeline.sh cifar10 1 v1
```
This will run the full pipeline, including the learning-rate selection, for all aggregators.

To run the batch size study for `cifar10` on the first seed / subset, you have to use the following
scripts:
```bash
   ./src/scripts/study_pipeline_4.sh cifar10 1 v1
   ./src/scripts/study_pipeline_16.sh cifar10 1 v1
   ./src/scripts/study_pipeline_64.sh cifar10 1 v1
```

To run the Adam study on `cifar10` and `svhn` on the first seed / subset, you have to run:
```bash
   ./src/scripts/study_pipeline_adam_cifar10.sh 1 v1
   ./src/scripts/study_pipeline_adam_svhn.sh 1 v1
```
