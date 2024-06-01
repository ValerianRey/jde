# Jacobian Descent Experiments
This repository contains deep learning experiments on benchmarking tasks. It compares
single-objective optimization to some multi-objective approaches. Multi-objective
optimization is performed via jacobian descent, using the library
[torchjd](https://github.com/ValerianRey/torchjd).

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

5) We use Weights and Biases to monitor the runs (losses, metrics and more). In order to use
it, you have to create an account on https://wandb.ai/home.

6) To quickly test the install, run:
   ```bash
   pdm run iwrm_study mnist tiny 1 --wandb-mode=online --name="test install"
   ```
   You might see some warnings and wandb should ask for some access credentials. If it works, at
   the end, you should see the message "11/11 successful experiments".

# Usage

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
