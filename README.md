# Project 2.1: AI and Machine Learning

- Group: AI/ML (Group 1)
- BSc Computer Science, Maastricht University
- Academic Year: 2026/2027

## Description

Fork of [DennisSoemers/ml-agents](https://github.com/DennisSoemers/ml-agents)
(branch `fix-numpy-release-21-branch`). We collect data from deep RL training
runs on Unity environments and train ML models to predict properties of those runs.
Upstream code is unchanged except this README, our own code will be added in separate folders.

## Prerequisites - use the specific version mentioned

- Git, Conda
- Python **3.10.12** 
- Unity Editor **2022.3.4f1**
- macOS (Apple Silicon): Xcode command line tools (`xcode-select --install`), CMake,
  and Rosetta (`softwareupdate --install-rosetta --agree-to-license`)
> Everyone must use the same version for everything!!!

## Installation

1. Clone and enter the repository:
```bash
git clone --branch fix-numpy-release-21-branch https://github.com/UGOTTHEBIGL/ml-agents-group1.git
cd ml-agents-group1
```

2. Create and activate the environment:
```bash
conda create -n mlagents -c conda-forge python=3.10.12
conda activate mlagents
python -m pip install "setuptools<81"
```
3. Install grpcio 1.48.2 from the default channel - only for Mac
```bash
conda install -c defaults "grpcio=1.48.2"
```
4. Install ml-agents-envs, then ml-agents:
```bash
python -m pip install -e ./ml-agents-envs
CMAKE_POLICY_VERSION_MINIMUM=3.5 python -m pip install -e ./ml-agents
```
> The prefix in the second command is needed on macOS, if it doesn't work on linux or windows remove it

5. In Unity Hub, Add project from disk and select the `Project/` folder. The first open
   can take several minutes. If it seems stuck, see Known problems.

Windows and Linux: not yet tested.

### Check that it works
1. `python --version` should print 3.10.12
2. `python -c "import grpc; print(grpc.__version__)"` should print 1.48.2
3. `mlagents-learn --help` prints the usage text (the two warnings are normal)
4. In Unity Hub open the `Project/` folder with editor 2022.3.4f1 and the project should load propertly

### Known problems

| Problem | Fix |
|---|---|
| `Package requires a different Python: 3.10.20 not in '<=3.10.12,>=3.10.1'` | `conda install -c conda-forge python=3.10.12` |
| `No module named 'pkg_resources'` | `python -m pip install "setuptools<81"` |
| pip tries to build grpcio 1.48.2 from source and fails | do step 3 (grpcio from the default channel) before step 4 |
| `Compatibility with CMake < 3.5 has been removed` (building onnx) | prefix the command with `CMAKE_POLICY_VERSION_MINIMUM=3.5` |
| Unity stuck on "Initialize package manager" (macOS Apple Silicon) | install Rosetta (see Prerequisites), then restart Unity Hub |
