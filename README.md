# BDMA experimental repo

## Importing Network Diffusion library
Currently we're working on unstable version of the package and it's expected to
observe bugs. Therefore instead of installing it via PIP, we will clone it as
a git submodule. With this approach we will mimic that Network Diffusion is 
installed in our environment, and, at the same time, w will have easy access 
to its source code to report and fix bugses. The ultimate version of this repo
will work with PIP based Network Diffusion, but we will change that after
experiments are prepared and all noted bugs solved.

Adding new submodule (in case of any new needed):
```
mkdir submodules
git submodule add https://github.com/anty-filidor/network_diffusion ./submodules/network_diffusion
ln -s submodules/network_diffusion/network_diffusion network_diffusion  
cd submodules/network_diffusion
git checkout fix-bdma-paper
```

Cloning submodule while cloning this repo (a default way to get library right now):
```
git submodule init
git submodule update
```

## Installation of env
```
conda create --name bdma -y python=3.10
conda activate bdma
pip install -r requirements.txt
pip install -r submodules/network_diffusion/requirements/production.txt
python -m ipykernel install --user --name bdma
```

## Suspected-Infected-Removed + Unaware-Aware
Example of propagation coexisting processes that influence each other. Details
of model in `sir_ua.drawio`. Run eperiments with `sir_ua.ipynb`.

## UTS model
<here a short description>

## Another models
To be done...
