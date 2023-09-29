# Network Diffusion – Framework to Simulate Spreading Processes in Complex Networks

This repository contains code used to perform experiments and analyse results 
that have been attached to a paper submitted to 
[Big Data Mining and Analytics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8254253).

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

## Installation of environment
```
conda create --name bdma -y python=3.10
conda activate bdma
pip install -r requirements.txt
pip install -r submodules/network_diffusion/requirements/production.txt
python -m ipykernel install --user --name bdma
```

## Suspected-Infected-Removed + Unaware-Aware
Example of propagation two coexisting processes that influence each other: 
disease (following SIR model) and awarenes (following UA model). Perform 
experiments from `sir_ua.ipynb`. Details of model and results in diretory `sir_ua`.

## Multilayer Independent Cascade Model initialised with Minimum Dominating Set
Example of propagation of Independent Cascade Model in multilayer networks with
various seed selection methods. Especially a comparision of Minimum Dominating Set
(a concept from domain of netork controlability) with centrality metrics.

## Temporal Network Epistemology Model + CogSNet / Static Network
Comparison of a spreading of the Temporal Network Epistemology on two different
network models built from the same temporal edge list. Preform experiments with
`tnem_cogsnet.ipynb`. Results will be saved in `tnem_cogsnet` directory.

## Temporal Linear Threshold Model + CogSNet / Static Network
Comparison of a spreading of the Temporal Linear Threshold Model on two different
network models built from the same temporal edge list. Preform experiments with
`mltm_cogsnet.ipynb`. Results will be saved in `mltm_cogsnet` directory.
