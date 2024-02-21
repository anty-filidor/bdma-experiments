# Network Diffusion – Framework to Simulate Spreading Processes in Complex Networks

This repository contains code used to perform experiments and analyse results 
that have been attached to a paper published at 
[Big Data Mining and Analytics](https://doi.org/10.26599/BDMA.2024.9020010).

The Network Diffusion package (a backbone of the expermients) is available on
[PyPI](https://pypi.org/project/network-diffusion/) and
[GitHub](https://github.com/anty-filidor/network_diffusion).

## Installation of environment
```bash
conda create --name bdma -y python=3.10
conda activate bdma
pip install -r requirements.txt
pip install network_diffusion==0.13.0
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

## Efficiency test
Comparison of time efficiency of models used in experiments. Execute them with 
`efficiency_test.ipynb`. Results will be saved in `efficiency_test` directory.

## Citing

If you found the paper or the codebase useful in your research, please consider
citing us as follows:

```latex
@article{czuba2024networkdiffusion,
  title={Network Diffusion – Framework to Simulate Spreading Processes in Complex Networks},
  author={
    Czuba, Micha{\l} and Nurek, Mateusz and Serwata, Damian and Qi, Yu-Xuan and
    Jia, Mingshan and Musial, Katarzyna and Michalski, Rados{\l}aw and Br{\'o}dka, Piotr
  },
  journal={Big Data Mining And Analytics},
  volume={},
  number={},
  pages={1-13},
  year={2024},
  publisher={IEEE},
  doi = {10.26599/BDMA.2024.9020010},
  url={https://doi.org/10.26599/BDMA.2024.9020010},
}
```
