# Hydro-GRNNI: Hydrological Graph Recurrent Neural Network for Imputation

[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)](https://arxiv.org/submit/5840053/view)
[![arXiv](https://img.shields.io/badge/arXiv-6666.66666-b31b1b.svg?style=flat-square)](https://arxiv.org/submit/5840053/view)

This repository contains the code to replicate the experiments from the paper "Hydro-GRNNI: Hydrological Graph Recurrent Neural Network for Imputation." The paper presents a Graph Recurrent Neural Network (GRNN) that leverages river network structures and spatiotemporal data to enhance data resolution and accuracy.

**Author**: [Shuo Han](shuohan2024.1@u.northwestern.edu)

---

<h2 align=center>Inro to Hydro-GRNNI</h2>

This study introduces Hydro-GRNNI, an innovative method that improves the resolution of spatial and temporal hydrological data by combining river flow direction information with spatiotemporal inputs using a Graph Recurrent Neural Network (GRNN). By constructing a physical flow direction graph, Hydro-GRNNI establishes directional relationships among river monitoring stations and utilizes spatiotemporal encoders for accurate data imputation.

---

## Directory structure

The directory is structured as follows:

```
.
├── config
│   ├── bimpgru
│   ├── brits
│   ├── rgain
│   ├── mpgru
│   ├── var
│   ├── grin
│   └── hydro
├── datasets
│   └── flow_dis
├── lib
│   ├── __init__.py
│   ├── data
│   ├── datasets
│   ├── fillers
│   ├── nn
│   └── utils
├── conda_env.yml
└── scripts
    ├── run_baselines.py
    ├── run_baselines2.py
    └── run_grnni.py

```

## Configuration files

The `config` directory contains model-specific configuration files. Each subfolder corresponds to a different model configuration used in the experiments.

## Library Code

The core functionality, including model implementations and dataset loaders, is housed in the `lib` directory. If dataset paths need adjustment, modify the `__init__.py` file within this directory.

## Script Files

All scripts for running the experiments are located in the `scripts` directory:

* `run_baselines.py`：Calculates metrics for the `MEAN` and `KNN` imputation methods. Example usage:

	```
	python ./scripts/run_baselines.py --datasets flo --in-sample True
	```
 * `run_run_baselines2.py`: Computes metrics for `VAR`, `rGAIN`, `BRITS`, `MPGRU` and `GRIN` imputation methods. Example usage:

	```
	python ./scripts/run_baselines2.py --config config/grin/flo-dis.yaml --in-sample False
	```
* `run_grnni.py`: Runs the evaluation for the `Hydro-GRNNI` model. Example usage:

	```
	python ./scripts/run_grnni.py --config config/grnni/flo-dis.yaml --in-sample False
	```

## Requirements

The experiments are conducted using Python 3.8. The required packages are listed in `requirements.txt` and `conda_env.yml`.

## Bibtex Citation

If you find this code useful, we would greatly appreciate it if you could cite our work:

```
@mastersthesis{shuo2024hydro,
    title={Hydro-GRNNI: Hydrological Graph Recurrent Neural Network for Imputation},
    author={Shuo Han},
    school={Northwestern University},
    year={2024},
    url={https://arxiv.org/submit/5840053/view}
}
```

