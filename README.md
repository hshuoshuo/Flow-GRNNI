# Hydro-GRNNI: Hydrological Graph Recurrent Neural Network for Imputation - [pdf](https://openreview.net/pdf?id=kOu3-S3wJ7))

[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)](https://openreview.net/pdf?id=kOu3-S3wJ7)
[![arXiv](https://img.shields.io/badge/arXiv-2108.00298-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2108.00298)

This repository contains the code to replicate the experiments from the paper "Hydro-GRNNI: Hydrological Graph Recurrent Neural Network for Imputation." The paper presents a Graph Recurrent Neural Network (GRNN) that leverages river network structures and spatiotemporal data to enhance data resolution and accuracy.

**Authors**: [Shuo Han](shuohan2024.1@u.northwestern.edu)

---

<h2 align=center>Inro to Hydro-GRNNI</h2>

This study introduces Hydro-GRNNI, an innovative method that improves the resolution of spatial and temporal hydrological data by combining river flow direction information with spatiotemporal inputs using a Graph Recurrent Neural Network (GRNN). By constructing a physical flow direction graph, Hydro-GRNNI establishes directional relationships among river monitoring stations and utilizes spatiotemporal encoders for accurate data imputation.

<p align=center>
  <a href="https://github.com/marshka/sinfony">
    <img src="./grin.png" alt="Logo"/>
  </a>
</p>

---

## Directory structure

The directory is structured as follows:

```
.
├── config
│   ├── bimpgru
│   ├── brits
│   ├── grin
│   ├── mpgru
│   ├── rgain
│   └── var
├── datasets
│   ├── air_quality
│   ├── metr_la
│   ├── pems_bay
│   └── synthetic
├── lib
│   ├── __init__.py
│   ├── data
│   ├── datasets
│   ├── fillers
│   ├── nn
│   └── utils
├── requirements.txt
└── scripts
    ├── run_baselines.py
    ├── run_imputation.py
    └── run_synthetic.py

```
Note that, given the size of the files, the datasets are not readily available in the folder. See the next section for the downloading instructions.

## Datasets

We regret that the dataset used in our experiments is not open-source. However, you can adapt any suitable dataset to be compatible with our method.

## Configuration files

The `config` directory stores all the configuration files used to run the experiment. They are divided into folders, according to the model.

## Library

The support code, including the models and the datasets readers, are packed in a python library named `lib`. Should you have to change the paths to the datasets location, you have to edit the `__init__.py` file of the library.

## Scripts

The scripts used for the experiment in the paper are in the `scripts` folder.

* `run_baselines.py` is used to compute the metrics for the `MEAN`, `KNN`, `MF` and `MICE` imputation methods. An example of usage is

	```
	python ./scripts/run_baselines.py --datasets air36 air --imputers mean knn --k 10 --in-sample True --n-runs 5
	```

* `run_imputation.py` is used to compute the metrics for the deep imputation methods. An example of usage is

	```
	python ./scripts/run_imputation.py --config config/grin/air36.yaml --in-sample False
	```

* `run_synthetic.py` is used for the experiments on the synthetic datasets. An example of usage is

	```
	python ./scripts/run_synthetic.py --config config/grin/synthetic.yaml --static-adj False
	```

## Requirements

We run all the experiments in `python 3.8`, see `requirements.txt` for the list of `pip` dependencies.

## Bibtex Citation

Should you find this code beneficial, we would greatly appreciate it if you could cite our work:

```
@mastersthesis{shuo2024hydro,
    title={Hydro-GRNNI: Hydrological Graph Recurrent Neural Network for Imputation},
    author={Shuo Han},
    school={Northwestern University},
    year={2024},
    url={https://openreview.net/forum?id=kOu3-S3wJ7}
}
```

