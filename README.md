# Product Space FEM

Finite Element Method performed on domains structured
as a Cartesian product of smaller dimensional domains.
Assembly of the linear system uses FEniCS for integration.

## Requirements

A conda environment can be created with
```
conda env create -f requirements/conda-environment.yml
```
and then activated with
```
conda activate psf
```

## Development

Install the conda environment
```
conda env create -f requirements/CI-environment.yml
```
and then activate it. The tests are at
```
python3 -m pytest tests
```
