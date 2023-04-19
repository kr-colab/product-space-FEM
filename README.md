# Product Space FEM

Finite Element Method performed on domains structured
as a Cartesian product of smaller dimensional domains.
Assembly of the linear system uses FEniCS for integration.

## Requirements

A conda environment can be created with
```
mamba env create -f requirements/conda-environment.yml
```
and then activated with
```
mamba activate psf
```

## Development

Install the conda environment
```
mamba env create -f requirements/ci-environment.yml
mamba activate anaconda-client-env
```
and then activate it. The tests are at
```
python3 -m pytest tests
```

A very simple test workflow (with all parameters set so it runs pretty fast)
can be run with:
```
./test.sh
```

## Troubleshooting

In some systems, `quarto` fails due to the temporary directories not existing,
with an error like `ERROR: PermissionDenied: Permission denied (os error 13), mkdir '/run/user/1960/jt'`.
As a workaround on linux-based systems, run
```
export XDG_RUNTIME_DIR=$(mktemp -d)
```
(... and, don't forget to delete this at the end of the job.)
