# Global Multihazard Transport Risk Analysis (GMTRA)

This repository provides the code to perform a global transport asset risk analysis for earthquakes, floods and cyclones. 

It also provides all the code to reproduce the figures in Koks et al. (2019). 

## Prepare data paths

Copy `config.template.json` to `config.json` and edit the paths for data and
figures, for example:

```json
{
    "data_path": "/home/<user>/projects/GMTRA/data",
    "figures_path": "/home/<user>/projects/GMTRA/figures"
}
```

## Python requirements

Recommended option is to use a [miniconda](https://conda.io/miniconda.html)
environment to work in for this project, relying on conda to handle some of the
trickier library dependencies.

```bash

# Add conda-forge channel for extra packages
conda config --add channels conda-forge

# Create a conda environment for the project and install packages
conda env create -f .environment.yml
activate GMTRA

```


### License
Copyright (C) 2019 Elco Koks. All versions released under the [GNU Affero General Public License v3.0 license](LICENSE).

![ITRC](https://www.itrc.org.uk/wp-content/themes/itrc-mistral/images/ITRC-mistral.png)
