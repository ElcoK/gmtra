![ITRC](https://www.itrc.org.uk/wp-content/themes/itrc-mistral/images/ITRC-mistral.png)

# Global Multihazard Transport Risk Analysis (GMTRA)

This repository provides the code to perform a global transport asset risk analysis for earthquakes, floods and cyclones. 

It also provides Jupyter Notebooks to reproduce the figures in Koks et al. (2019). 

## Data requirements
* All transport data is based on OpenStreetMap (OSM), which can be freely downloaded. The planet file used in Koks et al. (2019) is downloaded at July 17, 2018. However, to run the code, the latest planet.osm.pbf file can be used.
* Global earthquake and cyclone hazard data is available from the UNISDR Global Assessment Report 2015 data portal (https://risk.preventionweb.net). 
* In Koks et al. (2019), global fluvial and surface SSBN flood hazard data (May 2017 version) is used with the permission of [Fathom Global](http://www.fathom.global/). 
* The coastal flood maps are developed by the Joint Research Centre of the European Commission. 

## Prepare data paths

Copy `config.template.json` to `config.json` and edit the paths for data and
figures, for example:

```json
{
    "data_path": "/home/<user>/projects/GMTRA/data",
    "hazard_path": "/home/<user>/projects/GMTRA/hazard_data",
    "figure_path": "/home/<user>/projects/GMTRA/figures"
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
conda env create -f environment.yml
activate GMTRA

```


### License
Copyright (C) 2019 Elco Koks. All versions released under the [GNU Affero General Public License v3.0 license](LICENSE).
