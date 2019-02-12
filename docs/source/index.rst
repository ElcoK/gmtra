.. gmtra documentation master file, created by
   sphinx-quickstart on Tue Feb  5 17:23:52 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/itrc_mistral.png
    :align: left   
|
|
|
 
Global Multihazard Transport Risk Analysis (GMTRA)
==================================================

This is the documentation of the code to perform a global transport asset risk analysis for earthquakes, floods and cyclones. 

Data requirements
-----------------
* All transport data is based on OpenStreetMap (OSM), which can be freely downloaded. The planet file used in Koks et al. (in review) is downloaded at July 17, 2018. However, to run the code, the latest planet.osm.pbf file can be used.
* Global earthquake and cyclone hazard data is available from the UNISDR Global Assessment Report 2015 data portal (https://risk.preventionweb.net). 
* In Koks et al. (in review), global fluvial and surface SSBN flood hazard data (May 2017 version) is used with the permission of Fathom Global (http://www.fathom.global/). 
* The coastal flood maps are developed by the Joint Research Centre of the European Commission. 

Prepare data paths
---------------------
Copy `config.template.json` to `config.json` and edit the paths for data and
figures, for example:

.. code-block:: bash

	{
		"data_path": "/home/<user>/projects/GMTRA/data",
		"hazard_path": "/home/<user>/projects/GMTRA/hazard_data",
		"figure_path": "/home/<user>/projects/GMTRA/figures"
	}
	
Python requirements
-------------------
Recommended option is to use a [miniconda](https://conda.io/miniconda.html)
environment to work in for this project, relying on conda to handle some of the
trickier library dependencies.

.. code-block:: bash
 
	# Add conda-forge channel for extra packages
	conda config --add channels conda-forge

	# Create a conda environment for the project and install packages
	conda env create -f environment.yml
	activate GMTRA


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   workflow
   osmtutorial
   utils
   preprocessing
   fetch
   hazard
   exposure
   damage
   sensitivity
   parallel
   
