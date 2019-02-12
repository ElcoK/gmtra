
=======================================
2. OSM and Python
=======================================

To be able to extract the data from OpenStreetMap (OSM), a few steps and downloads are required.


Add attributes to osmconf.ini
-----------------------------
We need to add a few attributes to the osmconf.ini file to be able to extract everything we want.

1. Find the location of the osmconf.ini of the conda enviroment you are working in. It is generally located here:

**Windows**:

.. code-block:: bash

	%USERPROFILE%\AppData\Local\Continuum\miniconda3\envs\py36\Library\share\gdal

**Linux**:

.. code-block:: bash

	/home/<user>/.conda/envs/py36/share/gdal

	
2. On line 48 (the attributes of **[lines]**), add **railway, bridge, service**, when they are not there yet. 

**Note**: make sure they are not in the list twice! This will cause problems.
	
	
Download osmconvert
-------------------
Osmconvert is a small tool that can be used to clip OSM data.

1. Download **osmconvert64** from http://wiki.openstreetmap.org/wiki/Osmconvert. 
2. Create a new directory in your working directory (where all the data is stored as well), called **osmconvert**
3. Move to the tool into this directory. 