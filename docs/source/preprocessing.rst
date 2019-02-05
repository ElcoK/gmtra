
==========================
3. Preprocessing functions
==========================

Download planet OSM
-------------------
.. autofunction:: gmtra.preprocessing.planet_osm

Create osm.pbf files for a single country
-----------------------------------------
.. autofunction:: gmtra.preprocessing.single_country

Create osm.pbf files for all specified countries
------------------------------------------------
.. autofunction:: gmtra.preprocessing.all_countries

Create global shapefiles
-------------------------
.. autofunction:: gmtra.preprocessing.global_shapefiles

Remove tiny shapes from large multipolygons
------------------------------------------------
.. autofunction:: gmtra.preprocessing.remove_tiny_shapes

Create .poly files
------------------------------------------------
.. autofunction:: gmtra.preprocessing.poly_files

Clip a region from a larger .osm.pbf file
------------------------------------------------
.. autofunction:: gmtra.preprocessing.clip_osm

Merge SSBN maps within a country
------------------------------------------------
.. autofunction:: gmtra.preprocessing.merge_SSBN_maps

Merge SSBN maps for all countries
------------------------------------------------
.. autofunction:: gmtra.preprocessing.run_SSBN_merge

Extract bridges from OpenStreetMap
------------------------------------------------
.. autofunction:: gmtra.preprocessing.region_bridges