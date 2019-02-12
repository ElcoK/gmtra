"""
Source code for Global Multihazard Transport Risk Analysis (GMTRA)

Functions to extract infrastructure asset data from OpenStreetMap.

Copyright (C) 2019 Elco Koks. All versions released under the GNU Affero General Public License v3.0 license.
"""

import geopandas
import shapely.wkt

from gmtra.utils import load_osm_data_region,load_osm_data

def roads(data_path,area_name,regional=False):
    """
    Function to extract all road assets from an .osm.pbf file. 
    
    Arguments:
        *data_path* : file path to location of all data.
        
        *area_name*: Admin code of the countryor region for which we want to extract the roads.
        
    Optional Arguments:
        *regional* : Set to True if we want to extract a region.
    """
    
    # open OpenStreetMap file
    if regional == True:
        data = load_osm_data_region(data_path,area_name)
    else:
        data = load_osm_data(data_path,area_name)
    
    roads=[]    
    if data is not None:

        # perform SQL query on the OpenStreetMap data
        sql_lyr = data.ExecuteSQL("SELECT osm_id,highway FROM lines WHERE highway IS NOT NULL")
        for feature in sql_lyr:
            try:
                if feature.GetField('highway') is not None:
                    osm_id = feature.GetField('osm_id')
                    shapely_geo = shapely.wkt.loads(feature.geometry().ExportToWkt()) 
                    if shapely_geo is None:
                        continue
                    highway=feature.GetField('highway')
                    roads.append([osm_id,highway,shapely_geo])
            except:
                    print("WARNING: skipped a road")
    else:
        print("ERROR: Nonetype error when requesting SQL. Check required.")    

    if len(roads) > 0:
        return geopandas.GeoDataFrame(roads,columns=['osm_id','infra_type','geometry'],crs={'init': 'epsg:4326'})
    else:
        print("WARNING: No roads or No Memory. returning empty GeoDataFrame") 
        return geopandas.GeoDataFrame(columns=['osm_id','infra_type','geometry'],crs={'init': 'epsg:4326'})

def railway(data_path,country,regional=True):
    """
    Function to extract all railway assets from an .osm.pbf file. 
    
    Arguments:
        *data_path* : file path to location of all data.
        
        *area_name*: Admin code of the countryor region for which we want to extract the roads.
        
    Optional Arguments:
        *regional* : Set to True if we want to extract a region.
    """
    # open OpenStreetMap file
    if regional == True:
        data = load_osm_data_region(data_path,country)
    else:
        data = load_osm_data(data_path,country)
           
    railways=[]    
    if data is not None:

        # perform SQL query on the OpenStreetMap data
        sql_lyr = data.ExecuteSQL("SELECT osm_id,service,railway FROM lines WHERE railway IS NOT NULL")
        
        for feature in sql_lyr:
            try:
                if feature.GetField('railway') is not None:
                    osm_id = feature.GetField('osm_id')
                    shapely_geo = shapely.wkt.loads(feature.geometry().ExportToWkt()) 
                    if shapely_geo is None:
                        continue
                    railway=feature.GetField('railway')
                    railways.append([osm_id,railway,shapely_geo])
            except:
                    print("warning: skipped a railway")
    else:
        print("ERROR: Nonetype error when requesting SQL. Check required.")    

    if len(railways) > 0:
        return geopandas.GeoDataFrame(railways,columns=['osm_id','infra_type','geometry'],crs={'init': 'epsg:4326'})
    else:
        print("WARNING: No railway or No Memory. returning empty GeoDataFrame") 
        return geopandas.GeoDataFrame(columns=['osm_id','infra_type','geometry'],crs={'init': 'epsg:4326'})
        
def bridges(data_path,area_name,regional=True):
    """
    Function to extract all bridges from an .osm.pbf file. 
    
    Arguments:
        *data_path* : file path to location of all data.
        
        *area_name*: Admin code of the countryor region for which we want to extract the roads.
        
    Optional Arguments:
        *regional* : Set to True if we want to extract a region.
    """    
    
    # open OpenStreetMap file
    if regional == True:
        data = load_osm_data_region(data_path,area_name)
    else:
        data = load_osm_data(data_path,area_name)
           
     # extract all bridges from the .osm.pbf file
    bridges=[]
    if data is not None:
       
        # perform SQL query on the OpenStreetMap data
        sql_lyr = data.ExecuteSQL("SELECT osm_id,bridge,highway,railway FROM lines WHERE bridge IS NOT NULL")
    
        for feature in sql_lyr:
            try:
                if feature.GetField('bridge') is not None:
                    osm_id = feature.GetField('osm_id')
                    bridge = feature.GetField('bridge')
                    highway = feature.GetField('highway')
                    railway = feature.GetField('railway')
                    shapely_geo = shapely.wkt.loads(feature.geometry().ExportToWkt()) 
                    if shapely_geo is None:
                        continue
                    bridges.append([osm_id,bridge,highway,railway,shapely_geo])
            except:
                    print("WARNING: skipped a bridge")
    else:
        print("ERROR: Nonetype error when requesting SQL. Check required.")        

    if len(bridges) > 0:
        return geopandas.GeoDataFrame(bridges,columns=['osm_id','bridge','road_type','rail_type','geometry'],crs={'init': 'epsg:4326'})
    else:
        print("WARNING: No railway or No Memory. returning empty GeoDataFrame") 
        return geopandas.GeoDataFrame(columns=['osm_id','bridge','road_type','rail_type','geometry'],crs={'init': 'epsg:4326'})
