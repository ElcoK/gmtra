# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:19:30 2019

@author: cenv0574
"""

import geopandas
import shapely.wkt

from utils import load_osm_data_region,load_osm_data

def roads(data_path,area_name,regional=False):
    """
    Function to extract all road assets from an .osm.pbf file. Will include bridges as well.
    
    Arguments:
        *data_path* : file path to location of all data.
        
        *area_name*: Admin code of the countryor region for which we want to extract the roads.
        
    Optional Arguments:
        *regional* : Set to True if we want to extract a region.
    """
    
    if regional == True:
        data = load_osm_data_region(data_path,area_name)
    else:
        data = load_osm_data(data_path,area_name)
       
    
    sql_lyr = data.ExecuteSQL("SELECT osm_id,highway FROM lines WHERE highway IS NOT NULL")
    
    roads=[]
    for feature in sql_lyr:
        if feature.GetField('highway') is not None:
            osm_id = feature.GetField('osm_id')
            shapely_geo = shapely.wkt.loads(feature.geometry().ExportToWkt()) 
            if shapely_geo is None:
                continue
            highway=feature.GetField('highway')
            roads.append([osm_id,highway,shapely_geo])
    
    if len(roads) > 0:
        return geopandas.GeoDataFrame(roads,columns=['osm_id','infra_type','geometry'],crs={'init': 'epsg:4326'})
    else:
        print('No roads in {}'.format(area_name))

def railway(data_path,country,regional=True):
    """
    Function to extract all railway assets from an .osm.pbf file. Will include bridges as well.
    
    Arguments:
        *data_path* : file path to location of all data.
        
        *area_name*: Admin code of the countryor region for which we want to extract the roads.
        
    Optional Arguments:
        *regional* : Set to True if we want to extract a region.
    """
        
    if regional == True:
        data = load_osm_data_region(data_path,country)
    else:
        data = load_osm_data(data_path,country)
           
    sql_lyr = data.ExecuteSQL("SELECT osm_id,service,railway FROM lines WHERE railway IS NOT NULL")
    
    roads=[]
    for feature in sql_lyr:
        if feature.GetField('railway') is not None:
            osm_id = feature.GetField('osm_id')
            service = feature.GetField('service')
            shapely_geo = shapely.wkt.loads(feature.geometry().ExportToWkt()) 
            if shapely_geo is None:
                continue
            railway=feature.GetField('railway')
            roads.append([osm_id,railway,service,shapely_geo])
    
    if len(roads) > 0:
        return geopandas.GeoDataFrame(roads,columns=['osm_id','infra_type','service','geometry'],crs={'init': 'epsg:4326'})
    else:
        print('No railway in {}'.format(country))
