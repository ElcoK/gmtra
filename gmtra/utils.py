# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:52:40 2018

@author: cenv0574
"""

import os
import sys
import rasterio as rio
import numpy as np
import geopandas as gpd
from osgeo import ogr
import shapely.wkt
import pandas as pd
import country_converter as coco
from collections import defaultdict
import shutil
from rasterstats import point_query

sys.path.append(os.path.join( '..'))
from miriam_py.utils import load_config

sys.path.append(os.path.join( '..'))

def clean_fluvial_dirs(hazard_path):
    """
    Remove all the data we do not use.
    """
    for root, dirs, files in os.walk(os.path.join(hazard_path,'InlandFlooding'), topdown=False):
        for name in dirs:
            if ('fluvial_defended' in name) or ('pluvial_defended' in name) or ('urban_defended' in name) or ('urban_mask' in name) or ('urban_undefended' in name):
                shutil.rmtree(os.path.join(root,name), ignore_errors=True)


def load_osm_data(data_path,country):
    """
    Load osm data for an entire country.
    """
    osm_path = os.path.join(data_path,'country_osm','{}.osm.pbf'.format(country))

    driver=ogr.GetDriverByName('OSM')
    return driver.Open(osm_path)

def load_osm_data_region(data_path,region):
    """
    Load osm data for a specific region.
    """    
    osm_path = os.path.join(data_path,'region_osm','{}.osm.pbf'.format(region))

    driver=ogr.GetDriverByName('OSM')
    return driver.Open(osm_path)

def load_hazard_map(EQ_path):

    with rio.open(EQ_path) as src:
        affine = src.affine
        # Read as numpy array
        array = src.read(1)
 
        array = np.array(array,dtype='int16')
    
    return array,affine

def load_ssbn_hazard(hazard_path,country_full,country_ISO2,flood_type,flood_type_abb,rp):

    flood_path = os.path.join(hazard_path,'InlandFlooding',country_full,'{}_{}_merged'.format(country_ISO2,flood_type),'{}-{}-{}.tif'.format(country_ISO2,flood_type_abb,rp))    

    with rio.open(flood_path) as src:
        affine = src.affine
        array = src.read(1)
        array[array == 999] = -9999
    
    return array,affine

def EQ_hazard(hazard_path,country_full,country_ISO2,flood_type,flood_type_abb,rp):

    flood_path = os.path.join(hazard_path,'InlandFlooding',country_full,'{}_{}'.format(country_ISO2,flood_type),'{}-{}-{}-1.tif'.format(country_ISO2,flood_type_abb,rp))    

    with rio.open(flood_path) as src:
        affine = src.affine
        array = src.read(1)
        array[array == 999] = -9999
    
    return array,affine

def gdf_clip(gdf,clip_geom):
    """
    Arguments:
        *gdf* : geopandas GeoDataFrame that we want to clip
        
         *clip_geom* : shapely geometry of region for what we do the calculation
               
    Returns:
        *gdf* : clipped geopandas GeoDataframe
    """
    return gdf.loc[gdf['geometry'].apply(lambda x: x.within(clip_geom))].reset_index(drop=True)

def get_raster_value(centroid,out_image,out_transform):
    return int(point_query(centroid,out_image,affine=out_transform,nodata=-9999,interpolate='nearest')[0] or 0)   


def extract_value_from_gdf(x,gdf_sindex,gdf,column_name):
    """
    Arguments:
        x : row of dataframe
        gdf_sindex : spatial index of dataframe of which we want to extract the value
        gdf : GeoDataFrame of which we want to extract the value
        column_name : column that contains the value we want to extract
        
    Returns:
        extracted value from other gdf
    """
    try:
        return gdf.loc[list(gdf_sindex.intersection(x.bounds))][column_name].values[0]
    except:
        return 0

def default_factory():
    return 'nodata'

def create_folder_lookup():

    data_path = load_config()['paths']['data']
    hazard_path =  load_config()['paths']['hazard_data']
    
    right_folder_name = defaultdict(default_factory,{'dr_congo':'democratic_republic_congo',
    'south_korea':'korea_south',
    'north_korea':'korea_north',
    'congo_republic':'congo',
    'bosnia_and_herzegovina':'bosnia',
    'turks_and_caicos_islands':'turks_caicos_islands',
    'sao_tome_and_principe':'sao_tome',
    'st._pierre_and_miquelon':'saint_pierre_miquelon',
    'st._lucia':'saint_lucia',
    'st._kitts_and_nevis':'saint_kitts_nevis',
    'cabo_verde':'cape_verde',
    'kyrgyz_republic':'kyrgyzstan',
    'tajikistan':'tadjikistan',
    'brunei_darussalam':'brunei',
    'kazakhstan':'kazachstan',
    'united_states':'united_states_of_america',
    'christmas_island':'christmas_islands',
    'bonaire,_saint_eustatius_and_saba':'bonaire_sint_saba',
    'st._vincent_and_the_grenadines':'saint_vincent_grenadines',
    'united_states_virgin_islands':'virgin_islands_usa',
    'south_georgia_and__south_sandwich_is.':'south_georgia',
    'norfolk_island':'norfolk_islands',
    'british_virgin_islands':'virgin_islands_british',
    'cocos_(keeling)_islands':'cocos_islands',
    'faeroe_islands':'faroe_islands',
    'wallis_and_futuna_islands':'wallis_futuna',
    'guinea-bissau':'guinea_bissau',
    'antigua_and_barbuda':'antigua',
    'botswana':'nodata',
    'norway':'nodata',
    'greenland':'nodata',
    'palestine':'nodata',
    'curacao':'nodata',
    'united_states_minor_outlying_islands':'nodata',
    'sint_maarten':'nodata',
    'svalbard_and_jan_mayen_islands':'nodata',
    'nauru':'nodata',
    'kiribati':'nodata',
    'tuvalu':'nodata',
    'timor-leste':'nodata',
    'french_polynesia':'nodata',
    'cook_islands':'nodata'})

    notmatchingfull = defaultdict(default_factory,{'GGY':'Guernsey','JEY':'Jersey','MAF':'Saint Martin','SDN':'Sudan','SSD':'South Sudan','XKO': 'Kosovo'})
    fullback = defaultdict(default_factory,{'Guernsey':'GGY','Jersey':'JEY','Saint Martin':'MAF','Sudan':'SUD','South Sudan':'SDS'})
    
    global_data = gpd.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    glob_data_full = [coco.convert(names=[x], to='name_short').lower().replace(' ','_').replace("'","") if x not in notmatchingfull 
                      else notmatchingfull[x] for x in list(global_data.ISO_3digit)]
    glob_data_full = ['micronesia' if x.startswith('micronesia') else str(x) for x in glob_data_full]
    glob_data_full = [x for x in glob_data_full if x != 'not_found']
    
    country_dataname = os.listdir(os.path.join(hazard_path,'InlandFlooding'))
    glob_name_folder = [right_folder_name[x] if x not in country_dataname else x for x in glob_data_full ]
    ISO3_lookup = [coco.convert(names=[x.replace('_',' ')], to='ISO3') if x not in fullback 
          else fullback[x] for x in glob_data_full]
    return dict(zip(ISO3_lookup,glob_name_folder))

def map_railway():
    """ 
    Mapping function to create a dictionary with an aggregated list of railway types. 
            
    Returns:
        *dictionary* : A dictionary with road types and their aggregated equivalent    
    """

    dict_map = defaultdict(default_factory,{
    "rail" : "primary_rail",
    "station" : "station",
    "platform_edge":"platform",        
    "platform" : "platform",
    "abandoned" : "disused",
    "razed": "disused",        
    "construction" : "construction",
    "disused" : "disused" ,
    "funicular" : "other" ,
    "light_rail" : "light_rail",
    "miniature" : "other",
    "narrow_gauge" : "other",
    "preserverd" : "other",
    "subway" : "subway",
    "tram" : "tram"
            
    })

    return dict_map

def map_roads():
    """ 
    Mapping function to create a dictionary with an aggregated list of road types. 
            
    Returns:
        *dictionary* : A dictionary with road types and their aggregated equivalent
        
    """

    dict_map = defaultdict(default_factory,{
    "disused" : "other",
    "dummy" : "other",
    "planned" : "other",
    "platform" : "other",
    "unsurfaced" : "track",
    "traffic_island" : "other",
    "razed" : "other",
    "abandoned" : "other",
    "services" : "track",
    "proposed" : "other",
    "corridor" : "track",
    "bus_guideway" : "other",
    "bus_stop" : "other",
    "rest_area" : "other",
    "yes" : "other",
    "trail" : "other",
    "escape" : "track",
    "raceway" : "other",
    "emergency_access_point" : "track",
    "emergency_bay" : "track",
    "construction" : "track",
    "bridleway" : "track",
    "cycleway" : "other",
    "footway" : "other",
    "living_street" : "tertiary",
    "path" : "track",
    "pedestrian" : "other",
    "primary" : "primary",
    "primary_link" : "primary",
    "residential" : "tertiary",
    "road" : "secondary",
    "secondary" : "secondary",
    "secondary_link" : "secondary",
    "service" : "tertiary",
    "steps" : "other",
    "tertiary" : "tertiary",
    "tertiary_link" : "tertiary",
    "track" : "track",
    "unclassified" : "tertiary",
    "trunk" : "primary",
    "motorway" : "primary",
    "trunk_link" : "primary",
    "motorway_link" : "primary"
    })
        
    return dict_map