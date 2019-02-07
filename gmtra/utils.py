"""
Source code for Global Multihazard Transport Risk Analysis (GMTRA)

General functions to be used throughout the code.

Copyright (C) 2019 Elco Koks. All versions released under the GNU Affero General Public License v3.0 license.
"""

import os
import json
import rasterio as rio
import numpy as np
import geopandas as gpd
from osgeo import ogr
import country_converter as coco
from collections import defaultdict
import shutil
from scipy import integrate
from geopy.distance import vincenty
from boltons.iterutils import pairwise
from rasterstats import point_query

def load_config():
    """Read config.json
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    return config

def clean_fluvial_dirs(hazard_path):
    """
    Remove all the data we do not use.
    
    Arguments:
        *hazard_path* : file path to location of all hazard data.
    
    """
    for root, dirs, files in os.walk(os.path.join(hazard_path,'InlandFlooding'), topdown=False):
        for name in dirs:
            if ('fluvial_defended' in name) or ('pluvial_defended' in name) or ('urban_defended' in name) or ('urban_mask' in name) or ('urban_undefended' in name):
                shutil.rmtree(os.path.join(root,name), ignore_errors=True)


def load_osm_data(data_path,country):
    """
    Load osm data for an entire country.
    
    Arguments:
        *data_path* : file path to location of all data.
        
        *country* : unique ID of the country for which we want to extract data from 
        OpenStreetMap. Must be matching with the country ID used for saving the .osm.pbf file.

    """
    osm_path = os.path.join(data_path,'country_osm','{}.osm.pbf'.format(country))

    driver=ogr.GetDriverByName('OSM')
    return driver.Open(osm_path)

def load_osm_data_region(data_path,region):
    """
    Load osm data for a specific region.
    
    Arguments:
        *data_path* : file path to location of all data.
        
        *region* : unique ID of the region for which we want to extract data from 
        OpenStreetMap. Must be matching with the region ID used for saving the .osm.pbf file.
        
    """    
    osm_path = os.path.join(data_path,'region_osm','{}.osm.pbf'.format(region))

    driver=ogr.GetDriverByName('OSM')
    return driver.Open(osm_path)

def load_hazard_map(hzd_path):
    """
    Load specific hazard map.
    
    Arguments:
        *hzd_path* : file path to location of the hazard map.
        
    """    
    # open hazard map using rasterio
    with rio.open(hzd_path) as src:
        affine = src.affine
        # Read as numpy array
        array = src.read(1)
 
        array = np.array(array,dtype='int16')
    
    return array,affine

def load_ssbn_hazard(hazard_path,country_full,country_ISO2,flood_type,flood_type_abb,rp):
    """
    Function to load a SSBN hazard map.
    
    Arguments:
        *hazard_path* : Path to location of all hazard data.
            
        *country_full* : Full name of country. Obtained from **create_folder_lookup**. 
            
        *country_ISO2* : ISO2 country code of the country.
            
        *flood_type* : Specifies whether it is a **pluvial** or **fluvial** flood.
            
        *flood_type_abb* : Abbrevated code of the flood type. **FU** for river flooding, **PU** for surface flooding.
            
        *rp* : Return period of the flood map we want to extract.
        
    Returns:
        *array*: NumPy Array with the raster values.
        
        *affine* : Affine of the **array**.
        
    """
    
    # specify path to the hazard map
    flood_path = os.path.join(hazard_path,'InlandFlooding',country_full,'{}_{}_merged'.format(country_ISO2,flood_type),'{}-{}-{}.tif'.format(country_ISO2,flood_type_abb,rp))    

    # open hazard map using rasterio
    with rio.open(flood_path) as src:
        affine = src.affine
        array = src.read(1)
        
        # change value 999 (waterways and bodies) to -9999 (nodata).
        array[array == 999] = -9999
    
    return array,affine


def gdf_clip(gdf,clip_geom):
    """
    Function to clip a GeoDataFrame with a shapely geometry.
    
    Arguments:
        *gdf* : geopandas GeoDataFrame that we want to clip.
        
        *clip_geom* : shapely geometry of region for which we do the calculation.
               
    Returns:
        *gdf* : clipped geopandas GeoDataframe
    """
    return gdf.loc[gdf['geometry'].apply(lambda x: x.within(clip_geom))].reset_index(drop=True)

def sum_tuples(l):
    """
    Function to sum a list of tuples.
    
    Arguments:
        *l* : list of tuples.
        
    Returns:
        *tuple* : a tuple with the sum of the list of tuples.
    
    """
    return tuple(sum(x) for x in zip(*l))

def sensitivity_risk(RPS,loss_list):
    """
    Function to estimate the monetary risk for a particular hazard within the sensitivity analysis.
    
    Arguments:
        *RPS* : list of return periods in floating probabilities (i.e. [1/10,1/20,1/50]).
        
        *loss_list* : list of lists with a monetary value per return period within each inner list.
        
    Returns:
        *collect_risks* : a list of all risks for each inner list of the input list.
    """

    collect_risks = []
    for y in range(50):
        collect_risks.append(integrate.simps([x[y] for x in loss_list][::-1], x=RPS[::-1]))
    return collect_risks


def monetary_risk(RPS,loss_list):
    """
    Function to estimate the monetary risk for a particular hazard.
    
    Arguments:
        *RPS* : list of return periods in floating probabilities (i.e. [1/10,1/20,1/50]).
        
        *loss_list* : list of lists with a monetary value per return period within each inner list.
        
    Returns:
        *collect_risks* : a list of all risks for each inner list of the input list.
    """
    collect_risks = []
    for y in range(7):
        collect_risks.append(integrate.simps([x[y] for x in loss_list][::-1], x=RPS[::-1]))
    return collect_risks

def exposed_length_risk(x,hzd,RPS):
    """
    Function to estimate risk in terms of exposed kilometers.
    
    Arguments:
        *x* : row in a GeoDataFrame that represents an unique infrastructure asset.
        
        *hzd* : abbrevation of the hazard we want to intersect. **EQ** for earthquakes,
        **Cyc** for cyclones, **FU** for river flooding, **PU** for surface flooding
        and **CF** for coastal flooding.
        
        *RPS* : list of return periods in floating probabilities (i.e. [1/10,1/20,1/50]). 
        Should match with the hazard we are considering.
        
    Returns:
        *risk value* : a floating number which represents the annual exposed kilometers of infrastructure.    
    """
    if hzd == 'EQ':
        return integrate.simps([x.length_EQ_rp250,x.length_EQ_rp475,x.length_EQ_rp975,x.length_EQ_rp1500,x.length_EQ_rp2475][::-1], x=RPS[::-1])
    elif hzd == 'Cyc':
        return integrate.simps([x.length_Cyc_rp50,x.length_Cyc_rp100,x.length_Cyc_rp250,x.length_Cyc_rp500,x.length_Cyc_rp1000][::-1], x=RPS[::-1])
    elif hzd == 'FU':
        return integrate.simps([x['length_FU-5'],x['length_FU-10'],x['length_FU-20'],x['length_FU-50'],x['length_FU-75'],x['length_FU-100'],
                                x['length_FU-200'],x['length_FU-250'],x['length_FU-500'],x['length_FU-1000']][::-1], x=RPS[::-1])
    elif hzd == 'PU':
        return integrate.simps([x['length_PU-5'],x['length_PU-10'],x['length_PU-20'],x['length_PU-50'],x['length_PU-75'],x['length_PU-100'],
                                x['length_PU-200'],x['length_PU-250'],x['length_PU-500'],x['length_PU-1000']][::-1], x=RPS[::-1])
    elif hzd == 'CF':
        return integrate.simps([x['length_CF-10'],x['length_CF-20'],x['length_CF-50'],x['length_CF-100'],x['length_CF-200'],x['length_CF-500'],
                                x['length_CF-1000']][::-1], x=RPS[::-1])

def total_length_risk(x,RPS):
    """
    Function to estimate risk if all assets would have been exposed.
    
    Arguments:
        *x* : row in a GeoDataFrame that represents an unique infrastructure asset.
        
        *RPS* : list of return periods in floating probabilities (i.e. [1/10,1/20,1/50]). 
        Should match with the hazard we are considering.
    """
    return integrate.simps([x.length]*len(RPS), x=RPS[::-1])

def square_m2_cost_range(x):
    """
    Function to specify the range of possible costs for a bridge.
    
    Arguments:
        *x* : row in a GeoDataFrame that represents an unique bridge asset.
        
    Returns:
        *list*: a list with the range of possible bridge costs.
    """
    
    # specify range of cost values for different bridge lengths.
    short_bridge = [int(10.76*115),int(10.76*200)]
    medium_bridge = [int(10.76*85),int(10.76*225)]
    long_bridge = [int(10.76*85),int(10.76*225)]

    # and return this range based on the bridge length
    if (x.length > 6) & (x.length <= 30):
        return short_bridge
    elif (x.length > 30) & (x.length <= 100):
        return medium_bridge
    elif x.length >= 100:
        return long_bridge

def extract_value_from_gdf(x,gdf_sindex,gdf,column_name):
    """
    Arguments:
        *x* : row in a geopandas GeoDataFrame.
        *gdf_sindex* : spatial index of dataframe of which we want to extract the value.
        
        *gdf* : GeoDataFrame of which we want to extract the value.
        
        *column_name* : column that contains the value we want to extract.
        
    Returns:
        extracted value from other GeoDataFrame
    """
    try:
        return gdf.loc[list(gdf_sindex.intersection(x.bounds))][column_name].values[0]
    except:
        return 0


def get_raster_value(centroid,raster_image,out_transform):
    """
    Function to extract a value from a raster file for a given shapely Point.
    
    Arguments:
        *centroid* : shapely Point
        
        *raster_image* : NumPy array, read by rasterio.
        
        *out_transform* : Affine of **raster_image*.
        
    Returns:
        
        *int* : integer value from the raster at the location of the **centroid**. Will return **zero** when there is no intersection.
    
    """
    return int(point_query(centroid,raster_image,affine=out_transform,nodata=-9999,interpolate='nearest')[0] or 0)   

def default_factory():
    return 'nodata'

def create_folder_lookup():
    """
    Function to create a dictionary in which we can lookup the 
    folder path where the surface and river flood maps are located for a country.
    """

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

    # first catch several countries that are not in the data
    notmatchingfull = defaultdict(default_factory,{'GGY':'Guernsey','JEY':'Jersey','MAF':'Saint Martin','SDN':'Sudan','SSD':'South Sudan','XKO': 'Kosovo'})
    fullback = defaultdict(default_factory,{'Guernsey':'GGY','Jersey':'JEY','Saint Martin':'MAF','Sudan':'SUD','South Sudan':'SDS'})
    
    # load data files and change some of the names to make them matching
    global_data = gpd.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    glob_data_full = [coco.convert(names=[x], to='name_short').lower().replace(' ','_').replace("'","") if x not in notmatchingfull 
                      else notmatchingfull[x] for x in list(global_data.ISO_3digit)]
    glob_data_full = ['micronesia' if x.startswith('micronesia') else str(x) for x in glob_data_full]
    glob_data_full = [x for x in glob_data_full if x != 'not_found']
    
    # and create a dictioniary that matches ISO3 codes with the country name datapaths for the FATHOM data.
    country_dataname = os.listdir(os.path.join(hazard_path,'InlandFlooding'))
    glob_name_folder = [right_folder_name[x] if x not in country_dataname else x for x in glob_data_full ]
    ISO3_lookup = [coco.convert(names=[x.replace('_',' ')], to='ISO3') if x not in fullback 
          else fullback[x] for x in glob_data_full]
    return dict(zip(ISO3_lookup,glob_name_folder))

def line_length(line, ellipsoid='WGS-84'):
    """Length of a line in meters, given in geographic coordinates

    Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

    Arguments:
        *line* : a shapely LineString object with WGS-84 coordinates
        
    Optional Arguments:
        *ellipsoid* : string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance)
        
    Returns:
        Length of line in meters
    """
    if line.geometryType() == 'MultiLineString':
        return sum(line_length(segment) for segment in line)

    return sum(
        vincenty(a, b, ellipsoid=ellipsoid).kilometers
        for a, b in pairwise(line.coords)
    )

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