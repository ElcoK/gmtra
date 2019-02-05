# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:55:09 2018

@author: cenv0574
"""

import os
import numpy
import pandas
import geopandas
import country_converter as coco
import urllib.request
from tqdm import tqdm
from pathos.multiprocessing import Pool,cpu_count

from shapely.geometry import MultiPolygon
from geopy.distance import vincenty

from utils import load_config,create_folder_lookup


def remove_tiny_shapes(x,regionalized=False):
    """This function will remove the small shapes of multipolygons. Will reduce the size of the file.
    
    Arguments:
        *x* : a geometry feature (Polygon) to simplify. Countries which are very large will see larger (unhabitated) islands being removed.
    
    Optional Arguments:
        *regionalized* : set to True to allow for different threshold settings (default: **False**).
        
    Returns:
        *MultiPolygon* : a shapely geometry MultiPolygon without tiny shapes.
        
    """
    if x.geometry.geom_type == 'Polygon':
        return x.geometry
    elif x.geometry.geom_type == 'MultiPolygon':
        
        if regionalized == False:
            area1 = 0.1
            area2 = 250
                
        elif regionalized == True:
            area1 = 0.01
            area2 = 50           

        # dont remove shapes if total area is already very small
        if x.geometry.area < area1:
            return x.geometry
        # remove bigger shapes if country is really big

        if x['GID_0'] in ['CHL','IDN']:
            threshold = 0.01
        elif x['GID_0'] in ['RUS','GRL','CAN','USA']:
            if regionalized == True:
                threshold = 0.01
            else:
                threshold = 0.01

        elif x.geometry.area > area2:
            threshold = 0.1
        else:
            threshold = 0.001

        # save remaining polygons as new multipolygon for the specific country
        new_geom = []
        for y in x.geometry:
            if y.area > threshold:
                new_geom.append(y)
        
        return MultiPolygon(new_geom)

def planet_osm():
    """
    This function will download the planet file from the OSM servers. 
    """
    data_path = load_config()['paths']['data']
    osm_path_in = os.path.join(data_path,'planet_osm')

    if not os.path.exists(osm_path_in):
        os.makedirs(osm_path_in)
    
    if 'planet-latest.osm.pbf' not in os.listdir(osm_path_in):
        
        url = 'https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf'
        urllib.request.urlretrieve(url, os.path.join(osm_path_in,'planet-latest.osm.pbf'))
    
    else:
        print('Planet file is already downloaded')

 
def global_shapefiles(regionalized=False):
    """ 
    This function will simplify shapes and add necessary columns, to make further processing more quickly
    
    For now, we will make use of the latest GADM data: https://gadm.org/download_world.html

    Optional Arguments:
        *regionalized*  : Default is **False**. Set to **True** will also create the global_regions.shp file.
    """

    data_path = load_config()['paths']['data']
   
    # path to country GADM file
    if regionalized == False:
        
        # load country file
        country_gadm_path = os.path.join(data_path,'GADM','gadm34_0.shp')
        gadm_level0 = geopandas.read_file(country_gadm_path)
    
        # remove antarctica, no roads there anyways
        gadm_level0 = gadm_level0.loc[~gadm_level0['NAME_0'].isin(['Antarctica'])]
        
        # remove tiny shapes to reduce size substantially
        gadm_level0['geometry'] =   gadm_level0.apply(remove_tiny_shapes,axis=1)
    
        # simplify geometries
        gadm_level0['geometry'] = gadm_level0.simplify(tolerance = 0.005, preserve_topology=True).buffer(0.01).simplify(tolerance = 0.005, preserve_topology=True)
        
        # add additional info
        glob_info_path = os.path.join(data_path,'input_data','global_information.xlsx')
        load_glob_info = pandas.read_excel(glob_info_path)
        
        gadm_level0 = gadm_level0.merge(load_glob_info,left_on='GID_0',right_on='ISO_3digit')
   
        #save to new country file
        glob_ctry_path = os.path.join(data_path,'input_data','global_countries.shp')
        gadm_level0.to_file(glob_ctry_path)
          
    else:

        # this is dependent on the country file, so check whether that one is already created:
        glob_ctry_path = os.path.join(data_path,'input_data','global_countries.shp')
        if os.path.exists(glob_ctry_path):
            gadm_level0 = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
        else:
            print('ERROR: You need to create the country file first')   
            return None
        
    # load region file
        region_gadm_path = os.path.join(data_path,'GADM','gadm34_1.shp')
        gadm_level1 = geopandas.read_file(region_gadm_path)
       
        # remove tiny shapes to reduce size substantially
        gadm_level1['geometry'] =   gadm_level1.apply(remove_tiny_shapes,axis=1)
    
        # simplify geometries
        gadm_level1['geometry'] = gadm_level1.simplify(tolerance = 0.005, preserve_topology=True).buffer(0.01).simplify(tolerance = 0.005, preserve_topology=True)
        
        # add additional info
        glob_info_path = os.path.join(data_path,'input_data','global_information.xlsx')
        load_glob_info = pandas.read_excel(glob_info_path)
        
        gadm_level1 = gadm_level1.merge(load_glob_info,left_on='GID_0',right_on='ISO_3digit')
        gadm_level1.rename(columns={'coordinates':'coordinate'}, inplace=True)
    
        # add some missing geometries from countries with no subregions
        get_missing_countries = list(set(list(gadm_level0.GID_0.unique())).difference(list(gadm_level1.GID_0.unique())))
        
        mis_country = gadm_level0.loc[gadm_level0['GID_0'].isin(get_missing_countries)]#
        mis_country['GID_1'] = mis_country['GID_0']+'_'+str(0)+'_'+str(1)
    
        gadm_level1 = geopandas.GeoDataFrame( pandas.concat( [gadm_level1,mis_country] ,ignore_index=True) )
        gadm_level1.reset_index(drop=True,inplace=True)
       
        #save to new country file
        gadm_level1.to_file(os.path.join(data_path,'input_data','global_regions.shp'))
   
      
def poly_files(data_path,global_shape,save_shapefile=False,regionalized=False):

    """
    This function will create the .poly files from the world shapefile. These
    .poly files are used to extract data from the openstreetmap files.
    
    This function is adapted from the OSMPoly function in QGIS.
    
    Arguments:
        *data_path* : base path to location of all files.
        
        *global_shape*: exact path to the global shapefile used to create the poly files.
        
    Optional Arguments:
        *save_shape_file* : Default is **False**. Set to **True** will the new shapefile with the 
        countries that we include in this analysis will be saved.     
        
        *regionalized*  : Default is **False**. Set to **True** will perform the analysis 
        on a regional level.
    
    Returns:
        *.poly file* for each country in a new dir in the working directory.
    """     
    
# =============================================================================
#     """ Create output dir for .poly files if it is doesnt exist yet"""
# =============================================================================
    poly_dir = os.path.join(data_path,'country_poly_files')
    
    if regionalized == True:
        poly_dir = os.path.join(data_path,'regional_poly_files')
    
    if not os.path.exists(poly_dir):
        os.makedirs(poly_dir)

# =============================================================================
#     """ Set the paths for the files we are going to use """
# =============================================================================
    wb_poly_out = os.path.join(data_path,'input_data','country_shapes.shp')
 
    if regionalized == True:
        wb_poly_out = os.path.join(data_path,'input_data','regional_shapes.shp')

# =============================================================================
#   """Load country shapes and country list and only keep the required countries"""
# =============================================================================
    wb_poly = geopandas.read_file(global_shape)
    
    # filter polygon file
    if regionalized == True:
        wb_poly = wb_poly.loc[wb_poly['GID_0'] != '-']
        wb_poly = wb_poly.loc[wb_poly['TYPE_1'] != 'Water body']

    else:
        wb_poly = wb_poly.loc[wb_poly['ISO_3digit'] != '-']
   
    wb_poly.crs = {'init' :'epsg:4326'}

    # and save the new country shapefile if requested
    if save_shapefile == True:
        wb_poly.to_file(wb_poly_out)
    
    # we need to simplify the country shapes a bit. If the polygon is too diffcult,
    # osmconvert cannot handle it.
#    wb_poly['geometry'] = wb_poly.simplify(tolerance = 0.1, preserve_topology=False)

# =============================================================================
#   """ The important part of this function: create .poly files to clip the country 
#   data from the openstreetmap file """    
# =============================================================================
    num = 0
    # iterate over the counties (rows) in the world shapefile
    for f in wb_poly.iterrows():
        f = f[1]
        num = num + 1
        geom=f.geometry

#        try:
        # this will create a list of the different subpolygons
        if geom.geom_type == 'MultiPolygon':
            polygons = geom
        
        # the list will be lenght 1 if it is just one polygon
        elif geom.geom_type == 'Polygon':
            polygons = [geom]

        # define the name of the output file, based on the ISO3 code
        ctry = f['GID_0']
        if regionalized == True:
            attr=f['GID_1']
        else:
            attr=f['GID_0']
        
        # start writing the .poly file
        f = open(poly_dir + "/" + attr +'.poly', 'w')
        f.write(attr + "\n")

        i = 0
        
        # loop over the different polygons, get their exterior and write the 
        # coordinates of the ring to the .poly file
        for polygon in polygons:

            if ctry == 'CAN':
                dist = vincenty(polygon.centroid.coords[:1][0], (83.24,-79.80), ellipsoid='WGS-84').kilometers
                if dist < 2000:
                    continue

            if ctry == 'RUS':
                dist = vincenty(polygon.centroid.coords[:1][0], (58.89,82.26), ellipsoid='WGS-84').kilometers
                if dist < 500:
                    print(attr)
                    continue
                
            polygon = numpy.array(polygon.exterior)

            j = 0
            f.write(str(i) + "\n")

            for ring in polygon:
                j = j + 1
                f.write("    " + str(ring[0]) + "     " + str(ring[1]) +"\n")

            i = i + 1
            # close the ring of one subpolygon if done
            f.write("END" +"\n")

        # close the file when done
        f.write("END" +"\n")
        f.close()
#        except:
#            print(f['GID_1'])

def clip_osm(data_path,planet_path,area_poly,area_pbf):
    """ Clip the an area osm file from the larger continent (or planet) file and save to a new osm.pbf file. 
    This is much faster compared to clipping the osm.pbf file while extracting through ogr2ogr.
    
    This function uses the osmconvert tool, which can be found at http://wiki.openstreetmap.org/wiki/Osmconvert. 
    
    Either add the directory where this executable is located to your environmental variables or just put it in the 'scripts' directory.
    
    Arguments:
        *continent_osm*: path string to the osm.pbf file of the continent associated with the country.
        
        *area_poly*: path string to the .poly file, made through the 'create_poly_files' function.
        
        *area_pbf*: path string indicating the final output dir and output name of the new .osm.pbf file.
        
    Returns:
        a clipped .osm.pbf file.
    """ 
    print('{} started!'.format(area_pbf))

    osm_convert_path = os.path.join(data_path,'osmconvert','osmconvert64')
    try: 
        if (os.path.exists(area_pbf) is not True):
            os.system('{}  {} -B={} --complete-ways -o={}'.format(osm_convert_path,planet_path,area_poly,area_pbf))
        print('{} finished!'.format(area_pbf))

    except:
        print('{} did not finish!'.format(area_pbf))
    

def single_country(country,regionalized=False,create_poly_files=False):
    """    
    Clip a country from the planet osm file and save to individual osm.pbf files
    
    This function has the option to extract individual regions
    
    Arguments:
        *country* : The country for which we want extract the data.
    
    Keyword Arguments:
        *regionalized* : Default is **False**. Set to **True** will parallelize the extraction over all regions within a country.
        *create_poly_files* : Default is **False**. Set to **True** will create new .poly files. 
        
    """
  
     # set data path
    data_path = load_config()['paths']['data']
    
    # path to planet file
    planet_path = os.path.join(data_path,'planet_osm','planet-latest.osm.pbf')

    # global shapefile path
    if regionalized == True:
        world_path = os.path.join(data_path,'input_data','global_regions.shp')
    else:
        world_path = os.path.join(data_path,'input_data','global_countries.shp')

    # create poly files for all countries
    if create_poly_files == True:
        poly_files(data_path,world_path,save_shapefile=False,regionalized=regionalized)

    if not os.path.exists(os.path.join(data_path,'country_poly_files')):
        os.makedirs(os.path.join(data_path,'country_poly_files'))

    if not os.path.exists(os.path.join(data_path,'country_osm')):
        os.makedirs(os.path.join(data_path,'country_osm'))
            
    ctry_poly = os.path.join(data_path,'country_poly_files','{}.poly'.format(country))
    ctry_pbf = os.path.join(data_path,'country_osm','{}.osm.pbf'.format(country))

    if regionalized == False:
        clip_osm(data_path,planet_path,ctry_poly,ctry_pbf)
        
    elif regionalized == True:
        
        if (os.path.exists(ctry_pbf) is not True):
            clip_osm(data_path,planet_path,ctry_poly,ctry_pbf)
        
        if not os.path.exists(os.path.join(data_path,'regional_poly_files')):
            os.makedirs(os.path.join(data_path,'regional_poly_files'))

        if not os.path.exists(os.path.join(data_path,'region_osm_admin1')):
            os.makedirs(os.path.join(data_path,'region_osm_admin1'))
        
        get_poly_files = [x for x in os.listdir(os.path.join(data_path,'regional_poly_files')) if x.startswith(country)]
        polyPaths = [os.path.join(data_path,'regional_poly_files',x) for x in get_poly_files]
        area_pbfs = [os.path.join(data_path,'region_osm_admin1',x.split('.')[0]+'.osm.pbf') for x in get_poly_files]
        data_paths = [data_path]*len(polyPaths)
        planet_paths = [ctry_pbf]*len(polyPaths)
      
        # and run all regions parallel to each other
        pool = Pool(cpu_count()-1)
        pool.starmap(clip_osm, zip(data_paths,planet_paths,polyPaths,area_pbfs)) 


def all_countries(subset = [], regionalized=False,reversed_order=False):
    """    
    Clip all countries from the planet osm file and save them to individual osm.pbf files
    
    Optional Arguments:
        *subset* : allow for a pre-defined subset of countries. Important to use ISO3 codes (default: {[]})
        
        *regionalized* : Default is **False**. Set to **True** if you want to have the regions of a country as well.
        
        *reversed_order* : Default is **False**. Set to **True**  to work backwards for a second process of the same country set to prevent overlapping calculations.
    
    Returns:
        clipped osm.pbf files for the defined set of countries (either the whole world by default or the specified subset)
    
    """
    
    # set data path
    data_path = load_config()['paths']['data']
    
    # path to planet file
    planet_path = os.path.join(data_path,'planet_osm','planet-latest.osm.pbf')
   
    # global shapefile path
    if regionalized == True:
        world_path = os.path.join(data_path,'input_data','global_regions.shp')
    else:
        world_path = os.path.join(data_path,'input_data','global_countries.shp')

    # create poly files for all countries
    poly_files(data_path,world_path,save_shapefile=False,regionalized=regionalized)
    
    # prepare lists for multiprocessing
    if not os.path.exists(os.path.join(data_path,'country_poly_files')):
        os.makedirs(os.path.join(data_path,'country_poly_files'))

    if not os.path.exists(os.path.join(data_path,'country_osm')):
        os.makedirs(os.path.join(data_path,'country_osm'))

    if regionalized == False:

        get_poly_files = os.listdir(os.path.join(data_path,'country_poly_files'))
        if len(subset) > 0:
            polyPaths = [os.path.join(data_path,'country_poly_files',x) for x in get_poly_files if x[:3] in subset]
            area_pbfs = [os.path.join(data_path,'region_osm_admin1',x.split('.')[0]+'.osm.pbf') for x in get_poly_files if x[:3] in subset]
        else:
            polyPaths = [os.path.join(data_path,'country_poly_files',x) for x in get_poly_files]
            area_pbfs = [os.path.join(data_path,'region_osm_admin1',x.split('.')[0]+'.osm.pbf') for x in get_poly_files]

        big_osm_paths = [planet_path]*len(polyPaths)
        
    elif regionalized == True:

        if not os.path.exists(os.path.join(data_path,'regional_poly_files')):
            os.makedirs(os.path.join(data_path,'regional_poly_files'))

        if not os.path.exists(os.path.join(data_path,'region_osm')):
            os.makedirs(os.path.join(data_path,'region_osm_admin1'))

        get_poly_files = os.listdir(os.path.join(data_path,'regional_poly_files'))
        if len(subset) > 0:
            polyPaths = [os.path.join(data_path,'regional_poly_files',x) for x in get_poly_files if x[:3] in subset]
            area_pbfs = [os.path.join(data_path,'region_osm_admin1',x.split('.')[0]+'.osm.pbf') for x in get_poly_files if x[:3] in subset]
            big_osm_paths = [os.path.join(data_path,'country_osm',x[:3]+'.osm.pbf') for x in get_poly_files if x[:3] in subset]
        else:
            polyPaths = [os.path.join(data_path,'regional_poly_files',x) for x in get_poly_files]
            area_pbfs = [os.path.join(data_path,'region_osm_admin1',x.split('.')[0]+'.osm.pbf') for x in get_poly_files]
            big_osm_paths = [os.path.join(data_path,'country_osm',x[:3]+'.osm.pbf') for x in get_poly_files]
            
    data_paths = [data_path]*len(polyPaths)

    # allow for reversed order if you want to run two at the same time (convenient to work backwards for the second process, to prevent overlapping calculation)   
    if reversed_order == True:
        polyPaths = polyPaths[::-1]
        area_pbfs = area_pbfs[::-1]
        big_osm_paths = big_osm_paths[::-1]

    # extract all country osm files through multiprocesing
    pool = Pool(cpu_count()-1)
    pool.starmap(clip_osm, zip(data_paths,big_osm_paths,polyPaths,area_pbfs)) 
    

def merge_SSBN_maps(country):
    """
    Function to merge SSBN maps to a country level.
    
    Arguments:
        *country* : ISO3 code of the country for which we want to merge the river and surface flood maps to country level.
   
    """
    try:
        print('{} started!'.format(country))
        
        hazard_path =  load_config()['paths']['hazard_data']
        
        folder_lookup = create_folder_lookup()
        
        country_ISO2 = coco.convert(names=[country], to='ISO2')
        country_full = folder_lookup[country]
        
        rps = ['5','10','20','50','75','100','200','250','1000']
        
        flood_types = ['fluvial_undefended','pluvial_undefended'] 
        flood_types_abb = ['FU','PU'] 
        
        flood_mapping = dict(zip(flood_types,flood_types_abb))
        
        for flood_type in flood_types:
            new_folder = os.path.join(hazard_path,'InlandFlooding',country_full,'{}_{}_merged'.format(country_ISO2,flood_type))
            try:
                os.mkdir(new_folder)
            except:
                None
            path_to_all_files = os.path.join(hazard_path,'InlandFlooding',country_full,'{}_{}'.format(country_ISO2,flood_type))
            full_paths = [os.path.join(path_to_all_files,x) for x in os.listdir(path_to_all_files) if x.endswith('.tif')]
            for rp in tqdm(rps,desc=flood_type+'_'+country,leave=False,total=len(rps),unit='rp'):
                get_one_rp = [x for x in full_paths if '-{}-'.format(rp) in x]
                stringlist_rp = ' '.join(get_one_rp)
                rp_out = os.path.join(new_folder,'{}-{}-{}.tif'.format(country_ISO2,flood_mapping[flood_type],rp))
                os.system('gdal_merge.py -q -o {} {} -co COMPRESS=LZW -co BIGTIFF=YES -co PREDICTOR=2 -co TILED=YES'.format(rp_out,stringlist_rp))
        print('{} finished!'.format(country))
        
    except:
        print('{} failed! It seems we do not have proper flood data for this country.'.format(country))
           

def run_SSBN_merge(from_=0,to_=235):
    """
    Merge all countries parallel.
    """
    
    data_path = load_config()['paths']['data']
    
    global_data = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_data = global_data[int(from_):int(to_)]
               
    with Pool(cpu_count()-1) as pool: 
        pool.map(merge_SSBN_maps,(global_data['ISO_3digit']),chunksize=1) 