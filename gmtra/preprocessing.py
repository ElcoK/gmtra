# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:55:09 2018

@author: cenv0574
"""

import os
import sys
import country_converter as coco
import geopandas as gpd

sys.path.append(os.path.join( '..'))
from miriam_py.utils import load_config

from tqdm import tqdm
from pathos.multiprocessing import Pool,cpu_count
from functions import create_folder_lookup


def merge_SSBN_maps(country):
    """
    Function to merge SSBN maps to a country level.
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
    
    global_data = gpd.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_data = global_data[int(from_):int(to_)]
               
    with Pool(cpu_count()-2) as pool: 
        pool.map(merge_SSBN_maps,(global_data['ISO_3digit']),chunksize=1) 