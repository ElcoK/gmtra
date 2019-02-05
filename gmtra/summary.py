import os
import sys
import geopandas as gpd
import pandas as pd
from pathos.multiprocessing import Pool,cpu_count

from gmtra.utils import load_config,line_length,map_roads
from gmtra.fetch import roads,railway
data_path = load_config()['paths']['data']

def get_region_road_stats(x):
    try:
        data_path = load_config()['paths']['data']

        if os.path.exists(os.path.join(data_path,'road_stats','{}_stats.csv'.format(x[3]))):
            print('{} already finished!'.format(x[3]))
            return None
        
        print('{} started!'.format(x[3]))
                
        road_dict = map_roads()
        road_gpd = roads(data_path,x[3],regional=True)
        road_gpd['length'] = road_gpd.geometry.apply(line_length)
        road_gpd['road_type'] = road_gpd.infra_type.apply(lambda x: road_dict[x])
        road_gpd = road_gpd.groupby('road_type').sum()
        road_gpd['continent'] = x[10]
        road_gpd['country'] = x[1]
        road_gpd['region'] = x[3]

        road_gpd.to_csv(os.path.join(data_path,'road_stats','{}_stats.csv'.format(x.GID_2)))
        
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(x[3],e))

def get_region_rail_stats(n):
    try:
        data_path = load_config()['paths']['data']
        global_data = gpd.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
        global_data = global_data.loc[global_data.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))])]

        x = global_data.iloc[n]

        if os.path.exists(os.path.join(data_path,'railway_stats','{}_stats.csv'.format(x.GID_2))):
            print('{} already finished!'.format(x.GID_2))
            return None
        
        print('{} started!'.format(x.GID_2))
                
        rail_gpd = railway(data_path,x.GID_2,regional=True)
        rail_gpd['length'] = rail_gpd.geometry.apply(line_length)
        rail_gpd = rail_gpd.groupby('infra_type').sum()
        rail_gpd['continent'] = x.continent
        rail_gpd['country'] = x.ISO_3digit
        rail_gpd['region'] = x.GID_2

        rail_gpd.to_csv(os.path.join(data_path,'railway_stats','{}_stats.csv'.format(x.GID_2)))
        
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(n,e))

def get_country_road_stats(iso3):
    try:
        print(iso3)
        data_path = load_config()['paths']['data']

        list_files = [os.path.join(data_path,'road_stats',x) for x in os.listdir(os.path.join(data_path,'road_stats')) if (iso3 in x)]

        collect_regions = []
        for file in list_files:
            collect_regions.append(pd.read_csv(file))
        return pd.concat(collect_regions).groupby(['road_type','continent','country']).sum()
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(iso3,e))
        
def all_country_stats():
    data_path = load_config()['paths']['data']

    global_countries = gpd.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    list_iso3 = [x.split('_')[0] for x in global_countries.ISO_3digit]

    with Pool(cpu_count()-1) as pool: 
        collect_countries = pool.map(get_country_road_stats,list_iso3,chunksize=1) 
    
    pd.concat(collect_countries).to_csv(os.path.join(data_path,'summarized','country_road_stats.csv'))    

def all_region_stats():
    data_path = load_config()['paths']['data']
    global_data = gpd.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    global_data = global_data.loc[global_data.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))])]

    with Pool(cpu_count()-1) as pool: 
        pool.map(get_region_road_stats,list(global_data.to_records()),chunksize=1) 

