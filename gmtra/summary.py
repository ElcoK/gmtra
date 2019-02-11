"""
Source code for Global Multihazard Transport Risk Analysis (GMTRA)

Functions to create summary statistics of the infrastructure data.

Copyright (C) 2019 Elco Koks. All versions released under the GNU Affero General Public License v3.0 license.
"""

import os
import tqdm
import geopandas as gpd
import pandas as pd
from pathos.multiprocessing import Pool,cpu_count

from gmtra.utils import load_config,line_length,map_roads
from gmtra.fetch import roads,railway


def all_outputs():
    """
    Summarize all outputs into .csv files per hazard and asset type.
    """
    data_path = load_config()['paths']['data']

    # Fluvial Flooding
    get_files = os.listdir(os.path.join(data_path,'FU_impacts'))
    with Pool(40) as pool: 
        tot_road_FU = list(tqdm(pool.imap(load_FU_csv,get_files), total=len(get_files)))
    pd.concat(tot_road_FU,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','FU_road_losses.csv'))        
    
    get_files = os.listdir(os.path.join(data_path,'FU_impacts_rail'))
    with Pool(40) as pool: 
        tot_road_FU = list(tqdm(pool.imap(load_FU_csv_rail,get_files), total=len(get_files)))
    pd.concat(tot_road_FU,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','FU_rail_losses.csv'))        
    
    # Pluvial Flooding
    get_files = os.listdir(os.path.join(data_path,'PU_impacts'))
    with Pool(40) as pool: 
        tot_road_PU = list(tqdm(pool.imap(load_PU_csv,get_files), total=len(get_files)))
    pd.concat(tot_road_PU,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','PU_road_losses.csv'))        

    get_files = os.listdir(os.path.join(data_path,'PU_impacts_rail'))
    with Pool(40) as pool: 
        tot_road_PU = list(tqdm(pool.imap(load_PU_csv_rail,get_files), total=len(get_files)))
    pd.concat(tot_road_PU,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','PU_rail_losses.csv'))        

    # Earthquakes
    get_files = os.listdir(os.path.join(data_path,'EQ_impacts'))
    with Pool(40) as pool: 
        tot_road_EQ = list(tqdm(pool.imap(load_EQ_csv,get_files), total=len(get_files)))
    pd.concat(tot_road_EQ,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','EQ_road_losses.csv'))        

    get_files = os.listdir(os.path.join(data_path,'EQ_impacts_rail'))
    with Pool(40) as pool: 
        tot_road_EQ = list(tqdm(pool.imap(load_EQ_csv_rail,get_files), total=len(get_files)))
    pd.concat(tot_road_EQ,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','EQ_rail_losses.csv'))        

   # Coastal Flooding
    get_files = os.listdir(os.path.join(data_path,'CF_impacts'))
    with Pool(40) as pool: 
        tot_road_CF = list(tqdm(pool.imap(load_CF_csv,get_files), total=len(get_files)))
    pd.concat(tot_road_CF,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','CF_road_losses.csv'))        

    get_files = os.listdir(os.path.join(data_path,'CF_impacts_rail'))
    with Pool(40) as pool: 
        tot_road_CF = list(tqdm(pool.imap(load_CF_csv_rail,get_files), total=len(get_files)))
    pd.concat(tot_road_CF,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','CF_rail_losses.csv'))        

    # Coastal Flooding
    get_files = os.listdir(os.path.join(data_path,'Cyc_impacts'))
    with Pool(40) as pool: 
        tot_road_Cyc = list(tqdm(pool.imap(load_Cyc_csv,get_files), total=len(get_files)))
    pd.concat(tot_road_Cyc,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','Cyc_road_losses.csv'))        

    get_files = os.listdir(os.path.join(data_path,'Cyc_impacts_rail'))
    with Pool(40) as pool: 
        tot_road_Cyc = list(tqdm(pool.imap(load_Cyc_csv_rail,get_files), total=len(get_files)))
    pd.concat(tot_road_Cyc,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','Cyc_rail_losses.csv'))        
 

    # Fluvial
    get_files = os.listdir(os.path.join(data_path,'FU_sensitivity'))
    with Pool(40) as pool: 
        tot_road_EQ = list(tqdm(pool.imap(load_FU_csv_sens,get_files), total=len(get_files)))
    pd.concat(tot_road_EQ,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','sa_FU_road_losses.csv'))    #
    # Pluvial
    get_files = os.listdir(os.path.join(data_path,'PU_sensitivity'))
    with Pool(40) as pool: 
        tot_road_EQ = list(tqdm(pool.imap(load_PU_csv_sens,get_files), total=len(get_files)))
    pd.concat(tot_road_EQ,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','sa_PU_road_losses.csv'))    #
    # Coastal
    get_files = os.listdir(os.path.join(data_path,'CF_sensitivity'))
    with Pool(40) as pool: 
        tot_road_EQ = list(tqdm(pool.imap(load_CF_csv_sens,get_files), total=len(get_files)))
    pd.concat(tot_road_EQ,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','sa_CF_road_losses.csv'))    #
    # Earthquakes
    get_files = os.listdir(os.path.join(data_path,'EQ_sensitivity'))
    with Pool(40) as pool: 
        tot_road_EQ = list(tqdm(pool.imap(load_EQ_csv_sens,get_files), total=len(get_files)))
    pd.concat(tot_road_EQ,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','sa_EQ_road_losses.csv'))    #
    # Cyclones
    get_files = os.listdir(os.path.join(data_path,'Cyc_sensitivity'))
    with Pool(40) as pool: 
        tot_road_Cyc = list(tqdm(pool.imap(load_Cyc_csv_sens,get_files), total=len(get_files)))
    pd.concat(tot_road_Cyc,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','sa_Cyc_road_losses.csv'))    #
    
    
    # Bridges
    get_files = os.listdir(os.path.join(data_path,'bridge_rail_risk'))
    with Pool(40) as pool: 
        tot_bridges = list(tqdm(pool.imap(load_bridge_rail_csv,get_files), total=len(get_files)))
    pd.concat(tot_bridges,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','bridge_rail_risk_.csv'))        
    
#    # Bridges
    get_files = os.listdir(os.path.join(data_path,'bridge_road_risk'))
    with Pool(40) as pool: 
        tot_bridges = list(tqdm(pool.imap(load_bridge_road_csv,get_files), total=len(get_files)))
    pd.concat(tot_bridges,sort=True).reset_index(drop=True).to_csv(os.path.join(data_path,'summarized','bridges_road_risk_.csv'))        

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

def load_FU_csv(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'FU_impacts',x)) 

def load_FU_csv_rail(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'FU_impacts_rail',x)) 

def load_FU_csv_sens(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'FU_sensitivity',x)) 

def load_CF_csv(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'CF_impacts',x))

def load_CF_csv_rail(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'CF_impacts_rail',x))

def load_CF_csv_sens(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'CF_sensitivity',x)) 

def load_PU_csv(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'PU_impacts',x))

def load_PU_csv_rail(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'PU_impacts_rail',x))

def load_PU_csv_sens(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'PU_sensitivity',x)) 

def load_EQ_csv(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'EQ_impacts',x))

def load_EQ_csv_rail(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'EQ_impacts_rail',x))

def load_EQ_csv_sens(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'EQ_sensitivity',x)) 

def load_Cyc_csv(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'Cyc_impacts',x))

def load_Cyc_csv_rail(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'Cyc_impacts_rail',x))

def load_Cyc_csv_sens(x):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'Cyc_sensitivity',x))

def load_bridge_rail_csv(file):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'bridge_rail_risk',file))

def load_bridge_road_csv(file):
    data_path = load_config()['paths']['data']
    return pd.read_csv(os.path.join(data_path,'bridge_road_risk',file))