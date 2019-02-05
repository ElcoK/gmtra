# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:00:35 2018

@author: cenv0574
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import mapping
from tqdm import tqdm
from pathos.multiprocessing import Pool

sys.path.append(os.path.join( '..'))
from utils import load_config,create_folder_lookup,map_roads,line_length,get_raster_value
from fetch import roads,railway,bridges

from rasterio.features import shapes
from shapely.geometry import MultiLineString
import country_converter as coco

def create_hzd_df_fl_single(flood_scen,region,geometry,country_ISO3,hzd='FU'):
    """
    Function to overlay a inland flood hazard maps with the infrastructure assets
    
    Arguments:
        *flood_scen* : Unique ID for the flood scenario to be used.
        *region* : Unique ID of the region that is intersected.
        *geometry* : Shapely geometry of the region that is being intersected.
        *country_ISO3* : ISO3 code of the country in which the region is situated. 
        Required to get the FATHOM flood maps.
    
    Optional Arguments:
        *hzd* : Default is **FU**. Can be changed to **PU** for surface flooding.
        
    Returns:
        *gdf* : A GeoDataFrame where each row is a poylgon with the same flood depth.
    """
     
    hazard_path =  load_config()['paths']['hazard_data']

    folder_dict = create_folder_lookup()

    if (country_ISO3 == 'SDN') | (country_ISO3 == 'SSD'):
        country_full = 'sudan'
        country_ISO2 = 'SD'
    else:
        country_full = folder_dict[country_ISO3]
        country_ISO2 = coco.convert(names=[country_ISO3], to='ISO2')

    geoms = [mapping(geometry)]

    if hzd == 'FU':
        flood_type = 'fluvial_undefended'
    else:
        flood_type = 'pluvial_undefended'

    flood_path = os.path.join(hazard_path,'InlandFlooding',country_full,'{}_{}_merged'.format(country_ISO2,flood_type),'{}-{}.tif'.format(country_ISO2,flood_scen))    
    with rio.open(flood_path) as src:
        out_image, out_transform = mask(src, geoms, crop=True)

        out_image[out_image == 999] = -1
        out_image[out_image <= 0] = -1
        out_image = np.round(out_image,1)
        out_image = np.array(out_image*100,dtype='int32')

        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v)
            in enumerate(
            shapes(out_image[0,:,:], mask=None, transform=out_transform)))

        gdf = gpd.GeoDataFrame.from_features(list(results),crs='epsg:4326')
        gdf = gdf.loc[gdf.raster_val > 0]
        gdf = gdf.loc[gdf.raster_val < 5000]
        gdf['geometry'] = gdf.buffer(0)
        gdf['hazard'] = flood_scen
    
    return gdf

def create_hzd_df(region,geometry,hzd_list,hzd_names):
    """
    Function to overlay a set of hazard maps with infrastructure assets. This function 
    is used for Earthquakes, Coastal floods and Cyclones (the hazard maps with global coverage).
    
    Arguments:
        *region* : Unique ID of the region that is intersected.
        *geometry* : Shapely geometry of the region that is being intersected.
        *hzd_list* : list of file paths to each hazard. Make sure *hzd_list* and *hzd_names* are matching.
        *hzd_names* : llist of unique hazard IDs. Most often these are the return periods.
        
    Returns:
        *gdf* : A GeoDataFrame where each row is a poylgon with the same hazard value.
    """    
    geoms = [mapping(geometry)]

    all_hzds = []

    for iter_,hzd_path in enumerate(hzd_list):
        # extract the raster values values within the polygon 
        with rio.open(hzd_path) as src:
            out_image, out_transform = mask(src, geoms, crop=True)

            out_image[out_image <= 0] = -1
            out_image = np.array(out_image,dtype='int32')

            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                shapes(out_image[0,:,:], mask=None, transform=out_transform)))

            gdf = gpd.GeoDataFrame.from_features(list(results),crs='epsg:4326')
            gdf = gdf.loc[gdf.raster_val >= 0]
            gdf = gdf.loc[gdf.raster_val < 5000]
            gdf['geometry'] = gdf.buffer(0)

            gdf['hazard'] = hzd_names[iter_]
            all_hzds.append(gdf)
    return pd.concat(all_hzds)        


def intersect_hazard(x,hzd_reg_sindex,hzd_region,liquefaction=False):
    """
    Function to intersect an infrastructure asset (within a GeoDataFrame) with the hazard maps.
    
    Arguments:
        *x* : row in a GeoDataFrame that represents an unique infrastructure asset.
        *hzd_reg_sindex* : Spatial Index of the hazard GeoDataFrame.
        *hzd_region* : GeoDataFrame of a unique hazard map.
        
    Optional arguments:
        *liquefaction* : Default is **False**. Set to **True** if you are intersecting 
        with the liquefaction map.
        
    Returns:
        *tuple* : a shapely.geometry of the part of the asset that is affected 
        and the average hazard value in this intersection.            
    """
    matches = hzd_region.iloc[list(hzd_reg_sindex.intersection(x.geometry.bounds))].reset_index(drop=True)
    try:
        if len(matches) == 0:
            return x.geometry,0
        else:
            append_hits = []
            for match in matches.itertuples():
                inter = x.geometry.intersection(match.geometry)
                if inter.is_empty == True:
                    continue
                else:
                    if inter.geom_type == 'MultiLineString':
                        for interin in inter:
                            append_hits.append((interin,match.raster_val))
                    else:
                         append_hits.append((inter,match.raster_val))
                       
                    
            if len(append_hits) == 0:
                return x.geometry,0
            elif len(append_hits) == 1:
                return append_hits[0][0],int(append_hits[0][1])
            else:
                if liquefaction:
                    return MultiLineString([x[0] for x in append_hits]),int(np.mean([x[1] for x in append_hits]))
                else:
                    return MultiLineString([x[0] for x in append_hits]),int(np.max([x[1] for x in append_hits]))

    except:
        return x.geometry,0


def fetch_hazard_values(n,hzd,rail=False):
    """
    Function to intersect all return periods of a particualar hazard with all 
    road or railway assets in the specific region. 
    
    Arguments:
        *n* : the index ID of a region in the specified shapefile with all the regions.
        *hzd* : abbrevation of the hazard we want to intersect. **EQ** for earthquakes,
        **Cyc** for cyclones, **FU** for river flooding, **PU** for surface flooding
        and **CF** for coastal flooding.
        
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    Returns:
        *output* : a GeoDataFrame with all intersections between the 
        infrastructure assets and the specified hazard. Will also be saved as .feather file.
            
    """

    data_path = load_config()['paths']['data']
    hazard_path =  load_config()['paths']['hazard_data']

    global_regions = gpd.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))

    x = global_regions.iloc[n]

    region = x.GID_2

    try:
        if (not rail) & os.path.exists(os.path.join(data_path,'output_{}_full'.format(hzd),'{}_{}.ft'.format(region,hzd))):
            print('{} already finished!'.format(region))
            return pd.read_feather(os.path.join(os.path.join(data_path,'output_{}_full'.format(hzd),'{}_{}.ft'.format(region,hzd))))

        elif (rail) & os.path.exists(os.path.join(data_path,'output_{}_rail_full'.format(hzd),
                                                             '{}_{}.ft'.format(region,hzd))):
            print('{} already finished!'.format(region))
            return pd.read_feather(os.path.join(os.path.join(data_path,'output_{}_rail_full'.format(hzd),
                                                             '{}_{}.ft'.format(region,hzd))))

        if hzd == 'EQ':
            hzd_name_dir = 'Earthquake'
            hzd_names = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475']
        elif hzd == 'Cyc':
            hzd_name_dir = 'Cyclones'
            hzd_names = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
        elif hzd == 'FU':
            hzd_name_dir = 'FluvialFlooding'
            hzd_names = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250', 'FU-500', 'FU-1000']
        elif hzd == 'PU':
            hzd_name_dir = 'PluvialFlooding'
            hzd_names = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250', 'PU-500', 'PU-1000']
        elif hzd == 'CF':
            hzd_name_dir = 'CoastalFlooding'
            hzd_names = ['CF-10', 'CF-20', 'CF-50', 'CF-100', 'CF-200', 'CF-500', 'CF-1000']

        try:
            if not rail:
                road_gpd = roads(data_path,region,regional=True)
                road_dict = map_roads()
                road_gpd['length'] = road_gpd.geometry.apply(line_length)
                road_gpd.geometry = road_gpd.geometry.simplify(tolerance=0.5)
                road_gpd['road_type'] = road_gpd.infra_type.apply(lambda x: road_dict[x])

                infra_gpd = road_gpd.copy()

            elif rail:
                rail_gpd = railway(data_path,region,regional=True)
                rail_gpd['length'] = rail_gpd.geometry.apply(line_length)
                rail_gpd['geometry'] = rail_gpd.geometry.simplify(tolerance=0.5) 

                infra_gpd = rail_gpd.copy()

            print('{} osm data loaded!'.format(region))
        except:
            print('{} osm data not properly loaded!'.format(region))
            return None

        if (hzd == 'EQ') | (hzd == 'Cyc') | (hzd == 'CF'):
            hazard_path =  load_config()['paths']['hazard_data']    
            hazard_path = os.path.join(hazard_path,hzd_name_dir,'Global')
            hzd_list = [os.path.join(hazard_path,x) for x in os.listdir(hazard_path)]
            try:
                hzds_data = create_hzd_df(region,x.geometry,hzd_list,hzd_names)
            except:
                hzds_data = pd.DataFrame(columns=['hazard'])

        for iter_,hzd_name in enumerate(hzd_names):
            if (hzd == 'PU') | (hzd == 'FU'):
                try:
                    hzds_data = create_hzd_df_fl_single(hzd_name,region,x.geometry,x.ISO_3digit,hzd)
                    hzd_region = hzds_data.loc[hzds_data.hazard == hzd_name]
                    hzd_region.reset_index(inplace=True,drop=True)
                except:
                    hzd_region = pd.DataFrame(columns=['hazard'])

            elif (hzd == 'EQ') | (hzd == 'Cyc') | (hzd == 'CF'):
                try:
                    hzd_region = hzds_data.loc[hzds_data.hazard == hzd_name]
                    hzd_region.reset_index(inplace=True,drop=True)
                except:
                    hzd_region == pd.DataFrame(columns=['hazard'])
                
            if len(hzd_region) == 0:
                infra_gpd['length_{}'.format(hzd_name)] = 0
                infra_gpd['val_{}'.format(hzd_name)] = 0
                continue

            hzd_reg_sindex = hzd_region.sindex
            tqdm.pandas(desc=hzd_name+'_'+region) 
            inb = infra_gpd.progress_apply(lambda x: intersect_hazard(x,hzd_reg_sindex,hzd_region),axis=1).copy()
            inb = inb.apply(pd.Series)
            inb.columns = ['geometry','val_{}'.format(hzd_name)]
            inb['length_{}'.format(hzd_name)] = inb.geometry.apply(line_length)
            infra_gpd[['length_{}'.format(hzd_name),'val_{}'.format(hzd_name)]] = inb[['length_{}'.format(hzd_name),
                                                                                       'val_{}'.format(hzd_name)]] 
        output = infra_gpd.drop(['geometry'],axis=1)
        output['country'] = global_regions.loc[global_regions['GID_2'] == region]['ISO_3digit'].values[0]
        output['continent'] = global_regions.loc[global_regions['GID_2'] == region]['continent'].values[0]
        output['region'] = region
        if not rail:
            output.to_feather(os.path.join(data_path,'output_{}_full'.format(hzd),'{}_{}.ft'.format(region,hzd)))
        else:
            output.to_feather(os.path.join(data_path,'output_{}_rail_full'.format(hzd),'{}_{}.ft'.format(region,hzd)))

        print('Finished {}!'.format(region))
        return output

    except Exception as e:
        print('Failed to finish {} because of {}!'.format(region,e))


def get_liquefaction_region(x,rail=False):
    """
    Function to intersect all return periods of a particualar hazard with all 
    road or railway assets in the specific region. 
    
    Arguments:
        *x* : row of a region in the specified shapefile with all the regions.
        
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    Returns:
        *output* : a GeoDataFrame with all intersections between the 
        infrastructure assets and the liquefaction map. Will be saved as .feather file.
    """
    region = x[3]
    reg_geom = x[-1]
    data_path = load_config()['paths']['data']
    try:
        if (not rail) & os.path.exists(os.path.join(data_path,'liquefaction_road','{}_liq.ft'.format(region))):
            print('{} already finished!'.format(region))
            return None
        
        if not rail:
            road_gpd = roads(data_path,region,regional=True)
            road_dict = map_roads()
            road_gpd['length'] = road_gpd.geometry.apply(line_length)
            road_gpd.geometry = road_gpd.geometry.simplify(tolerance=0.5)
            road_gpd['road_type'] = road_gpd.infra_type.apply(lambda y: road_dict[y])
            infra_gpd = road_gpd.copy()
    
        else:
            rail_gpd = railway(data_path,region,regional=True)
            rail_gpd['length'] = rail_gpd.geometry.apply(line_length)
            rail_gpd.geometry = rail_gpd.geometry.simplify(tolerance=0.5)
            infra_gpd = rail_gpd.copy()
    
            # Get tree density values 
        geoms = [mapping(reg_geom.envelope.buffer(1))]
    
        with rio.open(os.path.join(data_path,'Hazards','Liquefaction','Global','liquefaction_v1_deg.tif')) as src:
            out_image, out_transform = mask(src, geoms, crop=True)
            out_image = out_image[0,:,:]
    
            out_image[out_image <= 0] = -1
            out_image = np.array(out_image,dtype='int32')
    
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                shapes(out_image[:,:], mask=None, transform=out_transform)))
    
            gdf = gpd.GeoDataFrame.from_features(list(results),crs='epsg:4326')
            gdf['geometry'] = gdf.buffer(0)
    
        tqdm.pandas(desc=region) 
        inb = infra_gpd.progress_apply(lambda x: intersect_hazard(x,gdf.sindex,gdf,liquefaction=True),axis=1).copy()
        inb = inb.apply(pd.Series)
        inb.columns = ['geometry','liquefaction']
        inb['length_liq'] = inb.geometry.apply(line_length)
        infra_gpd[['length_liq','liquefaction']] = inb[['length_liq','liquefaction']] 
        output = infra_gpd.drop(['geometry'],axis=1)
        output['country'] = region[:3] 
        output['continent'] = x[10]
        output['region'] = region
        
        if not rail:    
            output.to_feather(os.path.join(data_path,'liquefaction_road','{}_liq.ft'.format(region)))
        else:
            output.to_feather(os.path.join(data_path,'liquefaction_rail','{}_liq.ft'.format(region)))
            
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(region,e))
        	
def get_tree_density(x,rail=False):
    """
    Function to intersect all return periods of a particualar hazard with all 
    road or railway assets in the specific region. 
    
    Arguments:
        *x* : row of a region in the specified shapefile with all the regions.
        
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    Returns:
        *output* : a GeoDataFrame with all intersections between the 
        infrastructure assets and the liquefaction map. Will be saved as .feather file.
    """
    try:
        region = x[3]
        reg_geom = x[-1]
    		
        data_path = load_config()['paths']['data']
    
        if not rail:
            road_gpd = roads(data_path,region,regional=True)
            road_dict = map_roads()
            road_gpd['road_type'] = road_gpd.infra_type.apply(lambda y: road_dict[y])
            infra_gpd = road_gpd.copy()
    
        else:
            rail_gpd = railway(data_path,region,regional=True)
            infra_gpd = rail_gpd.copy()
    
            # Get tree density values 
        geoms = [mapping(reg_geom.envelope.buffer(1))]
    
        with rio.open(os.path.join(data_path,'input_data','Crowther_Nature_Biome_Revision_01_WGS84_GeoTiff.tif')) as src:
            out_image, out_transform = mask(src, geoms, crop=True)
            out_image = out_image[0,:,:]
            tqdm.pandas(desc='Tree Density'+region)
            infra_gpd['Tree_Dens'] = infra_gpd.centroid.progress_apply(lambda x: get_raster_value(x,out_image,out_transform))
    
        infra_gpd['Tree_Dens']  = infra_gpd['Tree_Dens'].astype(float)
        infra_gpd['region'] = region
        infra_gpd = infra_gpd.drop('geometry',axis=1)
        if not rail:
            pd.DataFrame(infra_gpd).to_feather(os.path.join(data_path,'tree_cover_road','{}.ft'.format(region)))
        else:
            pd.DataFrame(infra_gpd).to_feather(os.path.join(data_path,'tree_cover_rail','{}.ft'.format(region)))

        print('{} finished!'.format(x[3]))

    except:
        print('{} failed!'.format(x[3]))
	

def region_bridges(x):
    region = x[3]
    try:
        data_path = load_config()['paths']['data']
            
        bridges_osm = bridges(data_path,region,regional=True)

        bridges_osm['length'] = bridges_osm.geometry.apply(line_length)
        bridges_osm['length'] = bridges_osm['length']*1000
        road_dict = map_roads()
        bridges_osm['road_type'] = bridges_osm.road_type.apply(lambda y: road_dict[y])        
        bridges_osm['region'] = region
        bridges_osm['country'] = region[:3]        
        
        bridges_osm.to_csv(os.path.join(data_path,'bridges_osm','{}.csv'.format(region)))
        
        print('{} finished!'.format(region))
        
        return bridges_osm
    
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(region,e))


def get_all_bridges():
    data_path = load_config()['paths']['data']

    global_regions = gpd.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    global_regions = global_regions.loc[(global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))]))]

    with Pool(40) as pool:
        collect_bridges = pool.map(region_bridges,list(global_regions.to_records()),chunksize=1) 

    all_bridges = pd.concat(collect_bridges)
    all_bridges.reset_index(inplace=True,drop=True)
    all_bridges.to_csv(os.path.join(data_path,'output_data','osm_bridges.csv'))

    
def get_all_regions(hzd,from_=0,to_=46433):
    """
    Function to run intersection for all regions parallel.
    """    
    road = True
    railway = False
    
    hzds = [hzd]*int(to_)
    Roads = [road]*int(to_)
    Railways = [railway]*int(to_)

    data_path = load_config()['paths']['data']

    global_regions = gpd.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    global_regions = global_regions.loc[global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))])]

    if road:
        global_regions = global_regions.loc[(global_regions.GID_2.isin([x[:-10] for x in os.listdir(os.path.join(data_path,'road_stats'))]))]
        global_regions = global_regions.loc[~(global_regions.GID_2.isin(['_'.join((x.split('.')[0]).split('_')[:4]) for x in os.listdir(os.path.join(data_path,'output_{}_full'.format(hzd)))]))]
    else:
        global_regions = global_regions.loc[(global_regions.GID_2.isin([x[:-10] for x in os.listdir(os.path.join(data_path,'railway_stats'))]))]
        global_regions = global_regions.loc[~(global_regions.GID_2.isin(['_'.join((x.split('.')[0]).split('_')[:4]) for x in os.listdir(os.path.join(data_path,'output_{}_rail_full'.format(hzd)))]))]

    regions = list(global_regions.index)[::-1]
    print(len(regions))

    with Pool(40) as pool: 
        pool.starmap(fetch_hazard_values,zip(regions,hzds,Roads,Railways),chunksize=1) 
 
       
def get_all_tree_values():
    """
    Function to run intersection for all regions parallel.
    """
    data_path = load_config()['paths']['data']

    global_regions = gpd.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    global_regions = global_regions.loc[~(global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'tree_cover_rail'))]))]
    global_regions = global_regions.loc[(global_regions.GID_2.isin([x[:-10] for x in os.listdir(os.path.join(data_path,'railway_stats'))]))]


    with Pool(40) as pool:
        pool.map(get_tree_density,list(global_regions.to_records()),chunksize=1) 


def get_all_liquefaction_overlays():
    """
    Function to run intersection for all regions parallel.
    """
    data_path = load_config()['paths']['data']

    global_regions = gpd.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    global_regions = global_regions.loc[(global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))]))]
    global_regions = global_regions.loc[~(global_regions.GID_2.isin([(x.split('.')[0][:-4]) for x in os.listdir(os.path.join(data_path,'liquefaction_road'))]))]

    global_regions['Size'] = global_regions.area
    global_regions = global_regions.sort_values(by='Size')
    global_regions.drop(['Size'],inplace=True,axis=1)

    with Pool(40) as pool:
        pool.map(get_liquefaction_region,list(global_regions.to_records()),chunksize=1) 
        