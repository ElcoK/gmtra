"""
Source code for Global Multihazard Transport Risk Analysis (GMTRA)

Function to intersect the infrastructure asset data with global hazard maps.

Copyright (C) 2019 Elco Koks. All versions released under the GNU Affero General Public License v3.0 license.
"""
import os
import numpy
import pandas
import geopandas
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import mapping
from tqdm import tqdm

from gmtra.utils import load_config,create_folder_lookup,map_roads,line_length,get_raster_value
from gmtra.fetch import roads,railway

from shapely.geometry import MultiLineString
import country_converter as coco

def single_polygonized(flood_scen,region,geometry,country_ISO3,hzd='FU'):
    """
    Function to overlay a surface or river flood hazard map with the infrastructure assets. 
    
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
    with rasterio.open(flood_path) as src:
        out_image, out_transform = mask(src, geoms, crop=True)

        out_image[out_image == 999] = -1
        out_image[out_image <= 0] = -1
        out_image = numpy.round(out_image,1)
        out_image = numpy.array(out_image*100,dtype='int32')

        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v)
            in enumerate(
            shapes(out_image[0,:,:], mask=None, transform=out_transform)))

        gdf = geopandas.GeoDataFrame.from_features(list(results),crs='epsg:4326')
        gdf = gdf.loc[gdf.raster_val > 0]
        gdf = gdf.loc[gdf.raster_val < 5000]
        gdf['geometry'] = gdf.buffer(0)
        gdf['hazard'] = flood_scen
    
    return gdf

def multiple_polygonized(region,geometry,hzd_list,hzd_names):
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
        with rasterio.open(hzd_path) as src:
            out_image, out_transform = mask(src, geoms, crop=True)

            out_image[out_image <= 0] = -1
            out_image = numpy.array(out_image,dtype='int32')

            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                shapes(out_image[0,:,:], mask=None, transform=out_transform)))

            gdf = geopandas.GeoDataFrame.from_features(list(results),crs='epsg:4326')
            gdf = gdf.loc[gdf.raster_val >= 0]
            gdf = gdf.loc[gdf.raster_val < 5000]
            gdf['geometry'] = gdf.buffer(0)

            gdf['hazard'] = hzd_names[iter_]
            all_hzds.append(gdf)
    return pandas.concat(all_hzds)        


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
                    return MultiLineString([x[0] for x in append_hits]),int(numpy.mean([x[1] for x in append_hits]))
                else:
                    return MultiLineString([x[0] for x in append_hits]),int(numpy.max([x[1] for x in append_hits]))

    except:
        return x.geometry,0


def region_intersection(n,hzd,rail=False):
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

    global_regions = geopandas.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))

    x = global_regions.iloc[n]

    region = x.GID_2

    try:
        if (not rail) & os.path.exists(os.path.join(data_path,'output_{}_full'.format(hzd),'{}_{}.ft'.format(region,hzd))):
            print('{} already finished!'.format(region))
            return pandas.read_feather(os.path.join(os.path.join(data_path,'output_{}_full'.format(hzd),'{}_{}.ft'.format(region,hzd))))

        elif (rail) & os.path.exists(os.path.join(data_path,'output_{}_rail_full'.format(hzd),
                                                             '{}_{}.ft'.format(region,hzd))):
            print('{} already finished!'.format(region))
            return pandas.read_feather(os.path.join(os.path.join(data_path,'output_{}_rail_full'.format(hzd),
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
                hzds_data = multiple_polygonized(region,x.geometry,hzd_list,hzd_names)
            except:
                hzds_data = pandas.DataFrame(columns=['hazard'])

        for iter_,hzd_name in enumerate(hzd_names):
            if (hzd == 'PU') | (hzd == 'FU'):
                try:
                    hzds_data = single_polygonized(hzd_name,region,x.geometry,x.ISO_3digit,hzd)
                    hzd_region = hzds_data.loc[hzds_data.hazard == hzd_name]
                    hzd_region.reset_index(inplace=True,drop=True)
                except:
                    hzd_region = pandas.DataFrame(columns=['hazard'])

            elif (hzd == 'EQ') | (hzd == 'Cyc') | (hzd == 'CF'):
                try:
                    hzd_region = hzds_data.loc[hzds_data.hazard == hzd_name]
                    hzd_region.reset_index(inplace=True,drop=True)
                except:
                    hzd_region == pandas.DataFrame(columns=['hazard'])
                
            if len(hzd_region) == 0:
                infra_gpd['length_{}'.format(hzd_name)] = 0
                infra_gpd['val_{}'.format(hzd_name)] = 0
                continue

            hzd_reg_sindex = hzd_region.sindex
            tqdm.pandas(desc=hzd_name+'_'+region) 
            inb = infra_gpd.progress_apply(lambda x: intersect_hazard(x,hzd_reg_sindex,hzd_region),axis=1).copy()
            inb = inb.apply(pandas.Series)
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
    
        with rasterio.open(os.path.join(data_path,'Hazards','Liquefaction','Global','liquefaction_v1_deg.tif')) as src:
            out_image, out_transform = mask(src, geoms, crop=True)
            out_image = out_image[0,:,:]
    
            out_image[out_image <= 0] = -1
            out_image = numpy.array(out_image,dtype='int32')
    
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                shapes(out_image[:,:], mask=None, transform=out_transform)))
    
            gdf = geopandas.GeoDataFrame.from_features(list(results),crs='epsg:4326')
            gdf['geometry'] = gdf.buffer(0)
    
        tqdm.pandas(desc=region) 
        inb = infra_gpd.progress_apply(lambda x: intersect_hazard(x,gdf.sindex,gdf,liquefaction=True),axis=1).copy()
        inb = inb.apply(pandas.Series)
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
    
        with rasterio.open(os.path.join(data_path,'input_data','Crowther_Nature_Biome_Revision_01_WGS84_GeoTiff.tif')) as src:
            out_image, out_transform = mask(src, geoms, crop=True)
            out_image = out_image[0,:,:]
            tqdm.pandas(desc='Tree Density'+region)
            infra_gpd['Tree_Dens'] = infra_gpd.centroid.progress_apply(lambda x: get_raster_value(x,out_image,out_transform))
    
        infra_gpd['Tree_Dens']  = infra_gpd['Tree_Dens'].astype(float)
        infra_gpd['region'] = region
        infra_gpd = infra_gpd.drop('geometry',axis=1)
        if not rail:
            pandas.DataFrame(infra_gpd).to_feather(os.path.join(data_path,'tree_cover_road','{}.ft'.format(region)))
        else:
            pandas.DataFrame(infra_gpd).to_feather(os.path.join(data_path,'tree_cover_rail','{}.ft'.format(region)))

        print('{} finished!'.format(x[3]))

    except:
        print('{} failed!'.format(x[3]))

def bridge_intersection(file,rail=False):
    """
    Function to obtain all bridge intersection values from the regional intersection data.
    
    To be able to do this, we require all other hazard intersection files to be finished.
    
    Arguments:
        *file* : file with all unique road bridges in a region.
        
    Returns:
        *.feather file* : a geopandas GeoDataframe, saved as .feather file with all intersection values. 
    
    """
    data_path = load_config()['paths']['data']
    
    if not rail:
        all_EQ_files = [os.path.join(data_path,'output_EQ_full',x) for x in os.listdir(os.path.join(data_path,'output_EQ_full'))]
        all_Cyc_files = [os.path.join(data_path,'output_Cyc_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_full'))]
        all_PU_files = [os.path.join(data_path,'output_PU_full',x) for x in os.listdir(os.path.join(data_path,'output_PU_full'))]
        all_FU_files = [os.path.join(data_path,'output_FU_full',x) for x in os.listdir(os.path.join(data_path,'output_FU_full'))]
        all_CF_files = [os.path.join(data_path,'output_FU_full',x) for x in os.listdir(os.path.join(data_path,'output_FU_full'))]

    else:
        all_EQ_files = [os.path.join(data_path,'output_EQ_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_EQ_rail_full'))]
        all_Cyc_files = [os.path.join(data_path,'output_Cyc_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_rail_full'))]
        all_PU_files = [os.path.join(data_path,'output_PU_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_PU_rail_full'))]
        all_FU_files = [os.path.join(data_path,'output_FU_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_FU_rail_full'))]        
        all_CF_files = [os.path.join(data_path,'output_CF_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_CF_rail_full'))]
        
    df_bridge = pandas.read_csv(file,index_col=[0])
    df_bridge['osm_id'] = df_bridge.osm_id.astype(str)

    df_EQ = pandas.read_feather([x for x in all_EQ_files if os.path.split(file)[1][:-6] in x][0])
    df_EQ['osm_id'] = df_EQ.osm_id.astype(str)

    df_Cyc = pandas.read_feather([x for x in all_Cyc_files if os.path.split(file)[1][:-6] in x][0])
    df_Cyc['osm_id'] = df_Cyc.osm_id.astype(str)

    df_PU = pandas.read_feather([x for x in all_PU_files if os.path.split(file)[1][:-6] in x][0])
    df_PU['osm_id'] = df_PU.osm_id.astype(str)

    df_FU = pandas.read_feather([x for x in all_FU_files if os.path.split(file)[1][:-6] in x][0])
    df_FU['osm_id'] = df_FU.osm_id.astype(str)

    df_CF = pandas.read_feather([x for x in all_CF_files if os.path.split(file)[1][:-6] in x][0])
    df_CF['osm_id'] = df_CF.osm_id.astype(str)
    
    if len(df_bridge.loc[df_bridge.osm_id.isin(list(df_EQ.osm_id))]) == 0:
        df_output = pandas.DataFrame(columns=list(df_EQ[[x for x in list(df_EQ.columns) if ('val' in x) | ('length_' in x)]].columns),index=df_bridge.index).fillna(0)
        df_bridge = pandas.concat([df_bridge,df_output],axis=1)
    else:
        region_bridges = df_bridge.loc[df_bridge.osm_id.isin(list(df_EQ.osm_id))]
        df_reg_bridges = df_EQ.loc[df_EQ.osm_id.isin([str(x) for x in list(region_bridges.osm_id)])]   
        df_bridge = df_bridge.merge(df_reg_bridges[[x for x in list(df_EQ.columns) 
                                        if ('val' in x) | ('length_' in x)]+['osm_id']],
                        left_on='osm_id',right_on='osm_id',how='left')

    if len(df_bridge.loc[df_bridge.osm_id.isin(list(df_Cyc.osm_id))]) == 0:
        df_output = pandas.DataFrame(columns=list(df_Cyc[[x for x in list(df_Cyc.columns) if ('val' in x) | ('length_' in x)]].columns),index=df_bridge.index).fillna(0)
        df_bridge = pandas.concat([df_bridge,df_output],axis=1)
    else:
        region_bridges = df_bridge.loc[df_bridge.osm_id.isin(list(df_Cyc.osm_id))]
        df_reg_bridges = df_Cyc.loc[df_Cyc.osm_id.isin([str(x) for x in list(region_bridges.osm_id)])]   
        df_bridge = df_bridge.merge(df_reg_bridges[[x for x in list(df_Cyc.columns) 
                                        if ('val' in x) | ('length_' in x)]+['osm_id']],
                        left_on='osm_id',right_on='osm_id',how='left')

    if len(df_bridge.loc[df_bridge.osm_id.isin(list(df_FU.osm_id))]) == 0:
        df_output = pandas.DataFrame(columns=list(df_FU[[x for x in list(df_FU.columns) if ('val' in x) | ('length_' in x)]].columns),index=df_bridge.index).fillna(0)
        df_bridge = pandas.concat([df_bridge,df_output],axis=1)
    else:
        region_bridges = df_bridge.loc[df_bridge.osm_id.isin(list(df_FU.osm_id))]
        df_reg_bridges = df_FU.loc[df_FU.osm_id.isin([str(x) for x in list(region_bridges.osm_id)])]   
        df_bridge = df_bridge.merge(df_reg_bridges[[x for x in list(df_FU.columns) 
                                        if ('val' in x) | ('length_' in x)]+['osm_id']],
                        left_on='osm_id',right_on='osm_id',how='left')

    if len(df_bridge.loc[df_bridge.osm_id.isin(list(df_PU.osm_id))]) == 0:
        df_output = pandas.DataFrame(columns=list(df_PU[[x for x in list(df_PU.columns) if ('val' in x) | ('length_' in x)]].columns),index=df_bridge.index).fillna(0)
        df_bridge = pandas.concat([df_bridge,df_output],axis=1)
    else:
        region_bridges = df_bridge.loc[df_bridge.osm_id.isin(list(df_PU.osm_id))]
        df_reg_bridges = df_PU.loc[df_PU.osm_id.isin([str(x) for x in list(region_bridges.osm_id)])]   
        df_bridge = df_bridge.merge(df_reg_bridges[[x for x in list(df_PU.columns) 
                                        if ('val' in x) | ('length_' in x)]+['osm_id']],
                        left_on='osm_id',right_on='osm_id',how='left')

    if len(df_bridge.loc[df_bridge.osm_id.isin(list(df_CF.osm_id))]) == 0:
        df_output = pandas.DataFrame(columns=list(df_CF[[x for x in list(df_CF.columns) if ('val' in x) | ('length_' in x)]].columns),index=df_bridge.index).fillna(0)
        df_bridge = pandas.concat([df_bridge,df_output],axis=1)
    else:
        region_bridges = df_bridge.loc[df_bridge.osm_id.isin(list(df_CF.osm_id))]
        df_reg_bridges = df_CF.loc[df_CF.osm_id.isin([str(x) for x in list(region_bridges.osm_id)])]   
        df_bridge = df_bridge.merge(df_reg_bridges[[x for x in list(df_CF.columns) 
                                        if ('val' in x) | ('length_' in x)]+['osm_id']],
                        left_on='osm_id',right_on='osm_id',how='left')
        
    df_bridge.drop('geometry',inplace=True,axis=1)
    
    if not rail:
        df_bridge.to_feather(os.path.join(data_path,'bridges_osm_roads','{}.ft'.format(list(df_bridge.region.unique())[0])))
    else:
        df_bridge.to_feather(os.path.join(data_path,'bridges_osm_rail','{}.ft'.format(list(df_bridge.region.unique())[0])))
        

        