"""
Source code for Global Multihazard Transport Risk Analysis (GMTRA)

Functions to run all regional code parallel.

Copyright (C) 2019 Elco Koks. All versions released under the GNU Affero General Public License v3.0 license.
"""

import os
import numpy
import pandas
import geopandas

from pathos.multiprocessing import Pool,cpu_count
from SALib.sample import morris

from gmtra.utils import load_config
import gmtra.sensitivity as sensitivity
import gmtra.damage as damage
from gmtra.hazard import region_intersection,get_tree_density,get_liquefaction_region
from gmtra.preprocessing import region_bridges,merge_SSBN_maps
from gmtra.exposure import regional_roads,regional_railway

def bridge_extraction(save_all=False):
    """
    Function to extract all bridges from OpenStreetMap.
    
    Optional Arguments:
        *save_all* : Default is **False**. Set to **True** if you would like to 
        save all bridges of the world in one csv file. Will become a big csv!
    
    """
    # set data path
    data_path = load_config()['paths']['data']
    
    # load shapefile with all regions
    global_regions = geopandas.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    global_regions = global_regions.loc[(global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))]))]

    # run the bridge extraction for all regions
    with Pool(cpu_count()-1) as pool: 
        collect_bridges = pool.map(region_bridges,list(global_regions.to_records()),chunksize=1) 

    # save all bridges in one file.
    if save_all:

        # concat them to one file
        all_bridges = pandas.concat(collect_bridges)
        all_bridges.reset_index(inplace=True,drop=True)

        all_bridges.to_csv(os.path.join(data_path,'output_data','osm_bridges.csv'))

def SSBN_merge(from_=0,to_=235):
    """
    Merge all countries parallel.
    
    Optional Arguments:
        *from_* : Default is **0**. Set to a different value if you would 
        like to select a different subset.
        
        *to_* : Default is **235**. Set to a different value if you would 
        like to select a different subset.
    """
    # set data path
    data_path = load_config()['paths']['data']
    
    # load shapefile with all countries
    global_data = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_data = global_data[int(from_):int(to_)]
               
    # run SSBN merge for all countries parallel
    with Pool(cpu_count()-1) as pool: 
        pool.map(merge_SSBN_maps,(global_data['ISO_3digit']),chunksize=1) 

def tree_values(rail=False):
    """
    Function to run intersection with global tree density map for all regions parallel.
    
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    """
    # set data path
    data_path = load_config()['paths']['data']

    # load shapefile with all regions
    global_regions = geopandas.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    
    if not rail:
        global_regions = global_regions.loc[(global_regions.GID_2.isin([x[:-10] for x in os.listdir(os.path.join(data_path,'road_stats'))]))]
        global_regions = global_regions.loc[~(global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'tree_cover_road'))]))]     
    else:
        global_regions = global_regions.loc[(global_regions.GID_2.isin([x[:-10] for x in os.listdir(os.path.join(data_path,'railway_stats'))]))]
        global_regions = global_regions.loc[~(global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'tree_cover_rail'))]))]

    # run tree value extraction for all regions parallel
    with Pool(cpu_count()-1) as pool: 
        pool.map(get_tree_density,list(global_regions.to_records()),chunksize=1) 


def liquefaction_overlays(rail=False):
    """
    Function to run intersection with global liquefaction map for all regions parallel.
    
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    """
    # set data path
    data_path = load_config()['paths']['data']

    # load shapefile with all regions and their information
    global_regions = geopandas.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    
    if not rail:
        global_regions = global_regions.loc[(global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))]))]
        global_regions = global_regions.loc[~(global_regions.GID_2.isin([(x.split('.')[0][:-4]) for x in os.listdir(os.path.join(data_path,'liquefaction_road'))]))]
    else:
        global_regions = global_regions.loc[(global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))]))]
        global_regions = global_regions.loc[~(global_regions.GID_2.isin([(x.split('.')[0][:-4]) for x in os.listdir(os.path.join(data_path,'liquefaction_rail'))]))]   

    global_regions['Size'] = global_regions.area
    global_regions = global_regions.sort_values(by='Size')
    global_regions.drop(['Size'],inplace=True,axis=1)

    # run liquefaction intersection for all regions parallel
    with Pool(cpu_count()-1) as pool: 
        pool.map(get_liquefaction_region,list(global_regions.to_records()),chunksize=1) 

def hazard_intersection(hzd,rail=False,from_=0,to_=46433):
    """
    Function to run intersection with hazard data for all regions parallel.
    
    Arguments:
        *hzd* : abbrevation of the hazard we want to intersect.  **FU** for river flooding, 
        **PU** for surface flooding and **CF** for coastal flooding.    
    
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
        *from_* : Default is **0**. Set to a different value if you would 
        like to select a different subset.
        
        *to_* : Default is **46433**. Set to a different value if you would 
        like to select a different subset.        
        
    """    
    # set the right booleans
    road = not rail
    railway = rail
    
    # create lists for the parallelization
    hzds = [hzd]*int(to_)
    Roads = [road]*int(to_)
    Railways = [railway]*int(to_)

    # set data path
    data_path = load_config()['paths']['data']

    # load shapefile with all regions
    global_regions = geopandas.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    global_regions = global_regions.loc[global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))])]

    # only keep regions for which we actually have stats and that are not done yet.
    if road:
        global_regions = global_regions.loc[(global_regions.GID_2.isin([x[:-10] for x in os.listdir(os.path.join(data_path,'road_stats'))]))]
        global_regions = global_regions.loc[~(global_regions.GID_2.isin(['_'.join((x.split('.')[0]).split('_')[:4]) for x in os.listdir(os.path.join(data_path,'output_{}_full'.format(hzd)))]))]
    else:
        global_regions = global_regions.loc[(global_regions.GID_2.isin([x[:-10] for x in os.listdir(os.path.join(data_path,'railway_stats'))]))]
        global_regions = global_regions.loc[~(global_regions.GID_2.isin(['_'.join((x.split('.')[0]).split('_')[:4]) for x in os.listdir(os.path.join(data_path,'output_{}_rail_full'.format(hzd)))]))]

    # get list of regions
    regions = list(global_regions.index)

    # run hazard intersections parallel
    with Pool(cpu_count()-1) as pool: 
        pool.starmap(region_intersection,zip(regions,hzds,Roads,Railways),chunksize=1) 

def exposure_analysis(rail=False): 
    """
    Get exposure statistics for all road or railway assets in all regions.
    
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
    
    """
    # set data path
    data_path = load_config()['paths']['data']

    # load shapefile with all regions
    global_regions = geopandas.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))

    # load csv with income group data and assign income group to regions
    incomegroups = pandas.read_csv(os.path.join(data_path,'input_data','incomegroups_2018.csv'),index_col=[0])
    income_dict = dict(zip(incomegroups.index,incomegroups.GroupCode))
    global_regions['wbincome'] = global_regions.GID_0.apply(lambda x : income_dict[x]) 
    
    # only keep regions for which we have data
    global_regions = global_regions.loc[global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))])]
    
    # create dictionary with information on protection standards
    prot_lookup = dict(zip(global_regions['GID_2'],global_regions['prot_stand']))

    # create lists for the parallelization
    regions = list(global_regions.to_records())
    prot_lookups = [prot_lookup]*len(regions)
    data_paths = [data_path]*len(regions)    

    # run exposure analysis parallel
    if not rail:
        with Pool(cpu_count()-1) as pool: 
            collect_output = pool.starmap(regional_roads,zip(regions,prot_lookups,data_paths),chunksize=1) 
    
        pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','total_exposure_road.csv'))
    
    else:
        with Pool(cpu_count()-1) as pool: 
            collect_output = pool.starmap(regional_railway,zip(regions,prot_lookups,data_paths),chunksize=1) 
    
        pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','total_exposure_railway.csv'))


def bridge_damage(rail=False):
    """
    Function to calculate the damage to bridges for all regions and all hazards.
    
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    """ 
    # set data path
    data_path  = load_config()['paths']['data']

    # get a list of all regions for which we can estimate the damages
    if not rail:
        all_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))]
    else:
        all_files = [os.path.join(data_path,'bridges_osm_rail',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_rail'))]
    
    # load csv with income group data and assign income group to regions
    incomegroups = pandas.read_csv(os.path.join(data_path,'input_data','incomegroups_2018.csv'),index_col=[0])
    income_lookup = dict(zip(incomegroups.CountryCode,incomegroups.GroupCode))
    
    # load earthquake curves
    eq_curve = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),sheet_name='bridge_curves',usecols=5)
    
    # create design standard table for floods and cyclones
    design_tables = numpy.array([[[1/200,1/200,1/200],[1/100,1/100,1/100],[1/50,1/50,1/50]],
                    [[1/100,1/100,1/100],[1/50,1/50,1/50],[1/20,1/20,1/20]],
                    [[1/50,1/50,1/50],[1/20,1/20,1/20],[1/10,1/10,1/10]]])
    
    # and specify the damage thresholds.
    depth_threshs = numpy.array([[700,600,500],[600,500,400],[500,400,300],[400,300,200]])
    wind_threshs = numpy.array([[400,375,350],[375,350,325],[350,325,300],[350,300,275]])    

    if not rail:
        # create the set of parameters for the sensitivity analysis.
        problem = {
                  'num_vars': 5,
                  'names': ['width', '4l_2l','2l_1l','cost','fragility'],
                  'bounds': [[2.7,4.6],[0,1],[0,1],[0,1],[1,4]]}
        
        # Generate samples
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)

    else:
         # create the set of parameters for the sensitivity analysis.
        problem = {
                  'num_vars': 4,
                  'names': ['width','2l_1l','cost','fragility'],
                  'bounds': [[3,5],[0,1],[0,1],[1,4]]}
        
        # Generate samples
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
    
    # prepare multiprocessing
    param_list = [param_values]*len(all_files)
    data_p_list = [data_path]*len(all_files)
    income_list = [income_lookup]*len(all_files)
    eq_curve_list = [eq_curve]*len(all_files)
    design_list = [design_tables]*len(all_files)
    depth_list = [depth_threshs]*len(all_files)
    wind_list = [wind_threshs]*len(all_files)
    rail_list = [rail]*len(all_files)

    # run bridge damage analysis parallel
    with Pool(cpu_count()-1) as pool: 
        collect_risks = pool.starmap(damage.regional_bridge,zip(all_files,data_p_list,param_list,income_list,eq_curve_list,design_list,depth_list,wind_list,rail_list),chunksize=1) 
    
    # and save all output.
    if not rail:
        pandas.concat(collect_risks).to_csv(os.path.join(data_path,'summarized','bridge_risk_road.csv'))
    else:
        pandas.concat(collect_risks).to_csv(os.path.join(data_path,'summarized','bridge_risk_rail.csv'))
        

def cyclone_damage(rail=False):  
    """
    Function to calculate the cyclone damage to road or railway assets for all regions.

    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.

    """ 
    # set data path
    data_path = load_config()['paths']['data']
  
    # set list of events
    events = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
    
    # get list of all files for which we have hazard intersection information
    
    # do this for roads
    if not rail:
        all_files = [os.path.join(data_path,'output_Cyc_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_full'))]
        
        # set parameter values
        problem = {
          'num_vars': 4,
          'names': ['x1', 'x2', 'x3','x4'],
          'bounds': [[5000,50000],[1000,10000],[500,5000],[0,500]]}
        
        # Generate samples
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
    
    # and for railway
    else:
        all_files = [os.path.join(data_path,'output_Cyc_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_rail_full'))]
     
        # set parameter values
        problem = {
          'num_vars': 3,
          'names': ['x1', 'x2', 'x3'],
          'bounds': [[5000,50000],[1000,10000],[0,1]]}
        
        # Generate samples
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
  
    # prepare multiprocessing
    data_paths = [data_path]*len(all_files)
    event_list = [events]*len(all_files)
    param_list = [param_values]*len(all_files)
    rail_list = [rail]*len(all_files)

   # run cyclone damage analysis parallel 
    with Pool(cpu_count()-1) as pool: 
        pool.starmap(damage.regional_cyclone,zip(all_files,data_paths,event_list,param_list,rail_list),chunksize=1) 


def earthquake_damage(rail=False):  
    """
    Function to calculate the earthquake damage to road or railway assets for all regions.

    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.

    """     
    # set data path
    data_path = load_config()['paths']['data']
    
    # load shapefile with country level information
    global_countries = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_countries.wbregion = global_countries.wbregion.str.replace('LCA','LAC')
    global_countries['wbregion'].loc[global_countries.wbregion.isnull()] = 'YHI'
    wbreg_lookup = dict(zip(global_countries['ISO_3digit'],global_countries['wbregion']))
    
    # import costs
    global_costs = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=7,header=0,index_col=0,skipfooter =45)
    global_costs.columns = ['SAS','SSA','MNA','EAP','LAC','ECA','YHI']
    
    # read csv file with information on paved and unpaved roads.
    paved_ratios = pandas.read_csv(os.path.join(data_path,'input_data','paved_ratios.csv'),index_col=[0,1])
    paved_ratios.index.names = ['ISO3','road_type']
    paved_ratios = paved_ratios.reset_index()
    
    # Load all files for which we have intersection data
    if not rail:
        all_files = [os.path.join(data_path,'output_EQ_full',x) for x in os.listdir(os.path.join(data_path,'output_EQ_full'))]
    else:
        all_files = [os.path.join(data_path,'output_EQ_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_EQ_rail_full'))]

    # set list of hazard events
    events = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475']

     # prepare multiprocessing
    data_paths = [data_path]*len(all_files)
    pav_cost_list = [global_costs]*len(all_files)
    pav_rat_list = [paved_ratios]*len(all_files)
    events_list = [events]*len(all_files)
    wbreg_list = [wbreg_lookup]*len(all_files)
    rail_list = [rail]*len(all_files)
    
   # run earthquake damage analysis parallel 
    with Pool(cpu_count()-1) as pool: 
        pool.starmap(damage.regional_earthquake,zip(all_files,data_paths,pav_cost_list,pav_rat_list,events_list,wbreg_list,rail_list),chunksize=1) 

              
def flood_damage(hazard,rail=False):  
    """
    Function to calculate the flood damage to road or railway assets for all regions.

    Arguments:
         *hzd* : abbrevation of the hazard we want to intersect. **FU** for 
         river flooding, **PU** for surface flooding and **CF** for coastal flooding.
        
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    """ 
    # set data path
    data_path = load_config()['paths']['data']
    
     # load shapefile with country level information
    global_countries = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_countries.wbregion = global_countries.wbregion.str.replace('LCA','LAC')
    global_countries['wbregion'].loc[global_countries.wbregion.isnull()] = 'YHI'
    wbreg_lookup = dict(zip(global_countries['ISO_3digit'],global_countries['wbregion']))
    
    # import curves
    flood_curve_paved = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=[1,2,3,4,5,6,7,8],
                                 sheet_name='Flooding',index_col=[0],skipfooter=9,header = [0,1])
    flood_curve_paved.columns = flood_curve_paved.columns.droplevel(0)
    flood_curve_unpaved = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=[11,12,13,14,15,16,17,18],
                                 sheet_name='Flooding',index_col=[0],skipfooter=9,header = [0,1])
    flood_curve_unpaved.columns = flood_curve_unpaved.columns.droplevel(0)
    
    # import cost values for different World Bank regions
    global_costs = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=7,header=0,index_col=0,skipfooter =45)
    global_costs.columns = ['SAS','SSA','MNA','EAP','LAC','ECA','YHI']
    
    # read csv file with information on paved and unpaved roads.
    paved_ratios = pandas.read_csv(os.path.join(data_path,'input_data','paved_ratios.csv'),index_col=[0,1])
    paved_ratios.index.names = ['ISO3','road_type']
    paved_ratios = paved_ratios.reset_index()
      
    # Load all files for which we have intersection data
    if not rail:
        all_files = [os.path.join(data_path,'output_{}_full'.format(hazard),x) for x in os.listdir(os.path.join(data_path,'output_{}_full'.format(hazard)))]
    else:
        all_files = [os.path.join(data_path,'output_{}_rail_full'.format(hazard),x) for x in os.listdir(os.path.join(data_path,'output_{}_rail_full'.format(hazard)))]
        
    # create list with all hazard events
    if hazard == 'FU':
        events = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
    elif hazard == 'PU':
        events = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500', 'PU-1000']
    elif hazard == 'CF':
        events = ['CF-10', 'CF-20', 'CF-50', 'CF-100', 'CF-200', 'CF-500', 'CF-1000']

    # prepare multiprocessing
    data_paths = [data_path]*len(all_files)
    pav_cost_list = [global_costs]*len(all_files)
    pav_rat_list = [paved_ratios]*len(all_files)
    cur_pav_list = [flood_curve_paved]*len(all_files)
    cur_unpav_list = [flood_curve_unpaved]*len(all_files)
    events_list = [events]*len(all_files)
    wbreg_list = [wbreg_lookup]*len(all_files)
    rail_list = [rail]*len(all_files)
 
   # run flood damage analysis parallel 
    with Pool(cpu_count()-1) as pool: 
       pool.starmap(damage.regional_flood,zip(all_files,data_paths,pav_cost_list,pav_rat_list,
                                                          cur_pav_list,cur_unpav_list,events_list,wbreg_list,rail_list),chunksize=1) 
    
def bridge_sensitivity(rail=False,region_count=1000):
    """
    Function to calculate the damage to bridges for all regions and all hazards.
    
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
        *region_count* : Default is **1000**. Change this number if you want to include a different amount of regions.
    """ 
    # set data path
    data_path  = load_config()['paths']['data']

    # get a list of all regions for which we can estimate the damages
    if not rail:
        all_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))]
    else:
        all_files = [os.path.join(data_path,'bridges_osm_rail',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_rail'))]
    
    # load csv with income group data and assign income group to regions
    incomegroups = pandas.read_csv(os.path.join(data_path,'input_data','incomegroups_2018.csv'),index_col=[0])
    income_lookup = dict(zip(incomegroups.CountryCode,incomegroups.GroupCode))
    
    # load earthquake curves
    eq_curve = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),sheet_name='bridge_curves',usecols=5)
    
    # create design standard table for floods and cyclones
    design_tables = numpy.array([[[1/200,1/200,1/200],[1/100,1/100,1/100],[1/50,1/50,1/50]],
                    [[1/100,1/100,1/100],[1/50,1/50,1/50],[1/20,1/20,1/20]],
                    [[1/50,1/50,1/50],[1/20,1/20,1/20],[1/10,1/10,1/10]]])
    
    # and specify the damage thresholds.
    depth_threshs = numpy.array([[700,600,500],[600,500,400],[500,400,300],[400,300,200]])
    wind_threshs = numpy.array([[400,375,350],[375,350,325],[350,325,300],[350,300,275]])    
    
    if not rail:
        # create the set of parameters for the sensitivity analysis.
        problem = {
                  'num_vars': 5,
                  'names': ['width', '4l_2l','2l_1l','cost','fragility'],
                  'bounds': [[2.7,4.6],[0,1],[0,1],[0,1],[1,4]]}
        
        # Generate samples and save them, to be used in uncertainty and sensitivity analysis of results
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
        param_values.tofile(os.path.join(data_path,'input_data','param_values_bridge_road.pkl'))

    else:
         # create the set of parameters for the sensitivity analysis.
        problem = {
                  'num_vars': 4,
                  'names': ['width','2l_1l','cost','fragility'],
                  'bounds': [[3,5],[0,1],[0,1],[1,4]]}
        
        # Generate samples and save them, to be used in uncertainty and sensitivity analysis of results
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
        param_values.tofile(os.path.join(data_path,'input_data','param_values_bridge_railway.pkl'))
    
    # prepare multiprocessing
    param_list = [param_values]*len(all_files)
    data_p_list = [data_path]*len(all_files)
    income_list = [income_lookup]*len(all_files)
    eq_curve_list = [eq_curve]*len(all_files)
    design_list = [design_tables]*len(all_files)
    depth_list = [depth_threshs]*len(all_files)
    wind_list = [wind_threshs]*len(all_files)
    rail_list = [rail]*len(all_files)

    # run bridge damage analysis parallel
    with Pool(cpu_count()-1) as pool: 
        collect_risks = pool.starmap(damage.regional_bridge,zip(all_files,data_p_list,param_list,income_list,eq_curve_list,design_list,depth_list,wind_list,rail_list),chunksize=1) 
    
    # and save all output.
    if not rail:
        pandas.concat(collect_risks).to_csv(os.path.join(data_path,'summarized','sa_bridge_road.csv'))
    else:
        pandas.concat(collect_risks).to_csv(os.path.join(data_path,'summarized','sa_bridge_rail.csv'))
        

def cyclone_sensitivity(rail=False,region_count=1000):  
    """
    Function to perform the caculations for a sensitivity analysis related 
    to cyclone damage to road or railway assets for all regions.
    
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
        *region_count* : Default is **1000**. Change this number if you want to include a different amount of regions.
        
    """
    # set data path
    data_path = load_config()['paths']['data']

   # set list of events
    events = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
    
    # get list of all files for which we have hazard intersection information
    
    # do this for roads
    if not rail:
        all_files = [os.path.join(data_path,'output_Cyc_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_full'))][:region_count]
        
        # set parameter values
        problem = {
          'num_vars': 4,
          'names': ['x1', 'x2', 'x3','x4'],
          'bounds': [[5000,50000],[1000,10000],[500,5000],[0,500]]}
        
        # Generate samples and save them, to be used in uncertainty and sensitivity analysis of results
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
        param_values.tofile(os.path.join(data_path,'input_data','param_values_cyc_road.pkl'))
            
    # and for railways
    else:
        all_files = [os.path.join(data_path,'output_Cyc_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_rail_full'))][:region_count]
     
        # set parameter values
        problem = {
          'num_vars': 3,
          'names': ['x1', 'x2', 'x3'],
          'bounds': [[5000,50000],[1000,10000],[0,1]]}
        
        # Generate samples and save them, to be used in uncertainty and sensitivity analysis of results
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
        param_values.tofile(os.path.join(data_path,'input_data','param_values_cyc_rail.pkl'))    
    

    # prepare multiprocessing
    data_paths = [data_path]*len(all_files)
    event_list = [events]*len(all_files)
    param_list = [param_values]*len(all_files)
    rail_list = [rail]*len(all_files)

    # run cyclone sensitivity analysis parallel and save outputs
    with Pool(cpu_count()-1) as pool: 
        if not rail:
            collect_output = pool.starmap(sensitivity.regional_cyclone,zip(all_files,data_paths,event_list,param_list,rail_list),chunksize=1) 
            pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','sa_cyc_road.csv'))            

        else:
            collect_output = pool.starmap(sensitivity.regional_cyclone,zip(all_files,data_paths,event_list,param_list,rail_list),chunksize=1) 
            pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','sa_cyc_rail.csv'))            


def earthquake_sensitivity(rail=False,region_count=1000):  
    """
    Function to perform the caculations for a sensitivity analysis related 
    to earthquake damage to road or railway assets for all regions.    

    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
        *region_count* : Default is **1000**. Change this number if you want to include a different amount of regions.


    """
    # set data path
    data_path = load_config()['paths']['data']
    
    # load shapefile with country level information
    global_countries = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_countries.wbregion = global_countries.wbregion.str.replace('LCA','LAC')
    global_countries['wbregion'].loc[global_countries.wbregion.isnull()] = 'YHI'
    wbreg_lookup = dict(zip(global_countries['ISO_3digit'],global_countries['wbregion']))
    
    # import costs
    global_costs = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=7,header=0,index_col=0,skipfooter =45)
    global_costs.columns = ['SAS','SSA','MNA','EAP','LAC','ECA','YHI']
    
    # read csv file with information on paved and unpaved roads.
    paved_ratios = pandas.read_csv(os.path.join(data_path,'input_data','paved_ratios.csv'),index_col=[0,1])
    paved_ratios.index.names = ['ISO3','road_type']
    paved_ratios = paved_ratios.reset_index()

    # Load all files for which we have intersection data
    if not rail:
        all_files = [os.path.join(data_path,'output_EQ_full',x) for x in os.listdir(os.path.join(data_path,'output_EQ_full'))][:region_count]
    else:
        all_files = [os.path.join(data_path,'output_EQ_full',x) for x in os.listdir(os.path.join(data_path,'output_EQ_rail_full'))][:region_count]
        
    # set list of hazard events
    events = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475']

     # prepare multiprocessing
    data_paths = [data_path]*len(all_files)
    pav_cost_list = [global_costs]*len(all_files)
    pav_rat_list = [paved_ratios]*len(all_files)
    events_list = [events]*len(all_files)
    wbreg_list = [wbreg_lookup]*len(all_files)
    rail_list = [rail]*len(all_files)
 
    # run earthquake sensitivity analysis parallel and save outputs
    with Pool(cpu_count()-1) as pool: 
        collect_output = pool.starmap(sensitivity.regional_earthquake,zip(all_files,data_paths,pav_cost_list,pav_rat_list,events_list,wbreg_list,rail_list),chunksize=1) 
        
    if not rail:
        pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','sa_eq_road.csv'))            
    else:
        pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','sa_eq_rail.csv'))            


def flood_sensitivity(hazard,rail=False,region_count=1000):  
    """
    Function to perform the caculations for a sensitivity analysis related 
    to flood damage to road or railway assets for all regions.

    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
        *region_count* : Default is **1000**. Change this number if you want to include a different amount of regions.

    """
    # set data path
    data_path = load_config()['paths']['data']
    
     # load shapefile with country level information
    global_countries = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_countries.wbregion = global_countries.wbregion.str.replace('LCA','LAC')
    global_countries['wbregion'].loc[global_countries.wbregion.isnull()] = 'YHI'
    wbreg_lookup = dict(zip(global_countries['ISO_3digit'],global_countries['wbregion']))
    
    # import curves, costs and paved vs unpaved ratios
    if not rail:
        flood_curve_paved = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=[1,2,3,4,5,6,7,8],
                                     sheet_name='Flooding',index_col=[0],skipfooter=9,header = [0,1])
        flood_curve_paved.columns = flood_curve_paved.columns.droplevel(0)
        flood_curve_unpaved = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=[11,12,13,14,15,16,17,18],
                                     sheet_name='Flooding',index_col=[0],skipfooter=9,header = [0,1])
        flood_curve_unpaved.columns = flood_curve_unpaved.columns.droplevel(0)
        
        global_costs = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=7,header=0,index_col=0,skipfooter =45)
        global_costs.columns = ['SAS','SSA','MNA','EAP','LAC','ECA','YHI']
        
        paved_ratios = pandas.read_csv(os.path.join(data_path,'input_data','paved_ratios.csv'),index_col=[0,1])
        paved_ratios.index.names = ['ISO3','road_type']
        paved_ratios = paved_ratios.reset_index()
        
    else:
        curve = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=[1,2,3,4,5,6,7,8],
                                 sheet_name='Flooding',index_col=[0],skipfooter=9,header = [0,1])
        curve.columns = curve.columns.droplevel(0)       
        
    # Load all files for which we have intersection data
    if not rail:
        all_files = [os.path.join(data_path,'output_{}_full'.format(hazard),x) for x in os.listdir(os.path.join(data_path,'output_{}_full'.format(hazard)))][:region_count]
    else:
        all_files = [os.path.join(data_path,'output_{}_rail_full'.format(hazard),x) for x in os.listdir(os.path.join(data_path,'output_{}_rail_full'.format(hazard)))][:region_count]
        
    # create list with all hazard events
    if hazard == 'FU':
        events = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
    elif hazard == 'PU':
        events = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500', 'PU-1000']
    elif hazard == 'CF':
        events = ['CF-10', 'CF-20', 'CF-50', 'CF-100', 'CF-200', 'CF-500', 'CF-1000']

    # prepare multiprocessing
    data_paths = [data_path]*len(all_files)
    events_list = [events]*len(all_files)
    wbreg_list = [wbreg_lookup]*len(all_files)
    
    if not rail:
        pav_cost_list = [global_costs]*len(all_files)
        pav_rat_list = [paved_ratios]*len(all_files)
        cur_pav_list = [flood_curve_paved]*len(all_files)
        cur_unpav_list = [flood_curve_unpaved]*len(all_files)
        hzd_list = [hazard]*len(all_files)
    else:
        data_paths = [data_path]*len(all_files)
        hazards = [hazard]*len(all_files)
        cur_list = [curve]*len(all_files)

    
   # run flood damage sensitivity analysis parallel and save outputs
    with Pool(cpu_count()-1) as pool: 
        if not rail:
            collect_output = pool.starmap(sensitivity.regional_flood,zip(all_files,data_paths,pav_cost_list,pav_rat_list,
                                                              cur_pav_list,cur_unpav_list,events_list,wbreg_list,hzd_list),chunksize=1) 
            pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','sa_{}_road.csv').format(hazard))

        else:
            collect_output = pool.starmap(sensitivity.regional_flood,zip(all_files,hazards,data_paths,cur_list,events_list,wbreg_list),chunksize=1) 
            pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','sa_{}_rail.csv').format(hazard))

       
