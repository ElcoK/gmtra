# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:46:10 2019

@author: cenv0574
"""

import os
import numpy
import pandas
import geopandas

from pathos.multiprocessing import Pool,cpu_count
from SALib.sample import morris

from gmtra.utils import load_config
import gmtra.sensitivity as sensitivity
import gmtra.losses as losses


def bridge_losses(rail=False):
    """
    Function to calculate the damage to bridges for all regions and all hazards.
    
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    """ 
    data_path  = load_config()['paths']['data']

    if not rail:
        all_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))]
    else:
        all_files = [os.path.join(data_path,'bridges_osm_rail',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_rail'))]
    
    incomegroups = pandas.read_csv(os.path.join(data_path,'input_data','incomegroups_2018.csv'),index_col=[0])
    income_lookup = dict(zip(incomegroups.CountryCode,incomegroups.GroupCode))
    
    eq_curve = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),sheet_name='bridge_curves',usecols=5)
    
    design_tables = numpy.array([[[1/200,1/200,1/200],[1/100,1/100,1/100],[1/50,1/50,1/50]],
                    [[1/100,1/100,1/100],[1/50,1/50,1/50],[1/20,1/20,1/20]],
                    [[1/50,1/50,1/50],[1/20,1/20,1/20],[1/10,1/10,1/10]]])
    
    depth_threshs = numpy.array([[700,600,500],[600,500,400],[500,400,300],[400,300,200]])
    wind_threshs = numpy.array([[400,375,350],[375,350,325],[350,325,300],[350,300,275]])    

    problem = {
              'num_vars': 5,
              'names': ['width', '4l_2l','2l_1l','cost','fragility'],
              'bounds': [[2.7,4.6],[0,1],[0,1],[0,1],[1,4]]}
    
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

    with Pool(cpu_count()-1) as pool: 
        collect_risks = pool.starmap(losses.regional_bridge,zip(all_files,data_p_list,param_list,income_list,eq_curve_list,design_list,depth_list,wind_list,rail_list),chunksize=1) 
    
    if not rail:
        pandas.concat(collect_risks).to_csv(os.path.join(data_path,'summarized','bridge_risk_road.csv'))
    else:
        pandas.concat(collect_risks).to_csv(os.path.join(data_path,'summarized','bridge_risk_rail.csv'))
        

def cyclone_losses(rail=False):  
    """
    Function to calculate the cyclone damage to road or railway assets for all regions.

    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.

    """ 
    data_path = load_config()['paths']['data']
   
    events = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
    
    
    if not rail:
        all_files = [os.path.join(data_path,'output_Cyc_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_full'))]
        
        problem = {
          'num_vars': 4,
          'names': ['x1', 'x2', 'x3','x4'],
          'bounds': [[5000,50000],[1000,10000],[500,5000],[0,500]]}
        
        # Generate samples
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
            
    else:
        all_files = [os.path.join(data_path,'output_Cyc_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_rail_full'))]
     
        problem = {
          'num_vars': 3,
          'names': ['x1', 'x2', 'x3'],
          'bounds': [[5000,50000],[1000,10000],[0,1]]}
        
        # Generate samples
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
  

    data_paths = [data_path]*len(all_files)
    event_list = [events]*len(all_files)
    param_list = [param_values]*len(all_files)
    rail_list = [rail]*len(all_files)

    
    with Pool(cpu_count()-1) as pool: 
        pool.starmap(losses.regional_cyclone,zip(all_files,data_paths,event_list,param_list,rail_list),chunksize=1) 


def earthquake_losses(rail=False):  
    """
    Function to calculate the earthquake damage to road or railway assets for all regions.

    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.

    """     
    data_path = load_config()['paths']['data']
    
    global_countries = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_countries.wbregion = global_countries.wbregion.str.replace('LCA','LAC')
    global_countries['wbregion'].loc[global_countries.wbregion.isnull()] = 'YHI'
    wbreg_lookup = dict(zip(global_countries['ISO_3digit'],global_countries['wbregion']))
    
    # import curves
    global_costs = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=7,header=0,index_col=0,skipfooter =45)
    global_costs.columns = ['SAS','SSA','MNA','EAP','LAC','ECA','YHI']
    
    paved_ratios = pandas.read_csv(os.path.join(data_path,'input_data','paved_ratios.csv'),index_col=[0,1])
    paved_ratios.index.names = ['ISO3','road_type']
    paved_ratios = paved_ratios.reset_index()
    
    # Load samples
    all_files = [os.path.join(data_path,'output_EQ_full',x) for x in os.listdir(os.path.join(data_path,'output_EQ_full'))]
    events = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475']

    data_paths = [data_path]*len(all_files)
    pav_cost_list = [global_costs]*len(all_files)
    pav_rat_list = [paved_ratios]*len(all_files)
    events_list = [events]*len(all_files)
    wbreg_list = [wbreg_lookup]*len(all_files)
    rail_list = [rail]*len(all_files)
    
    
    with Pool(cpu_count()-1) as pool: 
        pool.starmap(losses.regional_earthquake,zip(all_files,data_paths,pav_cost_list,pav_rat_list,events_list,wbreg_list,rail_list),chunksize=1) 

              
def flood_losses(hazard,rail=False):  
    """
    Function to calculate the flood damage to road or railway assets for all regions.

    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    """ 
    data_path = load_config()['paths']['data']
    
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
    
    global_costs = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=7,header=0,index_col=0,skipfooter =45)
    global_costs.columns = ['SAS','SSA','MNA','EAP','LAC','ECA','YHI']
    
    paved_ratios = pandas.read_csv(os.path.join(data_path,'input_data','paved_ratios.csv'),index_col=[0,1])
    paved_ratios.index.names = ['ISO3','road_type']
    paved_ratios = paved_ratios.reset_index()
       
    if not rail:
        all_files = [os.path.join(data_path,'output_{}_full'.format(hazard),x) for x in os.listdir(os.path.join(data_path,'output_{}_full'.format(hazard)))]
    else:
        all_files = [os.path.join(data_path,'output_{}_rail_full'.format(hazard),x) for x in os.listdir(os.path.join(data_path,'output_{}_rail_full'.format(hazard)))]
        
    if hazard == 'FU':
        events = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
    elif hazard == 'PU':
        events = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500', 'PU-1000']
    elif hazard == 'CF':
        events = ['CF-10', 'CF-20', 'CF-50', 'CF-100', 'CF-200', 'CF-500', 'CF-1000']

    data_paths = [data_path]*len(all_files)
    pav_cost_list = [global_costs]*len(all_files)
    pav_rat_list = [paved_ratios]*len(all_files)
    cur_pav_list = [flood_curve_paved]*len(all_files)
    cur_unpav_list = [flood_curve_unpaved]*len(all_files)
    events_list = [events]*len(all_files)
    wbreg_list = [wbreg_lookup]*len(all_files)
    rail_list = [rail]*len(all_files)
 
    with Pool(cpu_count()-1) as pool: 
       pool.starmap(losses.regional_flood,zip(all_files,data_paths,pav_cost_list,pav_rat_list,
                                                          cur_pav_list,cur_unpav_list,events_list,wbreg_list,rail_list),chunksize=1) 
    

def cyclone_sensitivity(rail=False,region_count=1000):  
    """
    Function to perform the caculations for a sensitivity analysis related 
    to cyclone damage to road or railway assets for all regions.
    
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
        *region_count* : Default is **1000**. Change this number if you want to include more regions.
        
    """

    data_path = load_config()['paths']['data']
    
    events = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
    
    if not rail:
        all_files = [os.path.join(data_path,'output_Cyc_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_full'))][:region_count]
        
        problem = {
          'num_vars': 4,
          'names': ['x1', 'x2', 'x3','x4'],
          'bounds': [[5000,50000],[1000,10000],[500,5000],[0,500]]}
        
        # Generate samples
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
        param_values.tofile(os.path.join(data_path,'input_data','param_values_cyc_road.pkl'))
            
    else:
        all_files = [os.path.join(data_path,'output_Cyc_rail_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_rail_full'))][:region_count]
     
        problem = {
          'num_vars': 3,
          'names': ['x1', 'x2', 'x3'],
          'bounds': [[5000,50000],[1000,10000],[0,1]]}
        
        # Generate samples
        param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
        param_values.tofile(os.path.join(data_path,'input_data','param_values_cyc_rail.pkl'))    
    

    data_paths = [data_path]*len(all_files)
    event_list = [events]*len(all_files)
    param_list = [param_values]*len(all_files)
    rail_list = [rail]*len(all_files)

    with Pool(40) as pool: 
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
        
        *region_count* : Default is **1000**. Change this number if you want to include more regions.


    """
    data_path = load_config()['paths']['data']
    
    global_countries = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_countries.wbregion = global_countries.wbregion.str.replace('LCA','LAC')
    global_countries['wbregion'].loc[global_countries.wbregion.isnull()] = 'YHI'
    wbreg_lookup = dict(zip(global_countries['ISO_3digit'],global_countries['wbregion']))
    
    # import curves
    if not rail:
        all_files = [os.path.join(data_path,'output_EQ_full',x) for x in os.listdir(os.path.join(data_path,'output_EQ_full'))][:region_count]
    else:
        all_files = [os.path.join(data_path,'output_EQ_full',x) for x in os.listdir(os.path.join(data_path,'output_EQ_rail_full'))][:region_count]

    global_costs = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=7,header=0,index_col=0,skipfooter =45)
    global_costs.columns = ['SAS','SSA','MNA','EAP','LAC','ECA','YHI']
    
    paved_ratios = pandas.read_csv(os.path.join(data_path,'input_data','paved_ratios.csv'),index_col=[0,1])
    paved_ratios.index.names = ['ISO3','road_type']
    paved_ratios = paved_ratios.reset_index()
        
    # Load samples
    events = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475']

    data_paths = [data_path]*len(all_files)
    pav_cost_list = [global_costs]*len(all_files)
    pav_rat_list = [paved_ratios]*len(all_files)
    events_list = [events]*len(all_files)
    wbreg_list = [wbreg_lookup]*len(all_files)
    rail_list = [rail]*len(all_files)
 
    with Pool(40) as pool: 
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
        *region_count* : Default is **1000**. Change this number if you want to include more regions.

    """
    data_path = load_config()['paths']['data']
    
    global_countries = geopandas.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
    global_countries.wbregion = global_countries.wbregion.str.replace('LCA','LAC')
    global_countries['wbregion'].loc[global_countries.wbregion.isnull()] = 'YHI'
    wbreg_lookup = dict(zip(global_countries['ISO_3digit'],global_countries['wbregion']))
    
    # import curves
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
    
    if not rail:
        all_files = [os.path.join(data_path,'output_{}_full'.format(hazard),x) for x in os.listdir(os.path.join(data_path,'output_{}_full'.format(hazard)))][:region_count]
    else:
        all_files = [os.path.join(data_path,'output_{}_rail_full'.format(hazard),x) for x in os.listdir(os.path.join(data_path,'output_{}_rail_full'.format(hazard)))][:region_count]
        
    if hazard == 'FU':
        events = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
    elif hazard == 'PU':
        events = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500', 'PU-1000']
    elif hazard == 'CF':
        events = ['CF-10', 'CF-20', 'CF-50', 'CF-100', 'CF-200', 'CF-500', 'CF-1000']

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

    
    with Pool(40) as pool: 
        if not rail:
            collect_output = pool.starmap(sensitivity.regional_flood,zip(all_files,data_paths,pav_cost_list,pav_rat_list,
                                                              cur_pav_list,cur_unpav_list,events_list,wbreg_list,hzd_list),chunksize=1) 
            pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','sa_{}_road.csv').format(hazard))

        else:
            collect_output = pool.starmap(sensitivity.regional_flood,zip(all_files,hazards,data_paths,cur_list,events_list,wbreg_list),chunksize=1) 
            pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','sa_{}_rai.csv').format(hazard))

