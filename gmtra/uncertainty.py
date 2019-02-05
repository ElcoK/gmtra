# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:53:41 2018

@author: cenv0574
"""

import os
import pandas
import numpy
import tqdm
from scipy import integrate

from losses import road_flood,rail_flood

def calc_risk_total(x,hazard,RPS,events):
    collect_risks = []
    for y in range(50):
        collect_risks.append(integrate.simps([x[y] for x in x[events]][::-1], x=RPS[::-1]))
    return collect_risks

def regional_flood(file,data_path,global_costs,paved_ratios,flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,hazard,rail=False):
    """
    Function to estimate the summary statistics of all flood damages in a region to road assets
    
    Arguments:
        *file* : path to the .feather file with all bridges of a region.
        *data_path* : file path to location of all data.
        *global_costs* : A pandas DataFrame with the total cost for different roads in different World Bank regions. These values 
        are based on the ROCKS database.
        *paved_ratios* : A pandas DataFrame with road pavement percentages per country for each road type.
        *flood_curve_paved* : A pandas DataFrame with a set of damage curves for paved roads.
        *flood_curve_unpaved* : A pandas DataFrame with a set of damage curves for unpaved roads.
        *events* : A list with the unique flood events.
        *wbreg_lookup* : a dictioniary that relates a country ID (ISO3 code) with its World Bank region.
        
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
    
    Returns:
        *DataFrame* : a pandas DataFrame with all damage results for the loaded region per road or railway asset type.
    """ 
    try:
        df = pandas.read_feather(file)
        region = df.region.unique()[0]
        val_cols = [x for x in list(df.columns) if 'val' in x]
        df = df.loc[~(df[val_cols] == 0).all(axis=1)]
        
        if len(df) == 0:
            print('No flooded roads in {}'.format(region))
            return None
        
        if len(os.listdir(os.path.join(data_path,'{}_sensitivity').format(hazard))) > 500:
            return None
        
        param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))[x:x+4] for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))), 4)]
    
        try:
            if not rail:
                param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))[x:x+4] 
                    for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))), 4)]

                all_bridge_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))]
                bridges = list(pandas.read_feather([x for x in all_bridge_files if os.path.split(file)[1][:-6] in x][0])['osm_id'])
                df = df.loc[~(df['osm_id'].isin(bridges))]
            else:
                param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values_fl_rail.pkl'))[x:x+3] 
                    for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values_fl_rail.pkl'))), 3)]

                all_bridge_files = [os.path.join(data_path,'bridges_osm_rail',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_rail'))]
                bridges = list(pandas.read_feather([x for x in all_bridge_files if os.path.split(file)[1][:-6] in x][0])['osm_id'])
                df = df.loc[~(df['osm_id'].isin(bridges))]                

        except:
            None
        
        df_output = pandas.DataFrame(columns=events,index=df.index).fillna(0)
        df = pandas.concat([df,df_output],axis=1)
    
        tqdm.pandas(desc = region)

        if not rail:
            df = df.progress_apply(lambda x: road_flood(x,global_costs,paved_ratios,
                                                         flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,param_values,val_cols,sensitivity=True),axis=1)
        else:
            curve = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=[1,2,3,4,5,6,7,8],
                                 sheet_name='Flooding',index_col=[0],skipfooter=9,header = [0,1])
            curve.columns = curve.columns.droplevel(0)
            
            df = df.progress_apply(lambda x: rail_flood(x,
                                                    curve,events,param_values,val_cols,wbreg_lookup,sensitivity=True),axis=1)         
    
        if (hazard == 'PU') | (hazard == 'FU'):
            df['risk'] = df.apply(lambda x: calc_risk_total(x,hazard,[1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000],events),axis=1)
        else:
            df['risk'] = df.apply(lambda x: calc_risk_total(x,hazard,[1/10,1/20,1/50,1/100,1/200,1/500,1/1000],events),axis=1)
        
        df = df.drop([x for x in list(df.columns) if (x in events) | ('val_' in x) | ('length_' in x)],axis=1)    
        df.reset_index(inplace=True,drop=True)
        
        df.to_csv(os.path.join(data_path,'{}_sensitivity'.format(hazard),'{}.csv'.format(region)))
        
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(file,e))