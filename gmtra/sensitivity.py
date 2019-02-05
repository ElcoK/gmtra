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

from utils import sum_tuples
from losses import road_flood,rail_flood,road_cyclone,rail_cyclone,road_earthquake,rail_earthquake

def calc_risk_total(x,hazard,RPS,events):
    collect_risks = []
    for y in range(50):
        collect_risks.append(integrate.simps([x[y] for x in x[events]][::-1], x=RPS[::-1]))
    return collect_risks

def regional_cyclone(cycfil,data_path,events,param_values,rail=False):
    """
    Function to estimate the summary statistics of all cyclone damages in a region to road assets

    Arguments:
        *cycfil* : path to the .feather file with all bridges of a region.
        *data_path* : file path to location of all data.
        *events* : A list with the unique cyclone events.
        *param_values* : A NumPy Array with sets of parameter values we would like to test.       
        
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    Returns:
        *DataFrame* : a pandas DataFrame with summary damage statistics for the loaded region.
        
    """    
    #    try:
    if not rail:
        all_tree_cov = [os.path.join(data_path,'tree_cover_road',x) for x in os.listdir(os.path.join(data_path,'tree_cover_road'))]
    else:
        all_tree_cov = [os.path.join(data_path,'tree_cover_rail',x) for x in os.listdir(os.path.join(data_path,'tree_cover_rail'))]
        
    
    df_cyc = pandas.read_feather(cycfil)
    val_cols = [x for x in list(df_cyc.columns) if 'val' in x]
    df_cyc = df_cyc.loc[~(df_cyc[val_cols] < 151).all(axis=1)]
    
    if len(df_cyc) > 0:
        region = list(df_cyc.region.unique())[0]
        df_cyc = df_cyc.rename({'val_Cyc_rp100':'val_Cyc_rp500',
                                'val_Cyc_rp500':'val_Cyc_rp1000', 
                                'val_Cyc_rp1000':'val_Cyc_rp100',
                                'length_Cyc_rp100':'length_Cyc_rp500',
                                'length_Cyc_rp500':'length_Cyc_rp1000', 
                                'length_Cyc_rp1000':'length_Cyc_rp100'}, axis='columns')
        df_tree = pandas.read_feather([x for x in all_tree_cov if os.path.split(cycfil)[1][:-7] in x][0])
        df_cyc = df_cyc.merge(df_tree[['osm_id','Tree_Dens']],left_on='osm_id',right_on='osm_id')
        df_cyc = df_cyc.loc[~(df_cyc['Tree_Dens'] <= 0)]
        df_cyc['Tree_Dens'].loc[(df_cyc['Tree_Dens'] >= 10000)] = 10000
        df_cyc['fail_prob'] = df_cyc.Tree_Dens/10000
        df_output = pandas.DataFrame(columns=events,index=df_cyc.index).fillna(0)
        df_cyc = pandas.concat([df_cyc,df_output],axis=1)
        tqdm.pandas(desc=region)

        if not rail:
            df_cyc = df_cyc.progress_apply(lambda x: road_cyclone(x,events,param_values,sensitivity=True),axis=1)
        else:
            df_cyc = df_cyc.progress_apply(lambda x: rail_cyclone(x,events,param_values),axis=1)

        df_cyc['risk'] = df_cyc.apply(lambda x: calc_risk_total(x,'Cyc',[1/50,1/100,1/250,1/500,1/1000],events),axis=1)
        df_cyc = df_cyc.drop([x for x in list(df_cyc.columns) if (x in events) | ('val_' in x) | ('length_' in x)],axis=1)    
        df_cyc.reset_index(inplace=True,drop=True)

        df_cyc.to_csv(os.path.join(data_path,'Cyc_sensitivity','{}.csv'.format(region)))
    
    else:
        return None

def regional_earthquake(file,data_path,global_costs,paved_ratios,events,wbreg_lookup,rail=False):
    """
    Function to estimate the summary statistics of all earthquake damages in a region to road assets

    Arguments:
        *file* : path to the .feather file with all bridges of a region.
        *data_path* : file path to location of all data.
        *global_costs* : A pandas DataFrame with the total cost for different roads in different World Bank regions. These values 
        are based on the ROCKS database.
        *paved_ratios* : A pandas DataFrame with road pavement percentages per country for each road type.        
        *events* : A list with the unique earthquake events.
        *wbreg_lookup* : a dictioniary that relates countries (in ISO3 codes) with World Bank regions.        
        
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.        

    Returns:
        *DataFrame* : a pandas DataFrame with summary damage statistics for the loaded region.
        
    """    
    try:
        df = pandas.read_feather(file)
        val_cols = [x for x in list(df.columns) if 'val' in x]
        region = list(df.region.unique())[0]
        df = df.loc[~(df[val_cols] < 92).all(axis=1)]        
        
        if len(df) == 0:
            print('No shaked assets in {}'.format(region))
            return None
        
        if os.path.exists(os.path.join(data_path,'EQ_impacts','{}.csv'.format(region))):
            return None
    
        if len(df) > 0:
            df = df.rename({'val_EQ_rp250':'val_EQ_rp475',
                                        'val_EQ_rp475':'val_EQ_rp1500', 
                                        'val_EQ_rp975':'val_EQ_rp250',
                                        'val_EQ_rp1500':'val_EQ_rp2475', 
                                        'val_EQ_rp2475':'val_EQ_rp975',
                                        'length_EQ_rp250':'length_EQ_rp475',
                                        'length_EQ_rp475':'length_EQ_rp1500',
                                        'length_EQ_rp975':'length_EQ_rp250',
                                        'length_EQ_rp1500':'length_EQ_rp2475',
                                        'length_EQ_rp2475':'length_EQ_rp975'}, axis='columns')
            if not rail:
                df_liq = pandas.read_feather(os.path.join(data_path,'liquefaction_road','{}_liq.ft'.format(region)))
            else:
                df_liq = pandas.read_feather(os.path.join(data_path,'liquefaction_rail','{}_liq.ft'.format(region)))
              
            df = df.merge(df_liq[['osm_id','liquefaction']],left_on='osm_id',right_on='osm_id')
            df = df.loc[~(df['liquefaction'] <= 1)]
            df = df.loc[~(df['liquefaction'] > 5)]
            if len(df) == 0:
                print('No liquid roads in {}'.format(region))
                return None            
  
        sorted_all = [[10,25,40],[20,40,60],[30,55,80],[40,70,100]]
        frag_tables = []

        if not rail:
            param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values_eq.pkl'))[x:x+4] 
            for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))), 4)]

            for sort_rand in sorted_all:
                EQ_fragility = numpy.array([[0]+sort_rand+[100],[0,0]+sort_rand,[0,0,0]+sort_rand[:2],[0,0,0,0]+sort_rand[:1],[0,0,0,0,0]]).T
                df_EQ_frag = pandas.DataFrame(EQ_fragility,columns=[5,4,3,2,1],index=[(0,92),(93,180),(181,340),(341,650),(651,5000)])
                frag_dict = df_EQ_frag.stack(0).to_dict()
                frag_tables.append(frag_dict)
        else:
            param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values_eq_rail.pkl'))[x:x+3] 
                        for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values_eq_rail.pkl'))), 3)]

            for sort_rand in sorted_all:
                EQ_fragility = numpy.array([[10]+sort_rand+[100],[0,10]+sort_rand,[0,0,10]+sort_rand[:2],[0,0,0,0]+sort_rand[:1],[0,0,0,0,0]]).T
                df_EQ_frag = pandas.DataFrame(EQ_fragility,columns=[5,4,3,2,1],index=[(0,92),(93,180),(181,340),(341,650),(651,5000)])
                frag_dict = df_EQ_frag.stack(0).to_dict()
                frag_tables.append(frag_dict)
    
        try:
            if not rail:
                all_bridge_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))]
            else:
                all_bridge_files = [os.path.join(data_path,'bridges_osm_rail',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_rail'))]
            bridges = list(pandas.read_feather([x for x in all_bridge_files if os.path.split(file)[1][:-6] in x][0])['osm_id'])
            df = df.loc[~(df['osm_id'].isin(bridges))]
        except:
            None
        
        df_output = pandas.DataFrame(columns=events,index=df.index).fillna(0)
        df = pandas.concat([df,df_output],axis=1)
    
        tqdm.pandas(desc = region)
        if not rail:
            df = df.progress_apply(lambda x: road_earthquake(x,global_costs,paved_ratios,frag_tables,events,wbreg_lookup,param_values,val_cols),axis=1)
        else:
            df = df.progress_apply(lambda x: rail_earthquake(x,frag_tables,events,param_values,val_cols),axis=1)

        df.reset_index(inplace=True,drop=True)
        
        df.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples).to_csv(os.path.join(data_path,'EQ_impacts','{}.csv'.format(region)))
        
        return df.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples)

    except Exception as e:
        print('Failed to finish {} because of {}!'.format(file,e))

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