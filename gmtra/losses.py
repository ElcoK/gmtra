# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:53:23 2018

@author: cenv0574
"""

import os
import re
import sys
import geopandas
import pandas
import numpy

from random import shuffle
from pathos.multiprocessing import Pool,cpu_count
from SALib.sample import morris
from tqdm import tqdm

sys.path.append(os.path.join( '..'))
from miriam_py.utils import load_config,square_m2_cost_range,monetary_risk,sum_tuples

pandas.set_option('chained_assignment',None)


def asset_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps):
    """
    Arguments:
        
    
    Returns:
        
    """    
    uncer_output = []
    for param in param_values:
        depth_thresh = depth_threshs[int(param[4]-1)]
        cost = x.cost[0]+((x.cost[1]-x.cost[0])*param[3])
        if x.road_type == 'primary':
            rps = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] < design_table[0][0]]
            rps_not = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] >= design_table[0][0]]

            frag = numpy.array([0]*len(rps_not)+list((x[rps] > depth_thresh[0])*1))
            uncer_output.append((cost*x.length*param[0]*param[1]*4+cost*x.length*param[0]*(1-param[1])*2)*frag)
        elif x.road_type == 'secondary':
            rps = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] < design_table[1][0]]
            rps_not = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] >= design_table[1][0]]

            frag = numpy.array([0]*len(rps_not)+list((x[rps] > depth_thresh[1])*1))
            uncer_output.append((cost*x.length*param[0]*param[1]*4+cost*x.length*param[0]*(1-param[1])*2)*frag)
        else:
            rps = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] < design_table[2][0]]
            rps_not = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] >= design_table[2][0]]

            frag = numpy.array([0]*len(rps_not)+list((x[rps] > depth_thresh[2])*1))
            uncer_output.append((cost*x.length*param[0]*param[1]*4+cost*x.length*param[0]*(1-param[1])*2)*frag)           
    
    return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                            numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                            numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                            numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))))


def asset_bridge_earthquake(x,eq_curve,param_values,vals_EQ,events,all_rps):
    """
    Arguments:
        
    
    Returns:
        
    """        
    uncer_output = []
    for param in param_values:
        curve = eq_curve.iloc[:,int(param[4])-1]
        frag = numpy.interp(list(x[[x for x in vals_EQ ]]),list(curve.index), curve.values)
        cost = x.cost[0]+((x.cost[1]-x.cost[0])*param[3])

        if x.road_type == 'primary':
            uncer_output.append((cost*x.length*param[0]*param[1]*4+cost*x.length*param[0]*(1-param[1])*2)*frag)
        else:
            uncer_output.append((cost*x.length*param[0]*param[2]*2+cost*x.length*param[0]*(1-param[2])*1)*frag)     
    
    return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                            numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                            numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                            numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))))

def asset_cyclone(x,events,param_values):
    """
    Arguments:
        
    
    Returns:
        
    """    
    uncer_output = []
    for param in param_values:
        cleanup_cost_dict = {'primary' : param[0],'secondary' : param[1],'tertiary' : param[2],'track' : param[3],'other' : param[3], 'nodata': param[3]}
        uncer_events = []
        for event in events:
            if x['val_{}'.format(event)] > 150:
                uncer_events.append(cleanup_cost_dict[x.road_type]*x['length_{}'.format(event)]*x.fail_prob)
            else:
                uncer_events.append(0) 
        uncer_output.append(numpy.asarray(uncer_events))

    x[events] = list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
    numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),
    numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),numpy.percentile(numpy.asarray(uncer_output), 100,axis=0)))

    return x

def asset_earthquake(x,global_costs,paved_ratios,frag_tables,events,wbreg_lookup,param_values,val_cols):
    """
    Arguments:
        
    
    Returns:
        
    """       
    wbreg = wbreg_lookup[x.country]

    ratios = paved_ratios.loc[(paved_ratios.ISO3 == x.country) & (paved_ratios.road_type == x.road_type)]
    costs = global_costs[wbreg]
    lengths = numpy.array(x[[x for x in list(x.index) if 'length_' in x]])

    loss_ratios = []
    uncer_output = []
    for param in param_values:
        loss_ratios = []
        for event in events:
            loss_ratios.append(frag_tables[int(param[2])][[z for z in frag_tables[int(param[2])] if (z[0][0] <= x['val_{}'.format(event)] <= z[0][1]) & (z[1] == x.liquefaction)][0]]/100)

        loss_ratios = numpy.array(loss_ratios)
        if x.road_type == 'primary':
            uncer_output.append(list((ratios.paved/100)*(param[0]*param[3]*costs['Paved 4L']+(1-param[0])*param[3]*costs['Paved 2L']))[0]*loss_ratios*lengths + 
            list((ratios.unpaved/100)*costs.Gravel)[0]*loss_ratios*lengths)
        elif x.road_type == 'secondary':
            uncer_output.append(list((ratios.paved/100)*(param[1]*param[3]*costs['Paved 2L']+(1-param[1])*param[3]*150000))[0]*loss_ratios*lengths + 
            list((ratios.unpaved/100)*costs.Gravel)[0]*loss_ratios*lengths)   
        else:
            uncer_output.append(list((ratios.paved/100)*(ratios.Paved_4L*param[3]*costs['Paved 2L']+ratios.Paved_2L*param[3]*150000))[0]*loss_ratios*lengths + 
    list((ratios.unpaved/100)*costs.Gravel)[0]*loss_ratios*lengths)    

    x[events] = list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
    numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
    numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),numpy.percentile(numpy.asarray(uncer_output), 100,axis=0)))

    return x

def asset_flood(x,global_costs,paved_ratios,flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,param_values,val_cols):
    """
    Arguments:
        
    
    Returns:
        
    """     
    wbreg = wbreg_lookup[x.country]

    curve_paved = flood_curve_paved.loc[:,wbreg]
    curve_unpaved = flood_curve_unpaved.loc[:,wbreg]
    ratios = paved_ratios.loc[(paved_ratios.ISO3 == x.country) & (paved_ratios.road_type == x.road_type)]
    costs = global_costs[wbreg]

    uncer_output = []
    for param in param_values:
        paved_frag = numpy.interp(list(x[[x for x in val_cols if 'val' in x]]),list(curve_paved.index),(curve_paved.values)*param[2])
        unpaved_frag = numpy.interp(list(x[[x for x in val_cols if 'val' in x]]),list(curve_unpaved.index), curve_unpaved.values)

        if x.road_type == 'primary':
            uncer_output.append(list((ratios.paved/100)*(param[0]*param[3]*costs['Paved 4L']+(1-param[0])*param[3]*costs['Paved 2L']))[0]*paved_frag + 
            list((ratios.unpaved/100)*costs.Gravel)[0]*unpaved_frag)
        elif x.road_type == 'secondary':
            uncer_output.append(list((ratios.paved/100)*(param[1]*param[3]*costs['Paved 2L']+(1-param[1])*param[3]*150000))[0]*paved_frag + 
            list((ratios.unpaved/100)*costs.Gravel)[0]*unpaved_frag)   
        else:
            uncer_output.append(list((ratios.paved/100)*(ratios.Paved_4L*param[3]*costs['Paved 4L']+ratios.Paved_2L*param[3]*150000))[0]*paved_frag + 
    list((ratios.unpaved/100)*costs.Gravel)[0]*unpaved_frag)    

    x[events] = list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
    numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
    numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
    numpy.percentile(numpy.asarray(uncer_output), 100,axis=0)))

    return x

  
def regional_bridge(file,data_path,param_values,income_lookup,eq_curve,design_tables,depth_threshs,wind_threshs):
    """
    Arguments:
        
    
    Returns:
        
    """    
    df = pandas.read_feather(file)

    df = df.loc[~(df.length < 6)]
    df = df.loc[~(df['road_type'] == 'nodata')]
    df['cost'] = df.apply(lambda x: square_m2_cost_range(x),axis=1)
    df.drop([x for x in list(df.columns) if 'length_' in x],axis=1,inplace=True)
    vals_EQ = [x for x in list(df.columns) if 'val_EQ' in x]

    df['IncomeGroup'] = income_lookup[list(df.country.unique())[0]]
    region = list(df.region.unique())[0]

    df = df.rename({'val_Cyc_rp100':'val_Cyc_rp500',
                    'val_Cyc_rp500':'val_Cyc_rp1000', 
                    'val_Cyc_rp1000':'val_Cyc_rp100',
                    'length_Cyc_rp100':'length_Cyc_rp500',
                    'length_Cyc_rp500':'length_Cyc_rp1000', 
                    'length_Cyc_rp1000':'length_Cyc_rp100',
                    'val_EQ_rp250':'val_EQ_rp475',
                    'val_EQ_rp475':'val_EQ_rp1500', 
                    'val_EQ_rp975':'val_EQ_rp250',
                    'val_EQ_rp1500':'val_EQ_rp2475', 
                    'val_EQ_rp2475':'val_EQ_rp975',
                    'length_EQ_rp250':'length_EQ_rp475',
                    'length_EQ_rp475':'length_EQ_rp1500',
                    'length_EQ_rp975':'length_EQ_rp250',
                    'length_EQ_rp1500':'length_EQ_rp2475',
                    'length_EQ_rp2475':'length_EQ_rp975'}, axis='columns')

    all_rps = [1/250,1/475,1/975,1/1500,1/2475]
    events = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475']
    tqdm.pandas('Earthquake '+region)
    df['EQ_risk'] = df.progress_apply(lambda x: asset_bridge_earthquake(x,eq_curve,param_values,vals_EQ,events,all_rps),axis=1)

    if df['IncomeGroup'].unique()[0] == 'HIC':
        design_table = design_tables[0]
    elif df['IncomeGroup'].unique()[0] == 'UMC':
        design_table = design_tables[1]
    else:
        design_table = design_tables[2]

    all_rps = [1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000]
    events = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
    tqdm.pandas(desc='Fluvial '+region)
    df['FU_risk'] = df.progress_apply(lambda x: asset_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1)


    events = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500','PU-1000']
    tqdm.pandas(desc='Pluvial '+region)
    df['PU_risk'] = df.progress_apply(lambda x: asset_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1)


    all_rps = [1/50,1/100,1/250,1/500,1/1000]
    events = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
    tqdm.pandas(desc='Cyclones '+region)
    df['Cyc_risk'] = df.progress_apply(lambda x: asset_bridge_flood_cyclone(x,design_table,wind_threshs,param_values,events,all_rps),axis=1)

    save_df = df.groupby(['road_type','region','country','IncomeGroup'])['EQ_risk','FU_risk','PU_risk','Cyc_risk'].agg(sum_tuples)

    save_df.to_csv(os.path.join(data_path,'bridge_road_risk','{}.csv'.format(region)))

    return df.groupby(['road_type','region','country','IncomeGroup'])['EQ_risk','FU_risk','PU_risk','Cyc_risk'].agg(sum_tuples)


def regional_cyclone(cycfil,data_path,events,param_values):
    """
    Arguments:
        
    
    Returns:
        
    """    
    #    try:
    all_tree_cov = [os.path.join(data_path,'tree_cover_road',x) for x in os.listdir(os.path.join(data_path,'tree_cover_road'))]
    
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

        df_cyc = df_cyc.progress_apply(lambda x: asset_cyclone(x,events,param_values),axis=1)
        return df_cyc.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples)
    
    else:
        return None

def regional_earthquake(file,data_path,global_costs,paved_ratios,events,wbreg_lookup):
    """
    Arguments:
        
    
    Returns:
        
    """    
    try:
        df = pandas.read_feather(file)
        val_cols = [x for x in list(df.columns) if 'val' in x]
        region = list(df.region.unique())[0]
        df = df.loc[~(df[val_cols] < 92).all(axis=1)]        
        
        if len(df) == 0:
            print('No shaked roads in {}'.format(region))
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
            df_liq = pandas.read_feather(os.path.join(data_path,'liquefaction_road','{}_liq.ft'.format(region)))
            df = df.merge(df_liq[['osm_id','liquefaction']],left_on='osm_id',right_on='osm_id')
            df = df.loc[~(df['liquefaction'] <= 1)]
            df = df.loc[~(df['liquefaction'] > 5)]
            if len(df) == 0:
                print('No liquid roads in {}'.format(region))
                return None            
    
        param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values_eq.pkl'))[x:x+4] 
        for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))), 4)]
    
        sorted_all = [[10,25,40],[20,40,60],[30,55,80],[40,70,100]]
        frag_tables = []
        for sort_rand in sorted_all:
            EQ_fragility = numpy.array([[0]+sort_rand+[100],[0,0]+sort_rand,[0,0,0]+sort_rand[:2],[0,0,0,0]+sort_rand[:1],[0,0,0,0,0]]).T
            df_EQ_frag = pandas.DataFrame(EQ_fragility,columns=[5,4,3,2,1],index=[(0,92),(93,180),(181,340),(341,650),(651,5000)])
            frag_dict = df_EQ_frag.stack(0).to_dict()
            frag_tables.append(frag_dict)
    
        try:
            all_bridge_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))]
            bridges = list(pandas.read_feather([x for x in all_bridge_files if os.path.split(file)[1][:-6] in x][0])['osm_id'])
            df = df.loc[~(df['osm_id'].isin(bridges))]
        except:
            None
        
        df_output = pandas.DataFrame(columns=events,index=df.index).fillna(0)
        df = pandas.concat([df,df_output],axis=1)
    
        tqdm.pandas(desc = region)
        df = df.progress_apply(lambda x: asset_earthquake(x,global_costs,paved_ratios,frag_tables,events,wbreg_lookup,param_values,val_cols),axis=1)
    
        df.reset_index(inplace=True,drop=True)
        
        df.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples).to_csv(os.path.join(data_path,'EQ_impacts','{}.csv'.format(region)))
        
        return df.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples)

    except Exception as e:
        print('Failed to finish {} because of {}!'.format(file,e))
            

def regional_flood(file,data_path,global_costs,paved_ratios,flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup):
    """
    Arguments:
        
    
    Returns:
        
    """ 
    try:
        df = pandas.read_feather(file)
        region = df.region.unique()[0]
        val_cols = [x for x in list(df.columns) if 'val' in x]
        df = df.loc[~(df[val_cols] == 0).all(axis=1)]
        
        if len(df) == 0:
            print('No flooded roads in {}'.format(region))
            return None
    
        param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))[x:x+4] for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))), 4)]
    
        try:
            all_bridge_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))]
            bridges = list(pandas.read_feather([x for x in all_bridge_files if os.path.split(file)[1][:-6] in x][0])['osm_id'])
            df = df.loc[~(df['osm_id'].isin(bridges))]
        except:
            None
        
        df_output = pandas.DataFrame(columns=events,index=df.index).fillna(0)
        df = pandas.concat([df,df_output],axis=1)
    
        tqdm.pandas(desc = region)
        df = df.progress_apply(lambda x: asset_flood(x,global_costs,paved_ratios,
                                                         flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,param_values,val_cols),axis=1)
    
        df.reset_index(inplace=True,drop=True)
        return df.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples)
    except:
        print('{} failed'.format(file))

def bridge_parallel():
    """
    Arguments:
        
    
    Returns:
        
    """ 
    data_path  = load_config()['paths']['data']

    all_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))][:10]
    
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
    
    with Pool(cpu_count()-1) as pool: 
        collect_risks = pool.starmap(regional_bridge,zip(all_files,data_p_list,param_list,income_list,eq_curve_list,design_list,depth_list,wind_list),chunksize=1) 
    
    
    pandas.concat(collect_risks).to_csv(os.path.join(data_path,'summarized','bridge_risk_road.csv'))

def cyclone_parallel():  
    """
    Arguments:
        
    
    Returns:
        
    """ 
    data_path = load_config()['paths']['data']
    
    all_cyc_road = [os.path.join(data_path,'output_Cyc_full',x) for x in os.listdir(os.path.join(data_path,'output_Cyc_full'))]
    
    events = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
    
    problem = {
      'num_vars': 4,
      'names': ['x1', 'x2', 'x3','x4'],
      'bounds': [[5000,50000],[1000,10000],[500,5000],[0,500]]}
    
    # Generate samples
    param_values = morris.sample(problem, 10, num_levels=4, grid_jump=2,local_optimization =True)
    

    data_paths = [data_path]*len(all_cyc_road)
    event_list = [events]*len(all_cyc_road)
    param_list = [param_values]*len(all_cyc_road)

    
    with Pool(cpu_count()-1) as pool: 
        collect_output = pool.starmap(regional_cyclone,zip(all_cyc_road,data_paths,event_list,param_list),chunksize=1) 

    pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','Cyc_road_losses_uncer.csv'))


def earthquake_parallel():  
    """
    Arguments:
        
    
    Returns:
        
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
    
    shuffle(all_files)    
    
    with Pool(cpu_count()-1) as pool: 
        pool.starmap(regional_earthquake,zip(all_files,data_paths,pav_cost_list,pav_rat_list,events_list,wbreg_list),chunksize=1) 

              
def flood_parallel(hazard,start_=0,end_=46233):  
    """
    Arguments:
        
    
    Returns:
        
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
    
    # Load samples
    if hazard == 'FU':
        all_files = [os.path.join('C','output_FU_full',x) for x in os.listdir(os.path.join('C','output_FU_full'))][start_:end_]
        events = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
    elif hazard == 'PU':
        all_files = [os.path.join(data_path,'output_PU_full',x) for x in os.listdir(os.path.join(data_path,'output_PU_full'))][start_:end_]
        events = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500', 'PU-1000']

    data_paths = [data_path]*len(all_files)
    pav_cost_list = [global_costs]*len(all_files)
    pav_rat_list = [paved_ratios]*len(all_files)
    cur_pav_list = [flood_curve_paved]*len(all_files)
    cur_unpav_list = [flood_curve_unpaved]*len(all_files)
    events_list = [events]*len(all_files)
    wbreg_list = [wbreg_lookup]*len(all_files)
    
    with Pool(cpu_count()-1) as pool: 
        collect_output = pool.starmap(regional_flood,zip(all_files,data_paths,pav_cost_list,pav_rat_list,
                                                          cur_pav_list,cur_unpav_list,events_list,wbreg_list),chunksize=1) 

    pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','{}_road_losses_uncer_{}_{}.csv'.format(hazard,start_,end_)))
    
    
