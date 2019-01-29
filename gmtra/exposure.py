# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:53:23 2018

@author: cenv0574
"""

import os
import sys
import pandas
import geopandas

pandas.options.mode.chained_assignment = None

from utils import load_config,total_length_risk,exposed_length_risk
data_path = load_config()['paths']['data']

from pathos.multiprocessing import Pool,cpu_count
pandas.set_option('chained_assignment',None)


def regional_roads(region,prot_lookup,data_path):
    """
    """
    try:
        print('{} started!'.format(region[3]))
        ID = region[3]
        wbincome = region[14]

        hazards = ['EQ','Cyc','PU','FU','CF']
        collect_risks = []
        reg_stats = pandas.read_csv(os.path.join(data_path,'road_stats','{}_stats.csv'.format(ID)))

        for hazard in hazards:
            df= pandas.read_feather(os.path.join(data_path,'output_{}_full'.format(hazard),'{}_{}.ft'.format(ID,hazard))) 
            if (hazard == 'FU') | (hazard == 'CF'):
                prot_stand = prot_lookup[ID]
                no_floods= [x for x in [x for x in df.columns if ('val' in x)] if prot_stand > int(x.split('-')[1])]
                df[no_floods] = 0

            if (hazard == 'PU'):
                if wbincome == 'HIC':
                    df.loc[df.road_type.isin(['primary','secondary']),['val_PU-5','val_PU-10','val_PU-20','val_PU-50']] = 0
                    df.loc[df.road_type.isin(['tertiary''track']),['val_PU-5','val_PU-10','val_PU-20']] = 0
                elif wbincome == 'UMC':
                    df.loc[df.road_type.isin(['primary','secondary']),['val_PU-5','val_PU-10','val_PU-20']] = 0
                    df.loc[df.road_type.isin(['tertiary''track']),['val_PU-5','val_PU-10']] = 0
                else:
                    df.loc[df.road_type.isin(['primary','secondary']),['val_PU-5','val_PU-10']] = 0
                    df.loc[df.road_type.isin(['tertiary''track']),['val_PU-5','val_PU-10']] = 0

            if (hazard == 'FU'):
                if wbincome == 'HIC':
                    df.loc[df.road_type.isin(['primary','secondary']),['val_FU-5','val_FU-10','val_FU-20','val_FU-50']] = 0
                    df.loc[df.road_type.isin(['tertiary''track']),['val_FU-5','val_FU-10','val_FU-20',]] = 0
                elif wbincome == 'UMC':
                    df.loc[df.road_type.isin(['primary','secondary']),['val_FU-5','val_FU-10','val_FU-20']] = 0
                    df.loc[df.road_type.isin(['tertiary''track']),['val_FU-5','val_FU-10']] = 0
                else:
                    df.loc[df.road_type.isin(['primary','secondary']),['val_FU-5','val_FU-10']] = 0
                    df.loc[df.road_type.isin(['tertiary''track']),['val_FU-5','val_FU-10']] = 0

            if (hazard == 'CF'):
                if wbincome == 'HIC':
                    df.loc[df.road_type.isin(['primary','secondary']),['val_CF-10','val_CF-20','val_CF-50']] = 0
                    df.loc[df.road_type.isin(['tertiary''track']),['val_CF-10','val_CF-20']] = 0
                elif wbincome == 'UMC':
                    df.loc[df.road_type.isin(['primary','secondary']),['val_CF-10','val_CF-20']] = 0
                    df.loc[df.road_type.isin(['tertiary''track']),['val_CF-10']] = 0
                else:
                    df.loc[df.road_type.isin(['primary','secondary']),['val_CF-10']] = 0
                    df.loc[df.road_type.isin(['tertiary''track']),['val_CF-10']] = 0
                    
            if hazard == 'EQ':
                reg_df = df.copy()
            elif hazard != 'EQ':
                reg_df = reg_df.merge(df[[x for x in df.columns if ('val_' in x) | ('length_' in x)]+['osm_id']],left_on='osm_id',right_on='osm_id')

            if hazard == 'EQ':
                event_list = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475'] #
                RPS = [1/250,1/475,1/975,1/1500,1/2475]
                cat_list = [1,2,3,4]
                bins = [-1,92,180,340,650,2000]
                df = df.rename({'val_EQ_rp250':'val_EQ_rp475',
                                'val_EQ_rp475':'val_EQ_rp1500', 
                                'val_EQ_rp975':'val_EQ_rp250',
                                'val_EQ_rp1500':'val_EQ_rp2475', 
                                'val_EQ_rp2475':'val_EQ_rp975',
                                'length_EQ_rp250':'length_EQ_rp475',
                                'length_EQ_rp475':'length_EQ_rp1500',
                                'length_EQ_rp975':'length_EQ_rp250',
                                'length_EQ_rp1500':'length_EQ_rp2475',
                                'length_EQ_rp2475':'length_EQ_rp975',}, axis='columns')
    
            elif hazard == 'Cyc':
                event_list = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
                RPS = [1/50,1/100,1/250,1/500,1/1000]
                cat_list = [1,2,3,4]
                bins = [-1,154,178,209,252,1000]
                df = df.rename({'val_Cyc_rp100':'val_Cyc_rp1000',
                                'val_Cyc_rp500':'val_Cyc_rp100', 
                                'val_Cyc_rp1000':'val_Cyc_rp500',
                                'length_Cyc_rp100':'length_Cyc_rp1000',
                                'length_Cyc_rp500':'length_Cyc_rp100', 
                                'length_Cyc_rp1000':'length_Cyc_rp500'}, axis='columns')
            elif hazard == 'FU':
                event_list = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
                RPS = [1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000]
                cat_list = [1,2,3,4]
                bins = [-1,25,50,100,200,2000]
            elif hazard == 'PU':
                event_list = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500', 'PU-1000']
                RPS = [1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000]
                cat_list = [1,2,3,4]
                bins = [-1,25,50,100,200,2000]
            elif hazard == 'CF':
                event_list = ['CF-10', 'CF-20', 'CF-50', 'CF-100', 'CF-200', 'CF-500', 'CF-1000']
                RPS = [1/10,1/20,1/50,1/100,1/200,1/500,1/1000]
                cat_list = [1,2,3,4]
                bins = [-1,25,50,100,200,2000]
                
            reg_stats[hazard] = reg_stats.apply(lambda x: total_length_risk(x,hazard,RPS),axis=1)        

            for event in event_list:
                reg_df['binned_{}'.format(event)] = pandas.cut(reg_df['val_{}'.format(event)], bins=bins, labels=[0]+cat_list)

            get_all_cats = []
            for cat in cat_list[:]:
                get_all_events = []
                for event in event_list:
                    event_sep = reg_df.loc[reg_df['binned_{}'.format(event)] == cat][['length_{}'.format(event),'country','region','continent','road_type']]
                    cont_out = pandas.DataFrame(event_sep.groupby(['continent','country','region','road_type'])['length_{}'.format(event)].sum())
                    get_all_events.append(cont_out)

                cat_df = pandas.concat(get_all_events,axis=1)
                cat_df = cat_df.fillna(0)

                if len(cat_df) == 0:
                    cat_df = pandas.DataFrame(columns = list(cat_df.columns)+['risk_{}_{}'.format(cat,hazard)],index=df.groupby(['continent','country','region','road_type']).sum().index).fillna(0)
                else:
                    cat_df['risk_{}_{}'.format(cat,hazard)] = cat_df.apply(lambda x: exposed_length_risk(x,hazard,RPS),axis=1)
                    cat_df.loc[cat_df['risk_{}_{}'.format(cat,hazard)] < 0] = 0
                    cat_df.reset_index(inplace=True)

                get_all_cats.append(cat_df.groupby(['continent','country','region','road_type']).sum()['risk_{}_{}'.format(cat,hazard)])

            collect_risks.append(pandas.concat(get_all_cats,axis=1).fillna(0))

        return (pandas.concat(collect_risks,axis=1).fillna(0)) 

    except Exception as e:
        print('Failed to finish {} because of {}!'.format(region[3],e))

def regional_railway(region,prot_lookup,data_path):
    """
    """
    try:
        print('{} started!'.format(region[3]))
        ID = region[3]
        wbincome = region[14]

        hazards = ['EQ','Cyc','PU','FU','CF']
        collect_risks = []
        reg_stats = pandas.read_csv(os.path.join(data_path,'railway_stats','{}_stats.csv'.format(ID)))

        for hazard in hazards:
            try:
                df= pandas.read_feather(os.path.join(data_path,'output_{}_rail_full'.format(hazard),'{}_{}.ft'.format(ID,hazard))) 
            except:
                continue
            if (hazard == 'FU'):
                prot_stand = prot_lookup[ID]
                no_floods= [x for x in [x for x in df.columns if ('val' in x)] if prot_stand > int(x.split('-')[1])]
                df[no_floods] = 0

            if (hazard == 'PU'):
                if wbincome == 'HIC':
                    df.loc[:,['val_PU-5','val_PU-10','val_PU-20','val_PU-50']] = 0
                elif wbincome == 'UMC':
                    df.loc[:,['val_PU-5','val_PU-10','val_PU-20']] = 0
                else:
                    df.loc[:,['val_PU-5','val_PU-10',]] = 0

            if (hazard == 'FU'):
                if wbincome == 'HIC':
                    df.loc[:,['val_FU-5','val_FU-10','val_FU-20','val_FU-50']] = 0
                elif wbincome == 'UMC':
                    df.loc[:,['val_FU-5','val_FU-10','val_FU-20',]] = 0
                else:
                    df.loc[:,['val_FU-5','val_FU-10']] = 0

            if (hazard == 'CF'):
                if wbincome == 'HIC':
                    df.loc[:,['val_CF-10','val_CF-20','val_CF-50',]] = 0
                elif wbincome == 'UMC':
                    df.loc[:,['val_CF-10','val_CF-20']] = 0
                else:
                    df.loc[:,['val_CF-10']] = 0
                    
            if hazard == 'EQ':
                reg_df = df.copy()
            elif hazard != 'EQ':
                reg_df = reg_df.merge(df[[x for x in df.columns if ('val_' in x) | ('length_' in x)]+['osm_id']],left_on='osm_id',right_on='osm_id')

            if hazard == 'EQ':
                event_list = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475'] #
                RPS = [1/250,1/475,1/975,1/1500,1/2475]
                cat_list = [1,2,3,4]
                bins = [-1,92,180,340,650,2000]
                df = df.rename({'val_EQ_rp250':'val_EQ_rp475',
                                'val_EQ_rp475':'val_EQ_rp1500', 
                                'val_EQ_rp975':'val_EQ_rp250',
                                'val_EQ_rp1500':'val_EQ_rp2475', 
                                'val_EQ_rp2475':'val_EQ_rp975',
                                'length_EQ_rp250':'length_EQ_rp475',
                                'length_EQ_rp475':'length_EQ_rp1500',
                                'length_EQ_rp975':'length_EQ_rp250',
                                'length_EQ_rp1500':'length_EQ_rp2475',
                                'length_EQ_rp2475':'length_EQ_rp975',}, axis='columns')
    
            elif hazard == 'Cyc':
                event_list = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
                RPS = [1/50,1/100,1/250,1/500,1/1000]
                cat_list = [1,2,3,4]
                bins = [-1,154,178,209,252,1000]
                df = df.rename({'val_Cyc_rp100':'val_Cyc_rp1000',
                                'val_Cyc_rp500':'val_Cyc_rp100', 
                                'val_Cyc_rp1000':'val_Cyc_rp500',
                                'length_Cyc_rp100':'length_Cyc_rp1000',
                                'length_Cyc_rp500':'length_Cyc_rp100', 
                                'length_Cyc_rp1000':'length_Cyc_rp500'}, axis='columns')
            elif hazard == 'FU':
                event_list = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
                RPS = [1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000]
                cat_list = [1,2,3,4]
                bins = [-1,25,50,100,200,2000]
            elif hazard == 'PU':
                event_list = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500', 'PU-1000']
                RPS = [1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000]
                cat_list = [1,2,3,4]
                bins = [-1,25,50,100,200,2000]
            elif hazard == 'CF':
                event_list = ['CF-10', 'CF-20', 'CF-50', 'CF-100', 'CF-200', 'CF-500', 'CF-1000']
                RPS = [1/10,1/20,1/50,1/100,1/200,1/500,1/1000]
                cat_list = [1,2,3,4]
                bins = [-1,25,50,100,200,2000]
  
                                
            reg_stats[hazard] = reg_stats.apply(lambda x: total_length_risk(x,hazard,RPS),axis=1)        

            for event in event_list:
                reg_df['binned_{}'.format(event)] = pandas.cut(reg_df['val_{}'.format(event)], bins=bins, labels=[0]+cat_list)

            get_all_cats = []
            for cat in cat_list[:]:
                get_all_events = []
                for event in event_list:
                    event_sep = reg_df.loc[reg_df['binned_{}'.format(event)] == cat][['length_{}'.format(event),'country','region','continent','infra_type']]
                    cont_out = pandas.DataFrame(event_sep.groupby(['continent','country','region','infra_type'])['length_{}'.format(event)].sum())
                    get_all_events.append(cont_out)

                cat_df = pandas.concat(get_all_events,axis=1)
                cat_df = cat_df.fillna(0)

                if len(cat_df) == 0:
                    cat_df = pandas.DataFrame(columns = list(cat_df.columns)+['risk_{}_{}'.format(cat,hazard)],index=df.groupby(['continent','country','region','infra_type']).sum().index).fillna(0)
                else:
                    cat_df['risk_{}_{}'.format(cat,hazard)] = cat_df.apply(lambda x: exposed_length_risk(x,hazard,RPS),axis=1)
                    cat_df.loc[cat_df['risk_{}_{}'.format(cat,hazard)] < 0] = 0
                    cat_df.reset_index(inplace=True)

                get_all_cats.append(cat_df.groupby(['continent','country','region','infra_type']).sum()['risk_{}_{}'.format(cat,hazard)])

            collect_risks.append(pandas.concat(get_all_cats,axis=1).fillna(0))

        return (pandas.concat(collect_risks,axis=1).fillna(0))

    except:
        print('{} failed'.format(region[3]))

def all_regions_parallel(road=True): 
    """
    """
    data_path = load_config()['paths']['data']
    global_regions = geopandas.read_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))
    incomegroups = pandas.read_csv(os.path.join(data_path,'input_data','incomegroups_2018.csv'),index_col=[0])
    income_dict = dict(zip(incomegroups.index,incomegroups.GroupCode))
    global_regions['wbincome'] = global_regions.GID_0.apply(lambda x : income_dict[x]) 
    
    global_regions = global_regions.loc[global_regions.GID_2.isin([(x.split('.')[0]) for x in os.listdir(os.path.join(data_path,'region_osm'))])]
    prot_lookup = dict(zip(global_regions['GID_2'],global_regions['prot_stand']))


    regions = list(global_regions.to_records())
    prot_lookups = [prot_lookup]*len(regions)
    data_paths = [data_path]*len(regions)    

    if road:
        with Pool(cpu_count()-1) as pool: 
            collect_output = pool.starmap(regional_roads,zip(regions,prot_lookups,data_paths),chunksize=1) 
    
        pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','total_exposure_road.csv'))
    
    else:
        with Pool(cpu_count()-1) as pool: 
            collect_output = pool.starmap(regional_roads,zip(regions,prot_lookups,data_paths),chunksize=1) 
    
        pandas.concat(collect_output).to_csv(os.path.join(data_path,'summarized','total_exposure_railway.csv'))
