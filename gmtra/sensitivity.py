"""
Source code for Global Multihazard Transport Risk Analysis (GMTRA)

Functions to perform a sensitivity analysis.

Copyright (C) 2019 Elco Koks. All versions released under the GNU Affero General Public License v3.0 license.
"""

import os
import pandas
import numpy
from tqdm import tqdm

from gmtra.utils import sum_tuples,square_m2_cost_range,sensitivity_risk
from gmtra.damage import road_flood,rail_flood,road_cyclone,rail_cyclone,road_earthquake,rail_earthquake,road_bridge_earthquake,road_bridge_flood_cyclone,rail_bridge_earthquake,rail_bridge_flood_cyclone
    
def regional_bridge(file,data_path,param_values,income_lookup,eq_curve,design_tables,depth_threshs,wind_threshs,rail=False):
    """
    Function to estimate the summary statistics of all bridge damages in a region
    
    Arguments:
        *file* : path to the .feather file with all bridges of a region.
        
        *data_path* : file path to location of all data.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.       
        
        *income_lookup* : A dictionary that relates a country ID (ISO3 code) with its World Bank income goup.
        
        *eq_curve* : A pandas DataFrame with unique damage curves for earthquake damages.
        
        *design_table* : A NumPy array that represents the design standards for different bridge types, dependent on road type.
        
        *depth_thresh* : A list with failure depth thresholds. 
        
        *wind_threshs* :  A list with failure wind gustspeed thresholds. 
        
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    Returns:
        *DataFrame* : a pandas DataFrame with summary damage statistics for the loaded region.
        
    """    
   
     # open regiona bridge file
    df = pandas.read_feather(file)

    # remove all bridges that are shorter than 6 meters. We assume those are culverts.
    df = df.loc[~(df.length < 6)]

    # only get the road or railway bridges, depending on what we want.
    if not rail:
        df = df.loc[~(df['road_type'] == 'nodata')]
    else:
        df = df.loc[~(df['rail_type'] == 'nodata')]
        
    # get range of possible costs based on the length of the bridge
    df['cost'] = df.apply(lambda x: square_m2_cost_range(x),axis=1)
    df.drop([x for x in list(df.columns) if 'length_' in x],axis=1,inplace=True)

    # get earthquake values.
    vals_EQ = [x for x in list(df.columns) if 'val_EQ' in x]

    # assign World Bank income group
    df['IncomeGroup'] = income_lookup[list(df.country.unique())[0]]
    region = list(df.region.unique())[0]

    # something went wrong in the order of the hazard maps, correct that here. 
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

    # estimate earthquake damages
    try:    
        all_rps = [1/250,1/475,1/975,1/1500,1/2475]
        events = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475']
        tqdm.pandas('Earthquake '+region)
        if not rail:
            df['EQ_risk'] = df.progress_apply(lambda x: road_bridge_earthquake(x,eq_curve,param_values,vals_EQ,events,all_rps,sensitivity=True),axis=1)
        else:
            df['EQ_risk'] = df.progress_apply(lambda x: rail_bridge_earthquake(x,eq_curve,param_values,vals_EQ,events,all_rps,sensitivity=True),axis=1)
            
    except:
        df['EQ_risk'] = [[(0,0)]*50]*len(df)   
        
    if df['IncomeGroup'].unique()[0] == 'HIC':
        design_table = design_tables[0]
    elif df['IncomeGroup'].unique()[0] == 'UMC':
        design_table = design_tables[1]
    else:
        design_table = design_tables[2]

    # estimate river flood damages
    try:
        all_rps = [1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000]
        events = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
        tqdm.pandas(desc='Fluvial '+region)
        if not rail:
            df['FU_risk'] = df.progress_apply(lambda x: road_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps,sensitivity=True),axis=1)
        else:
            df['FU_risk'] = df.progress_apply(lambda x: rail_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps,sensitivity=True),axis=1)
            
    except:
        df['FU_risk'] = [[(0,0)]*50]*len(df)
 
    # estimate surface flood damages 
    try:
        events = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500','PU-1000']
        tqdm.pandas(desc='Pluvial '+region)
        if not rail:
            df['PU_risk'] = df.progress_apply(lambda x: road_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps,sensitivity=True),axis=1)
        else:
            df['PU_risk'] = df.progress_apply(lambda x: rail_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps,sensitivity=True),axis=1) 
    except:
        df['PU_risk'] = [[(0,0)]*50]*len(df)      
        
    # estimate coastal flood damages        
    try:
        all_rps = [1/10,1/20,1/50,1/100,1/200,1/500,1/1000]
        events = ['CF-10', 'CF-20', 'CF-50', 'CF-100', 'CF-200', 'CF-500', 'CF-1000']
        tqdm.pandas(desc='Coastal '+region)
        if not rail:
            df['CF_risk'] = df.progress_apply(lambda x: road_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps,sensitivity=True),axis=1)
        else:
            df['CF_risk'] = df.progress_apply(lambda x: rail_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps,sensitivity=True),axis=1) 
    except:
        df['CF_risk'] = [[(0,0)]*50]*len(df)
        
    # estimate cyclone damages        
    try:
        all_rps = [1/50,1/100,1/250,1/500,1/1000]
        events = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
        tqdm.pandas(desc='Cyclones '+region)
        if not rail:
            df['Cyc_risk'] = df.progress_apply(lambda x: road_bridge_flood_cyclone(x,design_table,wind_threshs,param_values,events,all_rps,sensitivity=True),axis=1)
        else:
            df['Cyc_risk'] = df.progress_apply(lambda x: rail_bridge_flood_cyclone(x,design_table,wind_threshs,param_values,events,all_rps,sensitivity=True),axis=1) 
    except:
        df['Cyc_risk'] = [[(0,0)]*50]*len(df)
            
     # return output     
    if not rail:
        return df.groupby(['road_type','region','country','IncomeGroup'])['EQ_risk','FU_risk','PU_risk','CF_risk','Cyc_risk'].agg(sum_tuples)
    else:
        return df.groupby(['rail_type','region','country','IncomeGroup'])['EQ_risk','FU_risk','PU_risk','CF_risk','Cyc_risk'].agg(sum_tuples)


def regional_cyclone(file,data_path,events,param_values,rail=False):
    """
    Function to estimate the summary statistics of all cyclone damages in a region to road assets

    Arguments:
        *file* : path to the .feather file with all bridges of a region.
        
        *data_path* : file path to location of all data.
        
        *events* : A list with the unique cyclone events.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.       
        
    Optional Arguments:
        *rail* : Default is **False**. Set to **True** if you would like to 
        intersect the railway assets in a region.
        
    Returns:
        *DataFrame* : a pandas DataFrame with summary damage statistics for the loaded region.
        
    """    
    try:
       # open tree cover for the region
        if not rail:
            all_tree_cov = [os.path.join(data_path,'tree_cover_road',x) for x in os.listdir(os.path.join(data_path,'tree_cover_road'))]
        else:
            all_tree_cov = [os.path.join(data_path,'tree_cover_rail',x) for x in os.listdir(os.path.join(data_path,'tree_cover_rail'))]
            
        # open cyclone intersection file for the region
        df_cyc = pandas.read_feather(file)
        val_cols = [x for x in list(df_cyc.columns) if 'val' in x]
        
        # drop all rows where cyclone wind speed is below 151km/h, which is assumed to be the critical treshold for tree snapping.
        df_cyc = df_cyc.loc[~(df_cyc[val_cols] < 151).all(axis=1)]
        
        # stop here if it means that actually nothing is being affected.
        if len(df_cyc) == 0:
            return None            
            
        region = list(df_cyc.region.unique())[0]
        
        # something went wrong in the order of the hazard maps, correct that here. 
        df_cyc = df_cyc.rename({'val_Cyc_rp100':'val_Cyc_rp500',
                                'val_Cyc_rp500':'val_Cyc_rp1000', 
                                'val_Cyc_rp1000':'val_Cyc_rp100',
                                'length_Cyc_rp100':'length_Cyc_rp500',
                                'length_Cyc_rp500':'length_Cyc_rp1000', 
                                'length_Cyc_rp1000':'length_Cyc_rp100'}, axis='columns')
        df_tree = pandas.read_feather([x for x in all_tree_cov if os.path.split(file)[1][:-7] in x][0])
        df_cyc = df_cyc.merge(df_tree[['osm_id','Tree_Dens']],left_on='osm_id',right_on='osm_id')
        df_cyc = df_cyc.loc[~(df_cyc['Tree_Dens'] <= 0)]
        
        # estimate tree density
        df_cyc['Tree_Dens'].loc[(df_cyc['Tree_Dens'] >= 10000)] = 10000
        
        # add failure probability
        df_cyc['fail_prob'] = df_cyc.Tree_Dens/10000
        df_output = pandas.DataFrame(columns=events,index=df_cyc.index).fillna(0)
        df_cyc = pandas.concat([df_cyc,df_output],axis=1)
        tqdm.pandas(desc=region)
        
        if not rail:
            df_cyc = df_cyc.progress_apply(lambda x: road_cyclone(x,events,param_values,sensitivity=True),axis=1)
        else:
            df_cyc = df_cyc.progress_apply(lambda x: rail_cyclone(x,events,param_values,sensitivity=True),axis=1)

        df_cyc['risk'] = df_cyc.apply(lambda x: sensitivity_risk(x,'Cyc',[1/50,1/100,1/250,1/500,1/1000],len(param_values)),axis=1)
        df_cyc = df_cyc.drop([x for x in list(df_cyc.columns) if (x in events) | ('val_' in x) | ('length_' in x)],axis=1)    
        df_cyc.reset_index(inplace=True,drop=True)

        if not rail:
            df_cyc.to_csv(os.path.join(data_path,'Cyc_sensitivity','{}.csv'.format(region)))
        else:
            df_cyc.to_csv(os.path.join(data_path,'Cyc_sensitivity_rail','{}.csv'.format(region)))


        # and return if desired.
        return df_cyc.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples)

    except Exception as e:
        print('Failed to finish {} because of {}!'.format(file,e))

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
        # open earthquake intersection file for the region
        df = pandas.read_feather(file)
        val_cols = [x for x in list(df.columns) if 'val' in x]
        region = list(df.region.unique())[0]
        
        # remove all rows where PGA is below 0.092. We assume nothing will happens with these values.
        df = df.loc[~(df[val_cols] < 92).all(axis=1)]        
        
        # stop here if it means that actually nothing is being affected.
        if len(df) == 0:
            print('No shaked assets in {}'.format(region))
            return None
        
        # if we already did the damage calculations, stop right here.
        if os.path.exists(os.path.join(data_path,'EQ_impacts','{}.csv'.format(region))):
            return None
    
        if len(df) > 0:
            # something went wrong in the order of the hazard maps, correct that here. 
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
            
            # get the liquefaction files for either road or railway assets for the region       
            if not rail:
                df_liq = pandas.read_feather(os.path.join(data_path,'liquefaction_road','{}_liq.ft'.format(region)))
            else:
                df_liq = pandas.read_feather(os.path.join(data_path,'liquefaction_rail','{}_liq.ft'.format(region)))
             
            # and merge liquefaction output with pga 
            df = df.merge(df_liq[['osm_id','liquefaction']],left_on='osm_id',right_on='osm_id')
            df = df.loc[~(df['liquefaction'] <= 1)]
            df = df.loc[~(df['liquefaction'] > 5)]
            if len(df) == 0:
                print('No liquid roads in {}'.format(region))
                return None            
  
        # set base values for fragility table.
        sorted_all = [[10,25,40],[20,40,60],[30,55,80],[40,70,100]]
        frag_tables = []

        # load parameter values. We read them from an external file, to make sure we use the same set for all regions.
        if not rail:
            param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values_eq.pkl'))[x:x+4] 
            for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))), 4)]
            
            # create fragility table
            for sort_rand in sorted_all:
                EQ_fragility = numpy.array([[0]+sort_rand+[100],[0,0]+sort_rand,[0,0,0]+sort_rand[:2],[0,0,0,0]+sort_rand[:1],[0,0,0,0,0]]).T
                df_EQ_frag = pandas.DataFrame(EQ_fragility,columns=[5,4,3,2,1],index=[(0,92),(93,180),(181,340),(341,650),(651,5000)])
                frag_dict = df_EQ_frag.stack(0).to_dict()
                frag_tables.append(frag_dict)
        else:
            param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values_eq_rail.pkl'))[x:x+3] 
                        for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values_eq_rail.pkl'))), 3)]

            # create fragility table
            for sort_rand in sorted_all:
                EQ_fragility = numpy.array([[10]+sort_rand+[100],[0,10]+sort_rand,[0,0,10]+sort_rand[:2],[0,0,0,0]+sort_rand[:1],[0,0,0,0,0]]).T
                df_EQ_frag = pandas.DataFrame(EQ_fragility,columns=[5,4,3,2,1],index=[(0,92),(93,180),(181,340),(341,650),(651,5000)])
                frag_dict = df_EQ_frag.stack(0).to_dict()
                frag_tables.append(frag_dict)
    
        try:
            # remove all bridge assets from the calculation. We calculate those damages separately.
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
    
        # And we finally made it to the damage calculation!
        tqdm.pandas(desc = 'damage_{}'.format(region))
        if not rail:
            df = df.progress_apply(lambda x: road_earthquake(x,global_costs,paved_ratios,frag_tables,events,wbreg_lookup,param_values,val_cols,sensitivity=True),axis=1)
        else:
            df = df.progress_apply(lambda x: rail_earthquake(x,frag_tables,events,param_values,val_cols,sensitivity=True),axis=1)

        tqdm.pandas(desc = 'risk_{}'.format(region))

        # and calculate the risk
        df['risk'] = df.apply(lambda x: sensitivity_risk(x,[1/250,1/475,1/975,1/1500,1/2475],len(param_values)),axis=1)
        
        # and remove values and lenghts from dataframe
        df = df.drop([x for x in list(df.columns) if (x in events) | ('val_' in x) | ('length_' in x)],axis=1)    
        df.reset_index(inplace=True,drop=True)
        
        if not rail:
            df.to_csv(os.path.join(data_path,'EQ_sensitivity','{}.csv'.format(region)))
        else:
            df.to_csv(os.path.join(data_path,'EQ_sensitivity_rail','{}.csv'.format(region)))
        
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(file,e))

def regional_flood(file,hazard,data_path,global_costs,paved_ratios,flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,rail=False):
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
        # open flood intersection file for the region
        df = pandas.read_feather(file)
        region = df.region.unique()[0]
        val_cols = [x for x in list(df.columns) if 'val' in x]
        
        # remove all rows where the flood intersection value is zero (i.e. no inundation)
        df = df.loc[~(df[val_cols] == 0).all(axis=1)]
        
        # if this means we have no rows left, stop right here.
        if len(df) == 0:
            print('No flooded roads in {}'.format(region))
            return None
    
        # load parameter values. We read them from an external file, to make sure we use the same set for all regions.
        try:
            if not rail:
                param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))[x:x+4] 
                    for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))), 4)]
                
                # remove all bridge assets from the calculation. We calculate those damages separately.
                all_bridge_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))]
                bridges = list(pandas.read_feather([x for x in all_bridge_files if os.path.split(file)[1][:-6] in x][0])['osm_id'])
                df = df.loc[~(df['osm_id'].isin(bridges))]
            else:
                param_values = [numpy.fromfile(os.path.join(data_path,'input_data','param_values_fl_rail.pkl'))[x:x+3] 
                    for x in range(0, len(numpy.fromfile(os.path.join(data_path,'input_data','param_values_fl_rail.pkl'))), 3)]
    
                # remove all bridge assets from the calculation. We calculate those damages separately.
                all_bridge_files = [os.path.join(data_path,'bridges_osm_rail',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_rail'))]
                bridges = list(pandas.read_feather([x for x in all_bridge_files if os.path.split(file)[1][:-6] in x][0])['osm_id'])
                df = df.loc[~(df['osm_id'].isin(bridges))]                
        except:
            None
        
        df_output = pandas.DataFrame(columns=events,index=df.index).fillna(0)
        df = pandas.concat([df,df_output],axis=1)
    
        # And we finally made it to the damage calculation!
        tqdm.pandas(desc = 'damage_{}'.format(region))
        if not rail:
            df = df.progress_apply(lambda x: road_flood(x,global_costs,paved_ratios,
                                                         flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,param_values,val_cols,sensitivity=True),axis=1)
        else:
            curve = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=[1,2,3,4,5,6,7,8],
                                     sheet_name='Flooding',index_col=[0],skiprows=1)
            curve.columns = curve.columns.droplevel(0)
            
            df = df.progress_apply(lambda x: rail_flood(x,
                                                    curve,events,param_values,val_cols,wbreg_lookup,sensitivity=True),axis=1)         
    
        tqdm.pandas(desc = 'risk_{}'.format(region))

        # and calculate the risk based on all results 
        if (hazard == 'PU') | (hazard == 'FU'):
            df['risk'] = df.progress_apply(lambda x: sensitivity_risk([1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000],x,events,len(param_values)),axis=1)
        else:
            df['risk'] = df.progress_apply(lambda x: sensitivity_risk([1/10,1/20,1/50,1/100,1/200,1/500,1/1000],x,len(param_values)),axis=1)
        
        # and remove values and lenghts from dataframe
        df = df.drop([x for x in list(df.columns) if (x in events) | ('val_' in x) | ('length_' in x)],axis=1)    
        df.reset_index(inplace=True,drop=True)
        
        # and save output
        if not rail:
            df.to_csv(os.path.join(data_path,'{}_sensitivity'.format(hazard),'{}.csv'.format(region)))
        else:
            df.to_csv(os.path.join(data_path,'{}_sensitivity_rail'.format(hazard),'{}.csv'.format(region)))
        
        
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(file,e))