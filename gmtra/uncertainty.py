# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:53:41 2018

@author: cenv0574
"""

def regional_flood(file,data_path,global_costs,paved_ratios,flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,hazard):
    try:
        df = pd.read_feather(file)
        region = df.region.unique()[0]
        val_cols = [x for x in list(df.columns) if 'val' in x]
        df = df.loc[~(df[val_cols] == 0).all(axis=1)]
        
        if len(df) == 0:
            print('No flooded roads in {}'.format(region))
            return None
        
        if len(os.listdir(os.path.join(data_path,'{}_sensitivity').format(hazard))) > 500:
            return None
        
        param_values = [np.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))[x:x+4] for x in range(0, len(np.fromfile(os.path.join(data_path,'input_data','param_values.pkl'))), 4)]
    
        try:
            all_bridge_files = [os.path.join(data_path,'bridges_osm_road',x) for x in os.listdir(os.path.join(data_path,'bridges_osm_road'))]
            bridges = list(pd.read_feather([x for x in all_bridge_files if os.path.split(file)[1][:-6] in x][0])['osm_id'])
            df = df.loc[~(df['osm_id'].isin(bridges))]
        except:
            None
        
        df_output = pd.DataFrame(columns=events,index=df.index).fillna(0)
        df = pd.concat([df,df_output],axis=1)
    
        tqdm.pandas(desc = region)
        df = df.progress_apply(lambda x: loss_estimations_flooding(x,global_costs,paved_ratios,
                                                         flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,param_values,val_cols),axis=1)
    
        if (hazard == 'PU') | (hazard == 'FU'):
            df['risk'] = df.apply(lambda x: calc_risk_total(x,hazard,[1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000],events),axis=1)
        else:
            df['risk'] = df.apply(lambda x: calc_risk_total(x,hazard,[1/10,1/20,1/50,1/100,1/200,1/500,1/1000],events),axis=1)
        
        df = df.drop([x for x in list(df.columns) if (x in events) | ('val_' in x) | ('length_' in x)],axis=1)    
        df.reset_index(inplace=True,drop=True)
        
        df.to_csv(os.path.join(data_path,'{}_sensitivity'.format(hazard),'{}.csv'.format(region)))
        
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(file,e))