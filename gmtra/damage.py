"""
Source code for Global Multihazard Transport Risk Analysis (GMTRA)

Functions to asses damages to all infrastructure assets.

Copyright (C) 2019 Elco Koks. All versions released under the GNU Affero General Public License v3.0 license.
"""

import os
import re
import pandas
import numpy

from tqdm import tqdm

from gmtra.utils import square_m2_cost_range,monetary_risk,sum_tuples,sensitivity_risk

pandas.set_option('chained_assignment',None)

def road_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps,sensitivity=False):
    """
    Function to estimate the range of either flood or cyclone damages to an individual bridge asset.
    
    Arguments:
        *x* : row in geopandas GeoDataFrame that represents an individual asset.
        
        *design_table* : A NumPy array that represents the design standards for different bridge types, dependent on road type.
        
        *depth_thresh* : A list with failure thresholds. Either contains flood depths or gust speeds.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.
        
        *events* : A list with the unique hazard events in row **x**.
        
        *all_rps* : A list with all return periods for the hazard that is being considered.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.
  
    Returns:
        *list* : A list with the range of possible damages to the specified bridge, based on the parameter set.
        
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
    
    if not sensitivity:
        return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))))
    else:
        return sensitivity_risk(all_rps,[tuple(x) for x in numpy.array(uncer_output).T])


def rail_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps,sensitivity=False):
    """
    Function to estimate the range of either flood or cyclone damages to an individual bridge asset.
    
    Arguments:
        *x* : row in geopandas GeoDataFrame that represents an individual asset.
        
        *design_table* : A NumPy array that represents the design standards for different bridge types, dependent on road type.
        
        *depth_thresh* : A list with failure thresholds. Either contains flood depths or gust speeds.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.
        
        *events* : A list with the unique hazard events in row **x**.
        
        *all_rps* : A list with all return periods for the hazard that is being considered.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.
  
    Returns:
        *list* : A list with the range of possible damages to the specified bridge, based on the parameter set.
        
    """ 
    
    uncer_output = []
    for param in param_values:
        depth_thresh = depth_threshs[int(param[3]-1)]
        cost = x.cost[0]+((x.cost[1]-x.cost[0])*param[2])
        rps = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] < design_table[0][0]]
        rps_not = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] >= design_table[0][0]]

        frag = numpy.array([0]*len(rps_not)+list((x[rps] > depth_thresh[0])*1))
        uncer_output.append((cost*x.length*param[0]*param[1]*2+cost*x.length*param[0]*(1-param[1])*1)*frag)

    if not sensitivity:
        return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))))
    else:
        return sensitivity_risk(all_rps,[tuple(x) for x in numpy.array(uncer_output).T])


def road_bridge_earthquake(x,eq_curve,param_values,vals_EQ,all_rps,sensitivity=False):
    """
    Function to estimate the range of earthquake damages to an individual bridge asset.

     Arguments:
        *x* : A row in a geopandas GeoDataFrame that represents an individual asset.
        
        *eq_curve* : A pandas DataFrame with unique damage curves for earthquake damages.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.
        
        *vals_EQ* : A list with the unique hazard events in row **x**.
        
        *all_rps* : A list with all return periods for the hazard that is being considered.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.
        
    Returns:
        *list* : A list with the range of possible damages to the specified bridge, based on the parameter set.
        
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
    
    if not sensitivity:
        return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))))
    else:
        return sensitivity_risk(all_rps,[tuple(x) for x in numpy.array(uncer_output).T])


def rail_bridge_earthquake(x,eq_curve,param_values,vals_EQ,events,all_rps,sensitivity=False):
    """
    Function to estimate the range of earthquake damages to an individual bridge asset.

     Arguments:
        *x* : A row in a geopandas GeoDataFrame that represents an individual asset.
        
        *eq_curve* : A pandas DataFrame with unique damage curves for earthquake damages.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.
        
        *vals_EQ* : A list with the unique hazard events in row **x**.
        
        *all_rps* : A list with all return periods for the hazard that is being considered.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.
        
    Returns:
        *list* : A list with the range of possible damages to the specified bridge, based on the parameter set.
        
    """        
    uncer_output = []
    for param in param_values:
        curve = eq_curve.iloc[:,int(param[3])-1]
        frag = numpy.interp(list(x[[x for x in vals_EQ ]]),list(curve.index), curve.values)
        cost = x.cost[0]+((x.cost[1]-x.cost[0])*param[2])

        uncer_output.append((cost*x.length*param[0]*param[1]*2+cost*x.length*param[0]*(1-param[1])*1)*frag)

    if not sensitivity:
        return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))))
    else:
        return sensitivity_risk(all_rps,[tuple(x) for x in numpy.array(uncer_output).T])

def road_cyclone(x,events,param_values,sensitivity=False):
    """
    Function to estimate the range of cyclone damages to an individual road asset.

    Arguments:
        *x* : row in geopandas GeoDataFrame that represents an individual asset.
        
        *events* : A list with the unique hazard events in row **x**.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.
    
    Returns:
        *list* : A list with the range of possible damages to the specified asset, based on the parameter set.
        
    """    
    uncer_output = []
    for param in param_values:
        cleanup_cost_dict = {'primary' : param[0],'secondary' : param[1],'tertiary' : param[2],'track' : param[3],'other' : param[3], 'nodata': param[3]}
        uncer_events = []
        for event in events:
            if x['val_{}'.format(event)] > 150:
                uncer_events.append(cleanup_cost_dict[x.infra_type]*x['length_{}'.format(event)]*x.fail_prob)
            else:
                uncer_events.append(0) 
        uncer_output.append(numpy.asarray(uncer_events))

    if not sensitivity:
        x[events] = list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 100,axis=0)))
    else:
        x[events] = [tuple(x) for x in numpy.array(uncer_output).T]
 
    return x

def rail_cyclone(x,events,param_values,sensitivity=False):
    """
    Function to estimate the range of cyclone damages to an individual railway asset.

    Arguments:
        *x* : row in geopandas GeoDataFrame that represents an individual asset.
        
        *events* : A list with the unique hazard events in row **x**.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.
    
    Returns:
        *list* : A list with the range of possible damages to the specified asset, based on the parameter set.
    
    """
    
    uncer_output = []
    for param in param_values:
        uncer_events = []
        for event in events:
            if x['val_{}'.format(event)] > 150:
                uncer_events.append((param[0]*param[2]*x['length_{}'.format(event)]*x.fail_prob)+(param[1]*(1-param[2])*x['length_{}'.format(event)]*x.fail_prob))
            else:
                uncer_events.append(0) 
        uncer_output.append(numpy.asarray(uncer_events))

    if not sensitivity:
        x[events] = list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 100,axis=0)))
    else:
        x[events] = [tuple(x) for x in numpy.array(uncer_output).T]
 
    return x


def road_earthquake(x,global_costs,paved_ratios,frag_tables,events,wbreg_lookup,param_values,sensitivity=False):
    """
    Function to estimate the range of earthquake damages to an individual road asset.

    Arguments:
        *x* : row in geopandas GeoDataFrame that represents an individual asset.
        
        *global_costs* : A pandas DataFrame with the total cost for different roads in different World Bank regions. These values 
        are based on the ROCKS database.
        
        *paved_ratios* : A pandas DataFrame with road pavement percentages per country for each road type.
        
        *frag_tables* : A NumPy Array with a set of unique fragility tables which relate PGA to liquefaction 
        to estimate the damage to the asset.
        *events* : A list with the unique earthquake events.
        
        *wbreg_lookup* : a dictioniary that relates countries (in ISO3 codes) with World Bank regions.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.   

    Returns:
        *list* : A list with the range of possible damages to the specified asset, based on the parameter set.        
    """       
    wbreg = wbreg_lookup[x.country]

    ratios = paved_ratios.loc[(paved_ratios.ISO3 == x.country) & (paved_ratios.road_type == x.infra_type)]
    costs = global_costs[wbreg]
    lengths = numpy.array(x[[x for x in list(x.index) if 'length_' in x]])

    loss_ratios = []
    uncer_output = []
    for param in param_values:
        loss_ratios = []
        for event in events:
            loss_ratios.append(frag_tables[int(param[2])][[z for z in frag_tables[int(param[2])] if (z[0][0] <= x['val_{}'.format(event)] <= z[0][1]) & (z[1] == x.liquefaction)][0]]/100)

        loss_ratios = numpy.array(loss_ratios)
        if x.infra_type == 'primary':
            uncer_output.append(list((ratios.paved/100)*(param[0]*param[3]*costs['Paved 4L']+(1-param[0])*param[3]*costs['Paved 2L']))[0]*loss_ratios*lengths + 
            list((ratios.unpaved/100)*costs.Gravel)[0]*loss_ratios*lengths)
        elif x.infra_type == 'secondary':
            uncer_output.append(list((ratios.paved/100)*(param[1]*param[3]*costs['Paved 2L']+(1-param[1])*param[3]*150000))[0]*loss_ratios*lengths + 
            list((ratios.unpaved/100)*costs.Gravel)[0]*loss_ratios*lengths)   
        else:
            uncer_output.append(list((ratios.paved/100)*(ratios.Paved_4L*param[3]*costs['Paved 2L']+ratios.Paved_2L*param[3]*150000))[0]*loss_ratios*lengths + 
    list((ratios.unpaved/100)*costs.Gravel)[0]*loss_ratios*lengths)    

    if not sensitivity:
        x[events] = list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 100,axis=0)))
    else:
        x[events] = [tuple(x) for x in numpy.array(uncer_output).T]
 
    return x

def rail_earthquake(x,frag_tables,events,param_values,sensitivity=False):
    """
    Function to estimate the range of earthquake damages to an individual railway asset.

    Arguments:
        *x* : row in geopandas GeoDataFrame that represents an individual asset.
        
        *frag_tables* : A NumPy Array with a set of unique fragility tables which relate PGA to liquefaction 
        to estimate the damage to the asset.
        
        *events* : A list with the unique earthquake events.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.

    Returns:
        *list* : A list with the range of possible damages to the specified asset, based on the parameter set.        
        
    """
    costs = (750000,1000000)
    lengths = numpy.array(x[[x for x in list(x.index) if 'length_' in x]])

    uncer_output = []
    for param in param_values:
        loss_ratios = []
        for event in events:
            loss_ratios.append(frag_tables[int(param[2])][[z for z in frag_tables[int(param[2])] if (z[0][0] <= x['val_{}'.format(event)] <= z[0][1]) & (z[1] == x.liquefaction)][0]]/100)

        uncer_output.append(list((param[0]*(costs[0]*param[1])+(1-param[0])*(costs[1]*param[1]))*numpy.array(loss_ratios)*lengths))

    if not sensitivity:
        x[events] = list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 100,axis=0)))
    else:
        x[events] = [tuple(x) for x in numpy.array(uncer_output).T]
    return x
    

def road_flood(x,global_costs,paved_ratios,flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,param_values,val_cols,sensitivity=False):
    """
    Function to estimate the range of flood damages to an individual road asset.

    Arguments:
        *x* : row in geopandas GeoDataFrame that represents an individual asset.
        
        *global_costs* : A pandas DataFrame with the total cost for different roads in different World Bank regions. These values 
        are based on the ROCKS database.
        
        *paved_ratios* : A pandas DataFrame with road pavement percentages per country for each road type.
        
        *flood_curve_paved* : A pandas DataFrame with a set of damage curves for paved roads.
        
        *flood_curve_unpaved* : A pandas DataFrame with a set of damage curves for unpaved roads.
        
        *events* : A list with the unique flood events.
        
        *wbreg_lookup* : a dictioniary that relates a country ID (ISO3 code) with its World Bank region.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.   
        
        *val_cols* : A list with the unique flood events in row **x**.
    
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.
    
    Returns:
        *list* : A list with the range of possible damages to the specified asset, based on the parameter set.
        
    """     
    wbreg = wbreg_lookup[x.country]

    curve_paved = flood_curve_paved.loc[:,wbreg]
    curve_unpaved = flood_curve_unpaved.loc[:,wbreg]
    ratios = paved_ratios.loc[(paved_ratios.ISO3 == x.country) & (paved_ratios.road_type == x.infra_type)]
    costs = global_costs[wbreg]
    lengths = numpy.array(x[[x for x in list(x.index) if 'length_' in x]])

    uncer_output = []
    for param in param_values:
        paved_frag = numpy.interp(list(x[[x for x in val_cols if 'val' in x]]),list(curve_paved.index),(curve_paved.values)*param[2])
        unpaved_frag = numpy.interp(list(x[[x for x in val_cols if 'val' in x]]),list(curve_unpaved.index), curve_unpaved.values)

        if x.infra_type == 'primary':
            uncer_output.append(list((ratios.paved/100)*(param[0]*param[3]*costs['Paved 4L']
            +(1-param[0])*param[3]*costs['Paved 2L']))[0]*paved_frag*lengths 
             + list((ratios.unpaved/100)*costs.Gravel)[0]*unpaved_frag*lengths)
        elif x.infra_type == 'secondary':
            uncer_output.append(list((ratios.paved/100)*(param[1]*param[3]*costs['Paved 2L']
            +(1-param[1])*param[3]*150000))[0]*paved_frag*lengths 
            +list((ratios.unpaved/100)*costs.Gravel)[0]*unpaved_frag*lengths)   
        else:
            uncer_output.append(list((ratios.paved/100)*(ratios.Paved_4L*param[3]*costs['Paved 2L']
            +ratios.Paved_2L*param[3]*150000))[0]*paved_frag*lengths  
            + list((ratios.unpaved/100)*costs.Gravel)[0]*unpaved_frag*lengths)    

    if not sensitivity:
        x[events] = list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 100,axis=0)))
    else:
        x[events] = [tuple(x) for x in numpy.array(uncer_output).T]
        
    return x

def rail_flood(x,curve,events,param_values,val_cols,wbreg_lookup,sensitivity=False):
    """
    Function to estimate the range of flood damages to an individual road asset.

    Arguments:
        *x* : row in geopandas GeoDataFrame that represents an individual asset.
        
        *curve* : A pandas DataFrame with a set of damage curves for railways.
        
        *events* : A list with the unique flood events.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.  
        
        *val_cols* : A list with the unique flood events in row **x**.
        
        *wbreg_lookup* : a dictioniary that relates a country ID (ISO3 code) with its World Bank region.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.

    Returns:
        *list* : A list with the range of possible damages to the specified asset, based on the parameter set.        

    """    
    wbreg = wbreg_lookup[x.country]

    curve = curve.loc[:,wbreg]
    costs = (750000,1000000)
    lengths = numpy.array(x[[x for x in list(x.index) if 'length_' in x]])

    uncer_output = []
    for param in param_values:
        frag = numpy.interp(list(x[[x for x in val_cols if 'val' in x]]),list(curve.index),(curve.values)*param[2])

        uncer_output.append(list((param[0]*(costs[0]*param[1])+(1-param[0])*(costs[1]*param[1]))*frag*lengths))

    if not sensitivity:
        x[events] = list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
        numpy.percentile(numpy.asarray(uncer_output), 100,axis=0)))
    else:
        x[events] = [tuple(x) for x in numpy.array(uncer_output).T]
        
    return x
  
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
    df = pandas.read_feather(file)

    df = df.loc[~(df.length < 6)]
    if not rail:
        df = df.loc[~(df['road_type'] == 'nodata')]
    else:
        df = df.loc[(df['road_type'] == 'nodata')]
        
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
    try:    
        all_rps = [1/250,1/475,1/975,1/1500,1/2475]
        events = ['EQ_rp250','EQ_rp475','EQ_rp975','EQ_rp1500','EQ_rp2475']
        tqdm.pandas('Earthquake '+region)
        if not rail:
            df['EQ_risk'] = df.progress_apply(lambda x: road_bridge_earthquake(x,eq_curve,param_values,vals_EQ,events,all_rps),axis=1)
        else:
            df['EQ_risk'] = df.progress_apply(lambda x: rail_bridge_earthquake(x,eq_curve,param_values,vals_EQ,events,all_rps),axis=1)
            
    except:
        df['EQ_risk'] =[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*len(df)
        
    if df['IncomeGroup'].unique()[0] == 'HIC':
        design_table = design_tables[0]
    elif df['IncomeGroup'].unique()[0] == 'UMC':
        design_table = design_tables[1]
    else:
        design_table = design_tables[2]

    try:
        all_rps = [1/5,1/10,1/20,1/50,1/75,1/100,1/200,1/250,1/500,1/1000]
        events = ['FU-5', 'FU-10', 'FU-20', 'FU-50', 'FU-75', 'FU-100', 'FU-200', 'FU-250','FU-500', 'FU-1000']
        tqdm.pandas(desc='Fluvial '+region)
        if not rail:
            df['FU_risk'] = df.progress_apply(lambda x: road_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1)
        else:
            df['FU_risk'] = df.progress_apply(lambda x: rail_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1)
            
    except:
        df['FU_risk'] = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*len(df)
 
    try:
        events = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500','PU-1000']
        tqdm.pandas(desc='Pluvial '+region)
        if not rail:
            df['PU_risk'] = df.progress_apply(lambda x: road_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1)
        else:
            df['PU_risk'] = df.progress_apply(lambda x: rail_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1) 
    except:
        df['PU_risk'] = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*len(df)
        
    try:
        all_rps = [1/10,1/20,1/50,1/100,1/200,1/500,1/1000]
        events = ['CF-10', 'CF-20', 'CF-50', 'CF-100', 'CF-200', 'CF-500', 'CF-1000']
        tqdm.pandas(desc='Coastal '+region)
        if not rail:
            df['CF_risk'] = df.progress_apply(lambda x: road_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1)
        else:
            df['CF_risk'] = df.progress_apply(lambda x: rail_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1) 
    except:
        df['CF_risk'] = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*len(df)
        
    try:
        all_rps = [1/50,1/100,1/250,1/500,1/1000]
        events = ['Cyc_rp50','Cyc_rp100','Cyc_rp250','Cyc_rp500','Cyc_rp1000']
        tqdm.pandas(desc='Cyclones '+region)
        if not rail:
            df['Cyc_risk'] = df.progress_apply(lambda x: road_bridge_flood_cyclone(x,design_table,wind_threshs,param_values,events,all_rps),axis=1)
        else:
            df['Cyc_risk'] = df.progress_apply(lambda x: rail_bridge_flood_cyclone(x,design_table,wind_threshs,param_values,events,all_rps),axis=1) 
    except:
        df['Cyc_risk'] = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*len(df)

    
    if not rail:
        return df.groupby(['road_type','region','country','IncomeGroup'])['EQ_risk','FU_risk','PU_risk','CF_risk','Cyc_risk'].agg(sum_tuples)
    else:
        return df.groupby(['rail_type','region','country','IncomeGroup'])['EQ_risk','FU_risk','PU_risk','CF_risk','Cyc_risk'].agg(sum_tuples)


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
            df_cyc = df_cyc.progress_apply(lambda x: road_cyclone(x,events,param_values),axis=1)
        else:
            df_cyc = df_cyc.progress_apply(lambda x: rail_cyclone(x,events,param_values),axis=1)
            
        return df_cyc.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples)
    
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
        
        df.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples).to_csv(os.path.join(data_path,'EQ_impacts','{}.csv'.format(region)))
        
        return df.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples)

    except Exception as e:
        print('Failed to finish {} because of {}!'.format(file,e))
            

def regional_flood(file,data_path,global_costs,paved_ratios,flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,rail=False):
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
        *DataFrame* : a pandas DataFrame with summary damage statistics for the loaded region.
    """ 
    try:
        df = pandas.read_feather(file)
        region = df.region.unique()[0]
        val_cols = [x for x in list(df.columns) if 'val' in x]
        df = df.loc[~(df[val_cols] == 0).all(axis=1)]
        
        if len(df) == 0:
            print('No flooded roads in {}'.format(region))
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
                                                         flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,param_values,val_cols),axis=1)
        else:
            curve = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=[1,2,3,4,5,6,7,8],
                                 sheet_name='Flooding',index_col=[0],skipfooter=9,header = [0,1])
            curve.columns = curve.columns.droplevel(0)
            
            df = df.progress_apply(lambda x: rail_flood(x,
                                                    curve,events,param_values,val_cols,wbreg_lookup),axis=1)
            
    
        df.reset_index(inplace=True,drop=True)
        return df.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples)
    except:
        print('{} failed'.format(file))

