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
    
    # loop over all parameter combinations that are predefined
    uncer_output = []
    for param in param_values:
        
        # get depth threshold for this parameter set
        depth_thresh = depth_threshs[int(param[4]-1)]
        
        # get the range of possible cost for this bridge asset and the parameter values
        cost = x.cost[0]+((x.cost[1]-x.cost[0])*param[3])
        
        # different assumptions are made for different road types (i.e. 4 vs 2 lanes and width)
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
    
    # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
    if not sensitivity:
        return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))),events)
    else:
        return sensitivity_risk(all_rps,[tuple(x) for x in numpy.array(uncer_output).T],events,len([tuple(x) for x in numpy.array(uncer_output).T]))


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
    
    # loop over all parameter combinations that are predefined
    uncer_output = []
    for param in param_values:

        # get depth threshold for this parameter set
        depth_thresh = depth_threshs[int(param[3]-1)]

        # get the range of possible cost for this bridge asset and the parameter values
        cost = x.cost[0]+((x.cost[1]-x.cost[0])*param[2])
        
        # get hazard values from all return periods we are considering
        rps = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] < design_table[0][0]]
        rps_not = ['val_'+z for z in events if 1/[int(z) for z in re.findall('\d+',z)][0] >= design_table[0][0]]

        # estimate the fragility value and multiple this with the assumed cost.
        frag = numpy.array([0]*len(rps_not)+list((x[rps] > depth_thresh[0])*1))
        uncer_output.append((cost*x.length*param[0]*param[1]*2+cost*x.length*param[0]*(1-param[1])*1)*frag)

    # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
    if not sensitivity:
        return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))),events)
    else:
        return sensitivity_risk(all_rps,[tuple(x) for x in numpy.array(uncer_output).T],events,len([tuple(x) for x in numpy.array(uncer_output).T]))


def road_bridge_earthquake(x,eq_curve,param_values,events,all_rps,sensitivity=False):
    """
    Function to estimate the range of earthquake damages to an individual bridge asset.

     Arguments:
        *x* : A row in a geopandas GeoDataFrame that represents an individual asset.
        
        *eq_curve* : A pandas DataFrame with unique damage curves for earthquake damages.
        
        *param_values* : A NumPy Array with sets of parameter values we would like to test.
        
        *events* : A list with the unique hazard events in row **x**.
        
        *all_rps* : A list with all return periods for the hazard that is being considered.
    
    Optional Arguments:
        *sensitivity* : Default is **False**. Set to **True** if you would like to 
        return all damage values to be able to perform a sensitivity analysis.
        
    Returns:
        *list* : A list with the range of possible damages to the specified bridge, based on the parameter set.
        
    """        
    # loop over all parameter combinations that are predefined
    uncer_output = []
    for param in param_values:

        # pick a curve from the set of earthquake curvs
        curve = eq_curve.iloc[:,int(param[4])-1]

        # get the fragility value and cost based on the curve
        frag = numpy.interp(list(x[[x for x in events ]]),list(curve.index), curve.values)
        cost = x.cost[0]+((x.cost[1]-x.cost[0])*param[3])
        
        # different assumptions are made for different road types (i.e. 4 vs 2 lanes and width)
        if x.road_type == 'primary':
            uncer_output.append((cost*x.length*param[0]*param[1]*4+cost*x.length*param[0]*(1-param[1])*2)*frag)
        else:
            uncer_output.append((cost*x.length*param[0]*param[2]*2+cost*x.length*param[0]*(1-param[2])*1)*frag)     
    
     # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
    if not sensitivity:
        return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))),events)
    else:
        return sensitivity_risk(all_rps,[tuple(x) for x in numpy.array(uncer_output).T],events,len([tuple(x) for x in numpy.array(uncer_output).T]))


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
    # loop over all parameter combinations that are predefined
    uncer_output = []
    for param in param_values:
        
        # pick a curve from the set of earthquake curvs
        curve = eq_curve.iloc[:,int(param[3])-1]
        
        # get the fragility value and cost based on the curve
        frag = numpy.interp(list(x[[x for x in vals_EQ ]]),list(curve.index), curve.values)
        cost = x.cost[0]+((x.cost[1]-x.cost[0])*param[2])

        uncer_output.append((cost*x.length*param[0]*param[1]*2+cost*x.length*param[0]*(1-param[1])*1)*frag)

    # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
    if not sensitivity:
        return monetary_risk(all_rps,list(zip(numpy.percentile(numpy.asarray(uncer_output), 0,axis=0),numpy.percentile(numpy.asarray(uncer_output), 20,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 40,axis=0),numpy.percentile(numpy.asarray(uncer_output), 50,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 60,axis=0),numpy.percentile(numpy.asarray(uncer_output), 80,axis=0),
                                                numpy.percentile(numpy.asarray(uncer_output), 100,axis=0))),events)
    else:
        return sensitivity_risk(all_rps,[tuple(x) for x in numpy.array(uncer_output).T],events,len([tuple(x) for x in numpy.array(uncer_output).T]))

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
    # loop over all parameter combinations that are predefined
    uncer_output = []
    for param in param_values:
        # specify clean-up cost for different road types, based on set of parameter values.
        cleanup_cost_dict = {'primary' : param[0],'secondary' : param[1],'tertiary' : param[2],'track' : param[3],'other' : param[3], 'nodata': param[3]}
        uncer_events = []
        for event in events:
            
            # the threshold value for breaking of trees is
            # hard-coded and based on literature. See Koks et al. (2019)
            if x['val_{}'.format(event)] > 150:
                uncer_events.append(cleanup_cost_dict[x.road_type]*x['length_{}'.format(event)]*x.fail_prob)
            else:
                uncer_events.append(0) 
        uncer_output.append(numpy.asarray(uncer_events))

    # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
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
    
    # loop over all parameter combinations that are predefined
    uncer_output = []
    for param in param_values:
        uncer_events = []
        for event in events:

            # the threshold value for breaking of trees is
            # hard-coded and based on literature. See Koks et al. (2019)
            if x['val_{}'.format(event)] > 150:
                uncer_events.append((param[0]*param[2]*x['length_{}'.format(event)]*x.fail_prob)+(param[1]*(1-param[2])*x['length_{}'.format(event)]*x.fail_prob))
            else:
                uncer_events.append(0) 
        uncer_output.append(numpy.asarray(uncer_events))

    # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
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
    # get worldbank region 
    wbreg = wbreg_lookup[x.country]

    # get paved versus unpaved ratios for this region
    ratios = paved_ratios.loc[(paved_ratios.ISO3 == x.country) & (paved_ratios.road_type == x.road_type)]
    
    # get road building cost for this region
    costs = global_costs[wbreg]
    
    # get list of unique columns that have the intersection length
    lengths = numpy.array(x[[x for x in list(x.index) if 'length_' in x]])

    loss_ratios = []
    uncer_output = []
    # loop over all parameter combinations that are predefined
    for param in param_values:
        loss_ratios = []
        for event in events:
            loss_ratios.append(frag_tables[int(param[2])][[z for z in frag_tables[int(param[2])] if (z[0][0] <= x['val_{}'.format(event)] <= z[0][1]) & (z[1] == x.liquefaction)][0]]/100)

        # get the fragility value and cost based on the curve
        loss_ratios = numpy.array(loss_ratios)
        # different assumptions are made for different road types (i.e. 4 vs 2 lanes and width)
        if x.road_type == 'primary':
            uncer_output.append(list((ratios.paved/100)*(param[0]*param[3]*costs['Paved 4L']+(1-param[0])*param[3]*costs['Paved 2L']))[0]*loss_ratios*lengths + 
            list((ratios.unpaved/100)*costs.Gravel)[0]*loss_ratios*lengths)
        elif x.road_type == 'secondary':
            uncer_output.append(list((ratios.paved/100)*(param[1]*param[3]*costs['Paved 2L']+(1-param[1])*param[3]*150000))[0]*loss_ratios*lengths + 
            list((ratios.unpaved/100)*costs.Gravel)[0]*loss_ratios*lengths)   
        else:
            uncer_output.append(list((ratios.paved/100)*(ratios.Paved_4L*param[3]*costs['Paved 2L']+ratios.Paved_2L*param[3]*150000))[0]*loss_ratios*lengths + 
    list((ratios.unpaved/100)*costs.Gravel)[0]*loss_ratios*lengths)    

    # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
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
    # specify cost for diesel (750k) and electric (1m)
    costs = (750000,1000000)
    
    # get length of all intersections for different hazards
    lengths = numpy.array(x[[x for x in list(x.index) if 'length_' in x]])

    # loop over all parameter combinations that are predefined
    uncer_output = []
    for param in param_values:
        loss_ratios = []
        for event in events:
            loss_ratios.append(frag_tables[int(param[2])][[z for z in frag_tables[int(param[2])] if (z[0][0] <= x['val_{}'.format(event)] <= z[0][1]) & (z[1] == x.liquefaction)][0]]/100)

        uncer_output.append(list((param[0]*(costs[0]*param[1])+(1-param[0])*(costs[1]*param[1]))*numpy.array(loss_ratios)*lengths))

    # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
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
    # get worldbank region 
    wbreg = wbreg_lookup[x.country]

    # get set of fragility curves for paved and unpaved roads
    curve_paved = flood_curve_paved.loc[:,wbreg]
    curve_unpaved = flood_curve_unpaved.loc[:,wbreg]
    
     # get paved versus unpaved ratios for this region
    ratios = paved_ratios.loc[(paved_ratios.ISO3 == x.country) & (paved_ratios.road_type == x.road_type)]
    
    # get road building cost for this region
    costs = global_costs[wbreg]
    
    # get list of unique columns that have the intersection length
    lengths = numpy.array(x[[x for x in list(x.index) if 'length_' in x]])

    # loop over all parameter combinations that are predefined
    uncer_output = []
    for param in param_values:
        # get the fragility value and based on the curve
        paved_frag = numpy.interp(list(x[[x for x in val_cols if 'val' in x]]),list(curve_paved.index),(curve_paved.values)*param[2])
        unpaved_frag = numpy.interp(list(x[[x for x in val_cols if 'val' in x]]),list(curve_unpaved.index), curve_unpaved.values)

        # different assumptions are made for different road types (i.e. 4 vs 2 lanes and width)
        if x.road_type == 'primary':
            uncer_output.append(list((ratios.paved/100)*(param[0]*param[3]*costs['Paved 4L']
            +(1-param[0])*param[3]*costs['Paved 2L']))[0]*paved_frag*lengths 
             + list((ratios.unpaved/100)*costs.Gravel)[0]*unpaved_frag*lengths)
        elif x.road_type == 'secondary':
            uncer_output.append(list((ratios.paved/100)*(param[1]*param[3]*costs['Paved 2L']
            +(1-param[1])*param[3]*150000))[0]*paved_frag*lengths 
            +list((ratios.unpaved/100)*costs.Gravel)[0]*unpaved_frag*lengths)   
        else:
            uncer_output.append(list((ratios.paved/100)*(ratios.Paved_4L*param[3]*costs['Paved 2L']
            +ratios.Paved_2L*param[3]*150000))[0]*paved_frag*lengths  
            + list((ratios.unpaved/100)*costs.Gravel)[0]*unpaved_frag*lengths)    

    # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
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
    # get worldbank region 
    wbreg = wbreg_lookup[x.country]

    # get set of fragility curves
    curve = curve.loc[:,wbreg]

    # specify rail building cost for this region
    costs = (750000,1000000)

    # get list of unique columns that have the intersection length
    lengths = numpy.array(x[[x for x in list(x.index) if 'length_' in x]])

    # loop over all parameter combinations that are predefined
    uncer_output = []
    for param in param_values:
        # get the fragility value and based on the curve
        frag = numpy.interp(list(x[[x for x in val_cols if 'val' in x]]),list(curve.index),(curve.values)*param[2])

        uncer_output.append(list((param[0]*(costs[0]*param[1])+(1-param[0])*(costs[1]*param[1]))*frag*lengths))

    # for the sensitivity analysis we return all outputs, normally, we return a set of percentiles in the results.
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
    
    # open regiona bridge file
    df = pandas.read_feather(file)

    # remove all bridges that are shorter than 6 meters. We assume those are culverts.
    df = df.loc[~(df.length < 6)]
    
    # only get the road or railway bridges, depending on what we want.
    if not rail:
        df = df.loc[~(df['road_type'] == 'nodata')]
    else:
        df = df.loc[(df['road_type'] == 'nodata')]
        
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

    # estimate river flood damages
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

    # estimate surface flood damages 
    try:
        events = ['PU-5', 'PU-10', 'PU-20', 'PU-50', 'PU-75', 'PU-100', 'PU-200', 'PU-250','PU-500','PU-1000']
        tqdm.pandas(desc='Pluvial '+region)
        if not rail:
            df['PU_risk'] = df.progress_apply(lambda x: road_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1)
        else:
            df['PU_risk'] = df.progress_apply(lambda x: rail_bridge_flood_cyclone(x,design_table,depth_threshs,param_values,events,all_rps),axis=1) 
    except:
        df['PU_risk'] = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*len(df)

    # estimate coastal flood damages        
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

    # estimate cyclone damages        
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

    # return output     
    if not rail:
        return df.groupby(['road_type','region','country','IncomeGroup'])['EQ_risk','FU_risk','PU_risk','CF_risk','Cyc_risk'].agg(sum_tuples)
    else:
        return df.groupby(['rail_type','region','country','IncomeGroup'])['EQ_risk','FU_risk','PU_risk','CF_risk','Cyc_risk'].agg(sum_tuples)


def regional_cyclone(file,data_path,events,param_values,rail=False):
    """
    Function to estimate the summary statistics of all cyclone damages in a region to road assets. 
    Cyclone damage for roads is currently based on clean-up cost and repairs.

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

        # get actual cyclone damage
        if not rail:
            df_cyc = df_cyc.progress_apply(lambda x: road_cyclone(x,events,param_values),axis=1)
        else:
            df_cyc = df_cyc.progress_apply(lambda x: rail_cyclone(x,events,param_values),axis=1)


        # And save the outputs to a .csv file.
        if not rail:
            df_cyc.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples).to_csv(os.path.join(data_path,'Cyc_impacts','{}.csv'.format(region)))
            return df_cyc.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples)

        else:
            df_cyc.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples).to_csv(os.path.join(data_path,'Cyc_impacts_rail','{}.csv'.format(region)))
            return df_cyc.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples)


        # and return if desired.

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
        tqdm.pandas(desc = region)
        if not rail:
            df = df.progress_apply(lambda x: road_earthquake(x,global_costs,paved_ratios,frag_tables,events,wbreg_lookup,param_values,val_cols),axis=1)
        else:
            df = df.progress_apply(lambda x: rail_earthquake(x,frag_tables,events,param_values,val_cols),axis=1)

        df.reset_index(inplace=True,drop=True)
        
        # And save the outputs to a .csv file.
        if not rail:
            df.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples).to_csv(os.path.join(data_path,'EQ_impacts','{}.csv'.format(region)))
            return df.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples)
        else:
            df.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples).to_csv(os.path.join(data_path,'EQ_impacts_rail','{}.csv'.format(region)))
            return df.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples)
                    
        # and return if desired.

    except Exception as e:
        print('Failed to finish {} because of {}!'.format(file,e))
            

def regional_flood(file,hzd,data_path,global_costs,paved_ratios,flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,rail=False):
    """
    Function to estimate the summary statistics of all flood damages in a region to road assets
    
    Arguments:
        *file* : path to the .feather file with all bridges of a region.
        
        *hzd* : abbrevation of the hazard we want to intersect.  **FU** for river flooding, 
        **PU** for surface flooding and **CF** for coastal flooding.

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
#    try:
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

    
    df_output = pandas.DataFrame(columns=events,index=df.index).fillna(0)
    df = pandas.concat([df,df_output],axis=1)

    # And we finally made it to the damage calculation!
    tqdm.pandas(desc = region)
    if not rail:
        df = df.progress_apply(lambda x: road_flood(x,global_costs,paved_ratios,
                                                     flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,param_values,val_cols),axis=1)
    else:
        # for railway we just load the curves seperately.
        curve = pandas.read_excel(os.path.join(data_path,'input_data','Costs_curves.xlsx'),usecols=[1,2,3,4,5,6,7,8],
                             sheet_name='Flooding',index_col=[0],skiprows=1)
        
        df = df.progress_apply(lambda x: rail_flood(x,
                                                curve,events,param_values,val_cols,wbreg_lookup),axis=1)
    df.reset_index(inplace=True,drop=True)
        
    # And save the outputs to a .csv file.
    if not rail:
        df.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples).to_csv(os.path.join(data_path,'{}_impacts','{}.csv'.format(region)))
        return df.groupby(['road_type','country','continent','region'])[events].agg(sum_tuples)
    else:
        df.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples).to_csv(os.path.join(data_path,'{}_impacts_rail','{}.csv'.format(region)))
        return df.groupby(['infra_type','country','continent','region'])[events].agg(sum_tuples)

#    except Exception as e:
#        print('Failed to finish {} because of {}!'.format(file,e))

