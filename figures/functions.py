# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:41:17 2019

@author: cenv0574
"""

from scipy import integrate
import numpy

def sum_tuples(l):
    return tuple(sum(x) for x in zip(*l))

def calc_risk_total(x,hazard,RPS,events):
    collect_risks = []
    for y in range(7):
        collect_risks.append(integrate.simps([x[y] for x in x[events]][::-1], x=RPS[::-1]))
    return collect_risks

def set_prot_standard(x,prot_lookup,events):
    prot_stand = prot_lookup[x.region]
    no_floods= [z for z in events if prot_stand > int(z.split('-')[1])]
    for no_flood in no_floods:
        x[no_flood] = (0,0,0,0,0,0,0)
    return x

def pluvial_design(x,hazard):
    if x.GroupCode == 'HIC':
        if x.road_type in ['primary','secondary']:
            if hazard == 'PU':
                x[['PU-5','PU-10','PU-20','PU-50']] = [(0,0,0,0,0,0,0)]*4
            elif hazard == 'FU':
                x[['FU-5','FU-10','FU-20','FU-50']] = [(0,0,0,0,0,0,0)]*4
            elif hazard == 'CF':
                x[['CF-10','CF-20','CF-50']] = [(0,0,0,0,0,0,0)]*3               
        else:
            if hazard == 'PU':
                x[['PU-5','PU-10','PU-20']] = [(0,0,0,0,0,0,0)]*3
            elif hazard == 'FU':
                x[['FU-5','FU-10','FU-20']] = [(0,0,0,0,0,0,0)]*3
            elif hazard == 'CF':
                x[['CF-10','CF-20']] = [(0,0,0,0,0,0,0)]*2
    elif x.GroupCode == 'UMC':
        if x.road_type in ['primary','secondary']:
            if hazard == 'PU':
                x[['PU-5','PU-10','PU-20']] = [(0,0,0,0,0,0,0)]*3
            elif hazard == 'FU':
                x[['FU-5','FU-10','FU-20']] = [(0,0,0,0,0,0,0)]*3
            elif hazard == 'CF':
                x[['CF-10','CF-20']] = [(0,0,0,0,0,0,0)]*2
        else:
            if hazard == 'PU':
                x[['PU-5','PU-10']] = [(0,0,0,0,0,0,0)]*2
            elif hazard == 'FU':
                x[['FU-5','FU-10']] = [(0,0,0,0,0,0,0)]*2
            elif hazard == 'CF':
                x[['CF-10']] = [(0,0,0,0,0,0,0)]*1
    else:
        if x.road_type in ['primary','secondary']:
            if hazard == 'PU':
                x[['PU-5','PU-10']] = [(0,0,0,0,0,0,0)]*2
            elif hazard == 'FU':
                x[['FU-5','FU-10']] = [(0,0,0,0,0,0,0)]*2
            elif hazard == 'CF':
                x[['CF-10']] = [(0,0,0,0,0,0,0)]*1
       
    return x

def pluvial_design_1up(x,hazard):
    if x.GroupCode == 'HIC':
        if x.road_type in ['primary','secondary']:
            if hazard == 'PU':
                x[['PU-5','PU-10','PU-20','PU-50','PU-75','PU-100']] = [(0,0,0,0,0,0,0)]*6
            elif hazard == 'FU':
                x[['FU-5','FU-10','FU-20','FU-50','FU-75','FU-100']] = [(0,0,0,0,0,0,0)]*6
            elif hazard == 'CF':
                x[['CF-10','CF-20','CF-50','CF-100']] = [(0,0,0,0,0,0,0)]*4               
        else:
            if hazard == 'PU':
                x[['PU-5','PU-10','PU-20','PU-50']] = [(0,0,0,0,0,0,0)]*4
            elif hazard == 'FU':
                x[['FU-5','FU-10','FU-20','FU-50']] = [(0,0,0,0,0,0,0)]*4
            elif hazard == 'CF':
                x[['CF-10','CF-20','CF-50']] = [(0,0,0,0,0,0,0)]*3
    elif x.GroupCode == 'UMC':
        if x.road_type in ['primary','secondary']:
            if hazard == 'PU':
                x[['PU-5','PU-10','PU-20','PU-50']] = [(0,0,0,0,0,0,0)]*4
            elif hazard == 'FU':
                x[['FU-5','FU-10','FU-20','FU-50']] = [(0,0,0,0,0,0,0)]*4
            elif hazard == 'CF':
                x[['CF-10','CF-20','CF-50']] = [(0,0,0,0,0,0,0)]*3
        else:
            if hazard == 'PU':
                x[['PU-5','PU-10','PU-20']] = [(0,0,0,0,0,0,0)]*3
            elif hazard == 'FU':
                x[['FU-5','FU-10','FU-20']] = [(0,0,0,0,0,0,0)]*3
            elif hazard == 'CF':
                x[['CF-10','CF-20']] = [(0,0,0,0,0,0,0)]*2
    else:
        if x.road_type in ['primary','secondary']:
            if hazard == 'PU':
                x[['PU-5','PU-10','PU-20']] = [(0,0,0,0,0,0,0)]*3
            elif hazard == 'FU':
                x[['FU-5','FU-10','FU-20']] = [(0,0,0,0,0,0,0)]*3
            elif hazard == 'CF':
                x[['CF-10','CF-20']] = [(0,0,0,0,0,0,0)]*2
        else:
            if hazard == 'PU':
                x[['PU-5','PU-10']] = [(0,0,0,0,0,0,0)]*2
            elif hazard == 'FU':
                x[['FU-5','FU-10']] = [(0,0,0,0,0,0,0)]*2
            elif hazard == 'CF':
                x[['CF-10']] = [(0,0,0,0,0,0,0)]*1
        
    return x


def pluvial_design_rail(x,hazard):
    if (hazard == 'PU'):
        if  x.GroupCode == 'HIC':
            x[['PU-5','PU-10','PU-20','PU-50']] = [(0,0,0,0,0,0,0)]*4
        elif x.GroupCode == 'UMC':
            x[['PU-5','PU-10','PU-20','PU-50']] = [(0,0,0,0,0,0,0)]*4
        else:
            x[['PU-5','PU-10','PU-20']] =[(0,0,0,0,0,0,0)]*3

    if (hazard == 'FU'):
        if  x.GroupCode == 'HIC':
            x[['FU-5','FU-10','FU-20','FU-50']] = [(0,0,0,0,0,0,0)]*4
        elif x.GroupCode == 'UMC':
            x[['FU-5','FU-10','FU-20','FU-50']] = [(0,0,0,0,0,0,0)]*4
        else:
            x[['FU-5','FU-10','FU-20']] = [(0,0,0,0,0,0,0)]*3


    if (hazard == 'CF'):
        if  x.GroupCode == 'HIC':
            x[['CF-10','CF-20','CF-50',]] =[(0,0,0,0,0,0,0)]*3
        elif x.GroupCode == 'UMC':
            x[['CF-10','CF-20','CF-50']] = [(0,0,0,0,0,0,0)]*3
        else:
            x[['CF-10','CF-20']] = [(0,0,0,0,0,0,0)]*2

    return x

def get_value(x,ne_sindex,ne_countries,col):
    matches = ne_countries.loc[ne_sindex.intersection(x.centroid.bounds[:2])]
    
    for match in matches.iterrows():
        if match[1].geometry.intersects(x) == True:
            return match[1][col]

def gdp_lookup(x,GDP_lookup):
    try:
        return GDP_lookup[x]
    except:
        return 0
       
def get_mean(x,columns):
    for col in columns:
        x[col] = numpy.mean(x[col])
    return x

def wbregion(x,wbc_lookup):
    try:
        return wbc_lookup[x]
    except:
        return 'YHI'