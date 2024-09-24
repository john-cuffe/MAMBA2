###this file has the score generator functions (refactoring after a long time getting annoyed at how large the file had become)

from __future__ import division
import os
import sys
sys.path.append(os.getcwd())
from programs.logger_setup import *
logger=logger_setup(CONFIG['log_file_name'])
from programs.soundex import soundex
from programs.soundex import nysiis
from programs.model_generators import *

def create_fuzzy_scores(core_dict, data1_values, data2_values, target_function):
    '''function to create the fuzzy scores'''
    ###now create the output array
    vals = [(data1_values[x['data1_id']][target_function['variable_name']], data2_values[x['data2_id']][target_function['variable_name']]) for x in core_dict]
    scores=[]
    for val in vals:
        if val[0]!='NULL' and val[1]!='NULL':
                try:
                    scores.append(target_function['function'](val[0],val[1]))
                except:
                    scores.append(np.nan)
        else:
            scores.append(np.nan)
    return scores

###now the numeric_distance functions
def numeric_distance(x):
    '''
    Little function to come in
    :param x: pair of tuples
    :return: an array
    '''
    if x[0] and x[1]:
        return x[0] - x[1]
    else:
        return  np.nan

def exact_match(x):
    '''
    do the items match exactly?
    :param x: pair of tuples
    :return: an array
    '''
    if x[0] and x[1] and x[0]==x[1]:
        return 1
    else:
        return 0

def date_match(x):
    '''
    function for date matching
    :param x: tuple of dates
    :return:
    '''
    if x[0] and x[1]:
        return feb.editdist(x[0],x[1])
    else:
        return 0

####The Haversine distinace between two points
import math

def haversine(coord1, coord2):
    '''
    Lots of research went into this
    lol jk https://janakiev.com/blog/gps-points-distance-python/
    Adapted to give distance in kilometers
    :param coord1: tuple of coordinates (lat,lon)
    :param coord2: tuple of coordinates
    :return:
    '''
    R = 6372  # Earth radius in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def geo_match(x):
    '''
    How to run the geo_match function.  Is fed a pair of tuples
    [(data1_lat, data1_lon), (data2_lat, data2_lon)]
    :param x:
    :return:
    '''
    if x[0][0] is not None and x[0][1] is not None and x[1][0] is not None and x[1][1] is not None:
        return haversine(x[0],x[1])
    else:
        ###otherwise, negative times the entire radius of the earth.  Future iterations should impute from here using other info?
        return -12742

def get_soundex(x):
    '''function to compare soundex codes'''
    if x[0] and x[1]:
        return feb.editdist(soundex(x[0]),soundex(x[1]))
    else:
        return 0

def get_nysiis(x):
    '''same for nysiis'''
    if x[0] and x[1]:
        return feb.editdist(nysiis(x[0]),nysiis(x[1]))
    else:
        return 0

def create_custom_scores(core_dict, data1_values, data2_values, target_function):
    '''function to create the fuzzy scores'''
    ###now create the output array
    if type(target_function['custom_variable_kwargs'])!=dict:
        target_function['custom_variable_kwargs']={}
    ###provide the headers manually
    vals = [(data1_values[x['data1_id']][target_function['variable_name'].lower()], data2_values[x['data2_id']][target_function['variable_name'].lower()]) for x in core_dict]
    scores=[]
    for val in vals:
        try:
            scores.append(target_function['function'](val, **target_function['custom_variable_kwargs']))
        except Exception as error:
            logger.info('Warning: Exception in creating custom score.  Function {}, error {}, values: {}.  This is going to break stuff.'.format(target_function['variable_name'], error, str(val)))
            scores.append('FAIL')
    return scores

