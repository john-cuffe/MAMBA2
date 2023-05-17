###########
###CHUNK: THE RANDOM FOREST FUNCTIONS
############
from __future__ import division
import copy
import os
import sys
sys.path.append(os.getcwd())
from copy import deepcopy as dcpy
from programs.write_to_db import write_to_db
from programs.create_db_helpers import get_db_connection, get_table_noconn
from programs.general_helpers import load_model
import numpy as np
from scipy.stats import randint as sp_randint
from programs.logger_setup import *
from sklearn.model_selection import RandomizedSearchCV
from multiprocessing import Pool
from sklearn.feature_selection import RFECV
if 'custom' in [k['match_type'] for k in var_types]:
    import custom_scoring_methods as cust_scoring
from inspect import getmembers, isfunction
logger=logger_setup(CONFIG['log_file_name'])
from sklearn.ensemble import RandomForestClassifier
import random
from programs.soundex import soundex
from programs.soundex import nysiis
import traceback

##Run the Classifier
def runRFClassifier(y, X, nT, nE, mD):
    """
    Description: Executes random forest classifier
    Parameters:
        y - array of truth 1/0 match status
        X - array of features with same index as y
        nT - number of trees to use to estimate
        nE - number of features per tree
        mD - max depth of each tree
    Returns: model, the model object for the random forest classifier
    """
    tfunc = time.time()
    logger.info('#' * 10 + str(round((time.time() - starttime) / 60, 2)) + ' Minutes: Run Random Forest' + '#' * 10)
    # initialize estimator
    logger.info('Random Forest Params:\n Number of Trees={0}\n Max Features={2}\n Max Depth={1}'.format(nT, mD, nE))
    model = RandomForestClassifier(n_estimators=nT, max_features=nE, max_depth=mD, random_state=12345)
    # fit model
    model.fit(X, y)
    # print('Random Forest Time (Min): {0}'.format((time.time()-tfunc)/60))
    return model

####VIF score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X, headers):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif['name'] = headers
    return(vif)

###Function to grab the predicted probabilities that are retruened from predict_proba as a tuple
def probsfunc(x, model_type):
    '''
    This function generates a single arrary of predicted probabilities  of a match
    inputs:
    probs: an array of predicted probabilites form an Classifier
    '''
    if model_type!='custom':
        ith = np.zeros(shape=(len(x), 1))
        for i in range(len(x)):
            ith[i] = x[i][1]
    else:
        ####flag if
        if all([i[0].isnumeric() for i in x])==True:
            ith = np.zeros(shape=(len(x), 1))
            for i in range(len(x)):
                ith[i] = x[i][0]
        else:
            return np.array([i[0] for i in x])
    return ith


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

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def nominal_impute(values, header_names, header_types, nominal_info):
    '''
    This function will impute the fuzzy and numeric distance variables for the missing data.
    It gets vectorized so only need to focus on one row
    :param values: the values
    :param header_names: the names of the headers
    :param header_types: the types of match for each score.
    :param nominal_info: A dictionary with the format {variable:{'min':xxx,'max':yyy}} for each of the datatypes
    :return:
    '''
    for i in range(len(header_types)):
        if np.isnan(values[i]).any() == True:
            if header_types[i] == 'fuzzy':
                ###check if there are any nulls
                for k in range(len(values[i])):
                    if np.isnan(values[i][k]) == True:
                        values[i][k] = -1
                    else:
                        values[i][k] = np.round(values[i][k], 1)
            elif header_types[i] == 'numeric':
                for k in range(len(values[i])):
                    if np.isnan(values[i][k]) == True:
                        my_dims = nominal_info[header_names[i]]
                        ##sub with 10 times the largest possible difference for the variable
                        values[i][k] = 10 * (my_dims['max'] - my_dims['min'])
    return values

###generic function to get target data
def get_target_data(input_data, varlist, method='full', comp_type='normal'):
    ###If we are using truth/training data, use those table names.  Otherwise use the main tables
    if method=='truth':
        table_1_name='{}_training'.format(CONFIG['data1_table_name'])
        table_2_name='{}_training'.format(CONFIG['data2_table_name'])
    else:
        table_1_name=CONFIG['data1_table_name']
        table_2_name=CONFIG['data2_table_name']
    db=get_db_connection(CONFIG)
    ###get the datai
    data1_ids = ','.join("'{}'".format(v) for v in input_data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
    data2_ids = ','.join("'{}'".format(v) for v in input_data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
    ###create an indexed list of the id pairs to serve as the core of our dictionary
    my_input_data = dcpy(input_data)
    my_input_data.reset_index(inplace=True, drop=False)
    ####now make the standard name IDs for ease
    my_input_data.rename(columns={'{}_id'.format(CONFIG['data1_name']):'data1_id','{}_id'.format(CONFIG['data2_name']):'data2_id'}, inplace=True)
    core_dict = my_input_data[['index', 'data1_id','data2_id']].to_dict('records')
    ###now get the values from the database
    data1_names = ','.join(['{} as {}'.format(i[CONFIG['data1_name']], i['variable_name']) for i in varlist])
    data2_names = ','.join(['{} as {}'.format(i[CONFIG['data2_name']], i['variable_name']) for i in varlist])
    ###get the values
    if comp_type=='normal':
        data1_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names,
                                                                                      table_name=table_1_name,
                                                                                      id_list=data1_ids), db)
        data2_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names,
                                                                                      table_name=table_2_name,
                                                                                      id_list=data2_ids), db)
    else:
        data1_values = get_table_noconn(
            '''select id, latitude, longitude from {table_name} where id in ({id_list})'''.format(
                                                                                      table_name=table_1_name,
                                                                                      id_list=data1_ids), db)
        data1_values = [{'id': x['id'], 'coords':(x['latitude'],x['longitude'])} for x in data1_values]
        data2_values = get_table_noconn(
            '''select id, latitude, longitude from {table_name} where id in ({id_list})'''.format(names=data2_names,
                                                                                      table_name=table_2_name,
                                                                                      id_list=data2_ids), db)
        data2_values = [{'id': x['id'], 'coords':(x['latitude'],x['longitude'])} for x in data2_values]
    ###give the data values the name for each searching
    data1_values = {str(item['id']): item for item in data1_values}
    data2_values = {str(item['id']): item for item in data2_values}
    db.close()
    ###now create the output array
    return core_dict, data1_values, data2_values

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
    vals = [(data1_values[x['data1_id']][target_function['variable_name'].lower()], data2_values[x['data2_id']][target_function['variable_name'].lower()]) for x in core_dict]
    scores=[]
    for val in vals:
            try:
                scores.append(target_function['function'](val, **target_function['custom_variable_kwargs']))
            except Exception as error:
                logger.info('Warning: Exception in creating custom score.  Function {}, error {}, values: {}.  This is going to break stuff.'.format(target_function['variable_name'], error, str(val)))
                scores.append('FAIL')
    return scores

def create_scores(input_data, score_type, varlist, headers, method='full'):
    '''
    this function produces the list of dictionaries for all of the scores for the fuzzy variables.
    each dict will have two arguments: the array and a list of names
    :param input_data: Either a block to query the database with OR a dataframe of variable types
    :param score_type;: the type of score to calculate
    :param varlist: the variable list
    :param headers: the list of headers
    :param method: either 'training' or 'full' (the default).  if 'training', pull from the _training database tables
    NOTE FOR FUTURE SELF: IN NORMAL FUNCTIONS (MODEL_TRAINING=FALSE) THIS WILL ALREADY BE RUN IN A SUB-PROCESS
    SO YOU HAVE TO SINGLE-THREAD THE CALCULATIONS, SO THE WHOLE THING IS SINGLE-THREADED
    return: an array of values for each pair to match/variable and a list of headers
    '''
    ###get the target data
    db=get_db_connection(CONFIG)
    if score_type not in ['geo_distance']:
        core_dict, data1_values, data2_values = get_target_data(input_data, varlist, method, 'normal')
    if score_type=='fuzzy':
        ##Now assemble the list of variables we need to generate
        method_names = [i.__name__ for i in methods]
        to_run = []
        out_headers = []
        #####if we aren't using all variables
        if headers != 'all':
            for header in headers:
                if '_' in header:  ####only fuzzy variables have a _
                    my_name = header.rsplit('_', 1)
                    ###see if the name is in the method names
                    if my_name[1] in method_names:
                        to_run.append({'variable_name': my_name[0],
                                       'function': [meth for meth in methods if meth.__name__ == my_name[1]][0]})
                        out_headers.append('{}_{}'.format(my_name[0], my_name[1]))
        else:
            to_run = [{'variable_name': i['variable_name'].lower(), 'function': m} for i in varlist for m in methods]
            out_headers = ['{}_{}'.format(i['variable_name'], i['function'].__name__) for i in to_run]
        out_arr = np.vstack([create_fuzzy_scores(core_dict, data1_values, data2_values, x) for x in to_run])
        return {'output': out_arr, 'names': out_headers}
    elif score_type=='numeric_dist':
        out_arr = np.stack([np.apply_along_axis(numeric_distance, 1, [(data1_values[x['data1_id']][y['variable_name']],
                 data2_values[x['data2_id']][y['variable_name']]) for x in core_dict]) for y in varlist], axis=1)
        return {'output': out_arr, 'names': [i['variable_name'] for i in varlist]}
    elif score_type=='exact':
        out_arr = np.stack([np.apply_along_axis(exact_match, 1, [(data1_values[x['data1_id']][y['variable_name']],
                 data2_values[x['data2_id']][y['variable_name']]) for x in core_dict]) for y in varlist], axis=1)
        return {'output': out_arr, 'names': [i['variable_name'] for i in varlist]}
    elif score_type=='geo_distance':
        core_dict, data1_values, data2_values = get_target_data(input_data, varlist, method, 'geo')
        out_arr = np.array([[geo_match(a)] for a in [(data1_values[b['data1_id']]['coords'],data2_values[b['data2_id']]['coords']) for b in core_dict]])
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': ['geo_distance']}
    elif score_type=='date':
        out_arr = np.stack([np.apply_along_axis(date_match, 1, [(data1_values[x['data1_id']][y['variable_name']],
                                                                  data2_values[x['data2_id']][y['variable_name']]) for x
                                                                 in core_dict]) for y in varlist], axis=1)
        return {'output': out_arr, 'names': [i['variable_name'] for i in varlist]}
    elif score_type=='custom':
        ###If we are using truth/training data, use those table names.  Otherwise use the main tables
        if method == 'truth':
            table_1_name = '{}_training'.format(CONFIG['data1_table_name'])
            table_2_name = '{}_training'.format(CONFIG['data2_table_name'])
        else:
            table_1_name = CONFIG['data1_table_name']
            table_2_name = CONFIG['data2_table_name']
        ###If we are running a training data model
        data1_ids = ','.join("'{}'".format(v) for v in input_data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
        data2_ids = ','.join("'{}'".format(v) for v in input_data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
        ###now get the values from the database
        data1_names = ','.join(['''{} as {}'''.format(i[CONFIG['data1_name']], i['variable_name']) for i in varlist])
        data2_names = ','.join(['''{} as {}'''.format(i[CONFIG['data2_name']], i['variable_name']) for i in varlist])
        ###get the values
        data1_values = get_table_noconn('''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names,table_name=table_1_name,id_list=data1_ids), db)
        data2_values = get_table_noconn('''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names,table_name=table_2_name,id_list=data2_ids), db)
        ###give the data values the name for each searching
        data1_values = {str(item['id']): item for item in data1_values}
        data2_values = {str(item['id']): item for item in data2_values}
        ###create an indexed list of the id pairs to serve as the core of our dictionary
        my_input_data = dcpy(input_data)
        my_input_data.reset_index(inplace=True, drop=False)
        ####now make the standard name IDs for ease
        my_input_data.rename(columns={'{}_id'.format(CONFIG['data1_name']): 'data1_id',
                                      '{}_id'.format(CONFIG['data2_name']): 'data2_id'}, inplace=True)
        core_dict = my_input_data[['index', 'data1_id', 'data2_id']].to_dict('records')
        ##convert fuzzy vars to a list
        ###get the corresponding function attached to the var list
        for i in varlist:
            m = [m for m in getmembers(cust_scoring) if i['custom_variable_name'] == m[0]][0]
            if m:
                i['function']= m[1]
        out_arr = np.vstack([create_custom_scores(core_dict, data1_values, data2_values, x) for x in varlist])
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': [i['variable_name'] for i in varlist]}
    elif score_type=='phoenetic':
        ###give the data values the name for each soundex code
        for f in varlist:
            if 'soundex' in f['match_type']:
                f['function']=get_soundex
            if 'nysiis' in f['match_type']:
                f['function']=get_nysiis
        out_arr = np.stack(
            [np.apply_along_axis(y['function'], 1, [(data1_values[x['data1_id']][y['variable_name'].lower()],
                                                   data2_values[x['data2_id']][y['variable_name'].lower()]) for x
                                                  in core_dict]) for y in varlist], axis=1)
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': [i['variable_name'] for i in varlist]}

def create_all_scores(input_data, method,headers='all'):
    '''
    This function takes input data and creates all the scores needed
    :param input_data: the data we want to generate scores for
    :param method: either "prediction" or "truth".  If prediction, just return an array of X values.  If "Truth" return X, and X_hdrs
    :return: y, X_imputed, and X_hdrs
    '''
    ###get the varible types
    var_rec = copy.deepcopy(var_types)
    for v in var_rec:
        if v['match_type']=='fuzzy':
            v['possible_headers'] = ['{}_{}'.format(v['variable_name'], meth.__name__) for meth in methods]
    ###We have three types of variables.
    #1) First, identify all the variables we are going to need.  Fuzzy, exact, numeric_dist
    if headers=='all':
        fuzzy_vars=[i for i in var_rec if i['match_type']=='fuzzy']
        numeric_dist_vars=[i for i in var_rec if i['match_type']=='num_distance']
        exact_match_vars = [i for i in var_rec if i['match_type'] == 'exact']
        geo_distance = [i for i in var_rec if i['match_type'] == 'geom_distance']
        date_vars = [i for i in var_rec if i['match_type'] == 'date']
        custom_vars = [i for i in var_rec if i['custom_variable_name'] is not None]
        phoenetic_vars = [i for i in var_rec if i['match_type'].lower() in ['soundex','nysiis']]
    else:
        target_headers = [i.lower() for i in headers]
        ###If we are running with recursive feature elimination, just find the features we are going to use
        fuzzy_vars = [i for i in var_rec if i['match_type'] == 'fuzzy' and any(item.lower() in target_headers for item in i['possible_headers'])==True]
        numeric_dist_vars = [i for i in var_rec if i['match_type'] == 'num_distance' and i['variable_name'].lower() in target_headers]
        exact_match_vars = [i for i in var_rec if i['match_type'] == 'exact' and i['variable_name'].lower() in target_headers]
        geo_distance = [i for i in var_rec if i['match_type'] == 'geom_distance' and 'geo_distance' in target_headers]
        date_vars = [i for i in var_rec if i['match_type'] == 'date' and i['variable_name'].lower() in target_headers]
        custom_vars = [i for i in var_rec if i['custom_variable_name'] is not None and i['variable_name'].lower() in target_headers]
        phoenetic_vars = [i for i in var_rec if i['match_type'].lower() in ['soundex','nysiis'] and i['variable_name'].lower() in target_headers]
    ### fuzzy vars
    if len(fuzzy_vars) > 0:
        fuzzy_values=create_scores(input_data, 'fuzzy', fuzzy_vars, headers, method)
        X=fuzzy_values['output']
        X_hdrs=copy.deepcopy(fuzzy_values['names'])
    else:
        #####if we don't have fuzzy values, need this a bit later on
        fuzzy_values={'names':[]}
    ### num_distance: get the numeric distance between the two values, with a score of -9999 if missing
    if len(numeric_dist_vars) > 0:
        numeric_dist_values=create_scores(input_data, 'numeric_dist', numeric_dist_vars, headers, method)
        if 'X' in locals():
            X=np.vstack((X, numeric_dist_values['output'].T))
            X_hdrs.extend(numeric_dist_values['names'])
        else:
            X = numeric_dist_values['output']
            X_hdrs = copy.deepcopy(numeric_dist_values['names'])
    ### exact: if they match, 1, else 0
    if len(exact_match_vars) > 0:
        exact_match_values=create_scores(input_data, 'exact', exact_match_vars, headers, method)
        if 'X' in locals():
            X=np.vstack((X, exact_match_values['output'].T))
            X_hdrs.extend(exact_match_values['names'])
        else:
            X = exact_match_values['output']
            X_hdrs = copy.deepcopy(exact_match_values['names'])
    ###Geo Distance
    if len(geo_distance) > 0:
        geo_distance_values = create_scores(input_data, 'geo_distance', geo_distance, headers, method)
        if type(geo_distance_values['output']) != str:
            if 'X' in locals():
                X = np.vstack((X, geo_distance_values['output'].T))
                X_hdrs.extend(geo_distance_values['names'])
            else:
                X = geo_distance_values['output']
                X_hdrs = copy.deepcopy(geo_distance_values['names'])
    #### date vars
    if len(date_vars) > 0:
        date_values = create_scores(input_data,'date',date_vars, headers, method)
        if type(date_values['output']) != str:
            if 'X' in locals():
                X = np.vstack((X, date_values['output'].T))
                X_hdrs.extend(date_values['names'])
            else:
                X = date_values['output']
                X_hdrs = copy.deepcopy(date_values['names'])
    ###custom values
    if len(custom_vars) > 0:
        custom_values = create_scores(input_data,'custom',custom_vars, headers, method)
        if type(custom_values['output']) != str:
            if 'X' in locals():
                X = np.hstack((X, custom_values['output'].T))
                X_hdrs.extend(custom_values['names'])
            else:
                X = custom_values['output'].T
                X_hdrs = copy.deepcopy(custom_values['names'])
    ### phonetic values:
    if len(phoenetic_vars) > 0:
        phoenetic_values = create_scores(input_data, 'phoenetic', phoenetic_vars, headers, method)
        if type(phoenetic_values['output']) != str:
            if 'X' in locals():
                X = np.vstack((X, phoenetic_values['output'].T))
                X_hdrs.extend(phoenetic_values['names'])
            else:
                X = phoenetic_values['output']
                X_hdrs = copy.deepcopy(phoenetic_values['names'])
    ##Impute the missing data
    ##first transpose
    #X=X.T
    ####Options for missing data: Impute, or Interval:
    ##Imputation: Just impute the scores based on an iterative imputer
    if CONFIG['imputation_method'] == 'Imputer':
        imp = IterativeImputer(max_iter=10, random_state=0)
        ###fit the imputation, note the tranpsose
        imp.fit(X.T)
        X = imp.transform(X)
        ###making the dependent variable array
        if method == 'truth':
            y = input_data['match'].values
            return y, X, X_hdrs, imp
        else:
            return X, X_hdrs
    elif CONFIG['imputation_method'] == 'Nominal':
        ###for the fuzzy values, we will cut to each .1, and make any missing -1
        db = get_db_connection(CONFIG)
        nominal_boundaries={}
        if len(numeric_dist_vars) > 0:
            for var in numeric_dist_vars:
                ###get the values from the database
                my_values = get_table_noconn('''select min(min_1, min_2) min, max(max_1, max_2) max
                                                 from 
                                                 (select '0' id, min({data1_var}) min_1, max({data1_var}) max_1 from {table1_name})
                                                 left outer join 
                                                 (select '0' id, min({data2_var}) min_2, max({data2_var}) max_2 from {table2_name})'''.format(table1_name=CONFIG['data1_name'],
                                                                                                                                              table2_name=CONFIG['data2_name'],
                                                                                                                                              data1_var=var[CONFIG['data1_name']],
                                                                                                                                              data2_var=var[CONFIG['data2_name']]),db)
                ###save as a dictionary entry
                nominal_boundaries[var['variable_name']] = my_values[0]
        ###first, get the min and max values for the possible nominal variables
        header_types=[]
        for k in range(len(X_hdrs)):
            if X_hdrs[k] in fuzzy_values['names']:
                header_types.append('fuzzy')
            elif X_hdrs[k] in [j['variable_name'] for j in numeric_dist_vars]:
                header_types.append('numeric')
            else:
                header_types.append('other')
        ###Apply the nominal imputation function to each row
        X=nominal_impute(X, X_hdrs, header_types, nominal_boundaries).T
        ###making the dependent variable array
        if method == 'truth':
            y = input_data['match'].values
            return y, X, X_hdrs, 'Nominal'
        else:
            return X, X_hdrs
    else:
        ####if there is no imputation method
        if method == 'truth':
            y = input_data['match'].values
            return y, X, X_hdrs, 'None'
        else:
            return X, X_hdrs

def filter_data(data, arg):
    '''
    This function takes a customized filter and applies to to the input data, basically limited the number
    of observations we want to consider
    :param data: the input_data ids
    :return:
    '''
    ####first get the data
    db=get_db_connection(CONFIG)
    data1_ids = ','.join("'{}'".format(v) for v in data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
    data2_ids = ','.join("'{}'".format(v) for v in data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
    ###create an indexed list of the id pairs to serve as the core of our dictionary
    data.reset_index(inplace=True, drop=False)
    data = data[['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]]
    ###get the values
    if arg['block_info']['variable_filter_info']['match_type']!='geo_distance':
        data1_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as text), {var_name} as data1_target from {table_name} where id in ({id_list})'''.format(var_name=arg['block_info']['variable_filter_info'][CONFIG['data1_name']],
                                                                                      table_name=CONFIG['data1_table_name'],
                                                                                      id_list=data1_ids), db))
        data2_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as text), {var_name} as data2_target from {table_name} where id in ({id_list})'''.format(var_name=arg['block_info']['variable_filter_info'][CONFIG['data2_name']],
                                                                                      table_name=CONFIG['data2_table_name'],
                                                                                      id_list=data2_ids), db))
    else:
        data1_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as text), 
            latitude as data1_latitude,
             longitude as data1_longitude from {table_name} where id in ({id_list})'''.format(table_name=CONFIG['data1_table_name'],
                                                                                      id_list=data1_ids), db))
        data2_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as text), 
            latitude as data2_latitude,
             longitude as data2_longitude from {table_name} where id in ({id_list})'''.format(table_name=CONFIG['data2_table_name'],
                                                                                      id_list=data2_ids), db))
    ###merge the three dataframes
    data = data.merge(data1_values, left_on='{}_id'.format(CONFIG['data1_name']), right_on='id')
    data = data.merge(data2_values, left_on='{}_id'.format(CONFIG['data2_name']), right_on='id')
    ###identify the function we need to run:
    if arg['block_info']['variable_filter_info']['match_type']=='fuzzy':
       target_fun=[i for i in methods if i.__name__==arg['block_info']['variable_filter_info']['fuzzy_name']][0]
       data['score'] = data.apply(lambda x: target_fun(x['data1_target'], x['data2_target']) if x['data1_target']!='NULL' and x['data2_target']!='NULL'
                                                     else 0, axis=1)
    ###Now the exact matches##
    elif arg['block_info']['variable_filter_info']['match_type']=='exact':
        data['score'] = np.where(data['data1_target']==data['data2_target'], 1, 0)
    elif arg['block_info']['variable_filter_info']['match_type']=='num_distance':
        data['score'] = data['data1_target'] - data['data2_target']
    elif arg['block_info']['variable_filter_info']['match_type']=='geo_distance':
        data['score'] = data.apply(lambda x: haversine((x['data1_latitude'], x['data1_longitude']),(x['data2_latitude'],x['data2_longitude'])), axis=1)
    elif arg['block_info']['variable_filter_info']['match_type']=='date':
        data['score'] = data.apply(lambda x: feb.editdist(x['data1_target'], x['data2_target']), axis=1)
    elif arg['block_info']['variable_filter_info']['match_type']=='custom':
        ####get the right arrangement of custom functions and targets
        custom_scoring_functions = [{'name': i[0], 'function': i[1]} for i in getmembers(cust_scoring, isfunction)]
        ###get the corresponding function attached to the var list
        my_function = [k for k in custom_scoring_functions if k['name'].lower()==arg['block_info']['variable_filter_info']['variable_name'].lower()][0]
        data['score'] = data.apply(lambda x: my_function['function'](x['data1_target'], x['data2_target']) if x['data1_target']!='NULL' and x['data2_target']!='NULL'
                                                     else 0, axis=1)
    ####now return the columns needed with just the rows that meet the criteria
    qry = '''score {} {}'''.format( arg['block_info']['variable_filter_info']['test'],
                                  arg['block_info']['variable_filter_info']['filter_value'])
    data=data.query(qry)
    if len(data) > 0:
        return data[['index','{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]]
    else:
        return pd.DataFrame(columns=['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])])

def generate_logit(y, X, X_hdrs):
    '''
    Generates a linear logistic regression model
    :param y: the truth data column showing the real values.
    :param X: the indepdent variables for the model
    :param X_hdrs: the headers of X
    :return: Dictionay with type, score, model, and means if applicable.  Note we have a warning for high multicollinearity, AND ONLY USED
    WHEN USING ACCURACY AS THE SCORE (ONLY REAL COMPARABLE OPTION OUT OF THE BOX)
    '''
    logger.info('######CREATE LOGISTIC REGRESSION MODEL ######')
    from sklearn.linear_model import LogisticRegression
    ##Run the Grid Search
    if ast.literal_eval(CONFIG['debugmode'])==True:
        niter = 5
    else:
        niter = 200
    ##Mean-Center
    X=pd.DataFrame(X)
    ##Save the means
    X.columns=X_hdrs
    X_means=X.mean().to_dict()
    X=X-X.mean()
    if ast.literal_eval(CONFIG['feature_elimination_mode']) == False:
        myparams = {
            'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        mod=LogisticRegression(random_state=0, solver='liblinear')
        cv_rfc = RandomizedSearchCV(estimator=mod, param_distributions=myparams, cv=10, scoring=scoringcriteria,n_iter=niter)
        cv_rfc.fit(X, y)
        preds=cv_rfc.predict(X)
        score=cv_rfc.score(X,y)
        try:
            vif=calc_vif(X, X_hdrs)
            if max(vif['VIF']) > 5:
                logger.info('WARNING, SOME VARIABLES HAVE HIGH COLINEARITY (EVEN WITH MEAN-CENTERING).  RECONSIDER USING THE LOGIT. VARIABLES WITH ISSUES ARE:')
                for i in vif.to_dict('records'):
                    if i['VIF'] > 5:
                        logger.info('{}: VIF = {}'.format(i['name'], round(i['VIF'],2)))
            return {'type': 'Logistic Regression', 'score': score, 'model': mod, 'means': X_means,
                    'variable_headers': X_hdrs}

        except Exception as error:
            logger.warning("Warning.  Unable to calculate VIF, error {}.  Restart MAMBA with logit disabled. Proceeding by ignoring Logit".format(error))
            return 'fail'
    else:
        myparams = {
            'estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        mod = LogisticRegression(random_state=0, solver='liblinear').fit(X, y)
        selector = RFECV(mod, step=1, cv=5)
        cv_rfc = RandomizedSearchCV(estimator=selector, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        preds = cv_rfc.predict(X)
        score = cv_rfc.score(X, y)
        try:
            vif = calc_vif(X, X_hdrs)
            if max(vif['VIF']) > 5:
                logger.info(
                    'WARNING, SOME VARIABLES HAVE HIGH COLINEARITY (EVEN WITH MEAN-CENTERING).  RECONSIDER USING THE LOGIT. VARIABLES WITH ISSUES ARE:')
                for i in vif.to_dict('records'):
                    if i['VIF'] > 5:
                        logger.info('{}: VIF = {}'.format(i['name'], round(i['VIF'], 2)))
        except Exception as error:
            logger.warning("Warning.  Unable to calculate VIF, error {}.  Restart MAMBA with logit disabled. Proceeding by ignoring Logit".format(error))
            return 'fail'
        # Pick the X header values we need to use
        new_X_hdrs = []
        to_delete = []
        for hdr in range(len(X_hdrs)):
            my_ranking = cv_rfc.best_estimator_.ranking_[hdr]
            if my_ranking > 7:
                to_delete.append(hdr)
            else:
                new_X_hdrs.append(X_hdrs[hdr])
        ####Now delete the columns from X
        new_X = np.delete(X, to_delete, axis=1)
        mod.fit(new_X, y)
        score = np.round(mod.score(new_X, y), 5)
        ###limit X to just what we need
        logger.info('Random Forest Complete.  Score={}'.format(score))
        return {'type': 'Logistic Regression', 'score': score, 'model': mod, 'variable_headers': new_X_hdrs, 'means': X_means}

def generate_rf_mod(y, X, X_hdrs):
    '''
    Generate the random forest model we are going to use for the matching
    :param y: the truth data column showing the real values.
    :param X: the indepdent variables for the model
    :param X_hdrs: the headers of X    '''
    logger.info('######CREATE RANDOM FOREST MODEL ######')
    ###Generate the Grid Search to find the ideal values
    features_per_tree = ['sqrt', 'log2']
    rf = RandomForestClassifier(n_jobs=int(CONFIG['rf_jobs']), max_depth=10, max_features='sqrt', n_estimators=10)
    myparams = {
        'n_estimators': sp_randint(1, 25),
        'max_features': features_per_tree,
        'max_depth': sp_randint(5, 25)}
    ##Run the Grid Search
    if ast.literal_eval(CONFIG['debugmode'])==True:
        niter = 1
    else:
        niter = 50
    if ast.literal_eval(CONFIG['feature_elimination_mode']) == False:
        cv_rfc = RandomizedSearchCV(estimator=rf, param_distributions=myparams, cv=5, scoring=scoringcriteria,n_iter=niter)
        cv_rfc.fit(X, y)
        ##Save the parameters
        trees = cv_rfc.best_params_['n_estimators']
        features_per_tree = cv_rfc.best_params_['max_features']
        score = cv_rfc.best_score_
        max_depth = cv_rfc.best_params_['max_depth']
        ###No obvious need for MAMBALITE here
        ##In future could just remove the worst performaning variable using a relimp measure then run the full model
        #if mambalite == False:
        logger.info('Random Forest Completed.  Score {}'.format(score))
        rf_mod = runRFClassifier(y, X, trees, features_per_tree, max_depth)
        return {'type':'Random Forest', 'score':score, 'model':rf_mod, 'variable_headers':X_hdrs}
    else:
        ##First, get the scores
        myparams = {
            'estimator__n_estimators': sp_randint(1, 25),
            'estimator__max_features': features_per_tree,
            'estimator__max_depth': sp_randint(5, 25)}
        selector = RFECV(rf, step=1, cv=5)
        cv_rfc = RandomizedSearchCV(estimator=selector, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        # generate the model
        rf_mod = runRFClassifier(y, X, cv_rfc.best_estimator_.get_params()['estimator__n_estimators'], cv_rfc.best_estimator_.get_params()['estimator__max_features'], cv_rfc.best_estimator_.get_params()['estimator__max_depth'])
        # Pick the X header values we need to use
        new_X_hdrs = []
        to_delete = []
        for hdr in range(len(X_hdrs)):
            my_ranking = cv_rfc.best_estimator_.ranking_[hdr]
            if my_ranking > cv_rfc.best_estimator_.n_features_:
                to_delete.append(hdr)
            else:
                new_X_hdrs.append(X_hdrs[hdr])
        ####Now delete the columns from X
        new_X = np.delete(X, to_delete, axis=1)
        rf_mod.fit(new_X, y)
        score = np.round(rf_mod.score(new_X, y), 5)
        ###limit X to just what we need
        logger.info('Random Forest Complete.  Score={}'.format(score))
        return {'type': 'Random Forest', 'score': score, 'model': rf_mod, 'variable_headers': new_X_hdrs}

def generate_ada_boost(y, X, X_hdrs):
    '''
    This function generates the adaboosted model
    :param y: the truth data column showing the real values.
    :param X: the indepdent variables for the model
    :param X_hdrs: the headers of X
    '''
    logger.info('######CREATE ADABoost MODEL ######')
    from sklearn.ensemble import AdaBoostClassifier
    ###Generate the Grid Search to find the ideal values
    ##setup the SVM
    ada = AdaBoostClassifier(
                         algorithm="SAMME",
                         n_estimators=200)
    if ast.literal_eval(CONFIG['debugmode'])==True:
        niter = 5
    else:
        niter = int(np.round(len(X)/float(2),0))
    features_per_tree = ['sqrt', 'log2', 10, 15]
    myparams = {
        'n_estimators': sp_randint(1, 25),
        'algorithm':['SAMME','SAMME.R']}
    if ast.literal_eval(CONFIG['feature_elimination_mode'])==False:
        ##First, get the scores
        cv_rfc = RandomizedSearchCV(estimator=ada, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        ##Save the parameters
        trees = cv_rfc.best_params_['n_estimators']
        algo = cv_rfc.best_params_['algorithm']
        score = cv_rfc.best_score_
        ###No obvious need for MAMBALITE here
        ##In future could just remove the worst performaning variable using a relimp measure then run the full model
        # if mambalite == False:
        ada_mod = AdaBoostClassifier(algorithm=algo, n_estimators=trees)
        logger.info('AdaBoost Complete.  Score={}'.format(score))
        return {'type': 'AdaBoost', 'score': score, 'model': ada_mod, 'variable_headers':X_hdrs}
    else:
        ##First, get the scores
        myparams = {
            'estimator__n_estimators': sp_randint(1, 25),
            'estimator__algorithm': ['SAMME', 'SAMME.R']}
        selector = RFECV(ada, step=1, cv=5)
        cv_rfc = RandomizedSearchCV(estimator=selector, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        # generate the model
        ada_mod = AdaBoostClassifier(algorithm=cv_rfc.best_estimator_.get_params()['estimator__algorithm'], n_estimators=cv_rfc.best_estimator_.get_params()['estimator__n_estimators'])
        # Pick the X header values we need to use
        new_X_hdrs = []
        to_delete = []
        for hdr in range(len(X_hdrs)):
            my_ranking = cv_rfc.best_estimator_.ranking_[hdr]
            if my_ranking > cv_rfc.best_estimator_.n_features_:
                to_delete.append(hdr)
            else:
                new_X_hdrs.append(X_hdrs[hdr])
        ####Now delete the columns from X
        new_X = np.delete(X, to_delete, axis=1)
        ada_mod.fit(new_X,y)
        score = np.round(ada_mod.score(new_X,y), 5)
        ###limit X to just what we need
        logger.info('AdaBoost Complete.  Score={}'.format(score))
        return {'type': 'AdaBoost', 'score': score, 'model': ada_mod, 'variable_headers':new_X_hdrs}

def generate_svn_mod(y, X, X_Hdrs):
    '''
    Generate the SVM model we are going to use for the matching
    :param y: the truth data column showing the real values.
    :param X: the indepdent variables for the model
    :param X_hdrs: the headers of X    '''
    logger.info('\n\n######CREATE SVN MODEL ######\n\n')

    ###Generate the Grid Search to find the ideal values
    from sklearn import svm
    ##setup the SVM
    svc = svm.SVC(class_weight='balanced', gamma='scale')
    if ast.literal_eval(CONFIG['debugmode'])==True:
        niter = 5
    else:
        niter = int(np.round(len(X)/float(2),0))
    myparams={
        'kernel':['linear','poly','rbf']
    }
    if ast.literal_eval(CONFIG['feature_elimination_mode'])==False:
        ##First, get the scores
        svn_rfc = RandomizedSearchCV(estimator=svc, param_distributions=myparams, cv=5, scoring=scoringcriteria)
        svn_rfc.fit(X, y)
        ##Save the parameters
        kernel = svn_rfc.best_params_['kernel']
        score = svn_rfc.best_score_
        logger.info('SVM Complete.  Max Score={}'.format(score))
        svc=svm.SVC(gamma=gamma, degree=degree, kernel=kernel)
        ###Note for when you return--you need to change the predict function to do cross_val_predict
        return {'type':'SVM', 'score':score, 'model':svc, 'variable_headers':X_hdrs}
    else:
        ##First, get the scores
        myparams = {
            'estimator__kernel':['linear','poly','rbf']}
        selector = RFECV(svc, step=1, cv=5)
        cv_rfc = RandomizedSearchCV(estimator=selector, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        # generate the model
        svn_mod = svm.SVC(kernel=cv_rfc.best_estimator_.get_params()['estimator__kernel'])
        # Pick the X header values we need to use
        new_X_hdrs = []
        to_delete = []
        for hdr in range(len(X_hdrs)):
            my_ranking = cv_rfc.best_estimator_.ranking_[hdr]
            if my_ranking > cv_rfc.best_estimator_.n_features_:
                to_delete.append(hdr)
            else:
                new_X_hdrs.append(X_hdrs[hdr])
        ####Now delete the columns from X
        new_X = np.delete(X, to_delete, axis=1)
        svn_mod.fit(new_X,y)
        score = np.round(svn_mod.score(new_X,y), 5)
        ###limit X to just what we need
        logger.info('Support Vector Machine Complete.  Score={}'.format(score))
        return {'type': 'SVN', 'score': score, 'model': svn_mod, 'variable_headers':new_X_hdrs}

def choose_model(truthdat):
    '''
    This function generates a random forest, svn, and adaboost classifier for the training data, then returns the
    :param truthdat:
    :return:
    '''
    models = []
    y, X, X_hdrs, imputer = create_all_scores(truthdat, method='truth')
    ####if we are running the full MAMBA model suite:
    if ast.literal_eval(CONFIG['use_mamba_models'])==True:
        '''
        if ast.literal_eval(CONFIG['use_logit'])==True:
            logit=generate_logit(y, X, X_hdrs)
            if logit!='fail':
                logit = {'score': 0}
        else:
            logit={'score':0}
        models.append(logit)
        '''
        if ast.literal_eval(CONFIG['use_rf'])==True:
            rf=generate_rf_mod(y, X, X_hdrs)
        else:
            rf={'score':0}
        models.append(rf)
        if ast.literal_eval(CONFIG['use_svn'])==True:
            svn=generate_svn_mod(truthdat)
        else:
            svn={'score':0}
        models.append(svn)
        if ast.literal_eval(CONFIG['use_ada'])==True:
            ada=generate_ada_boost(y, X, X_hdrs)
        else:
            ada={'score':0}
        models.append(ada)
        ###Are we using the custom model?
        if ast.literal_eval(CONFIG['use_custom_model'])==True:
            import programs.custom_model as cust
            logger.info('\n\n######CREATE CUSTOM MODEL {}######\n\n'.format(cust.my_model_name))
            custom_model = cust.my_custom_model(y, X, X_hdrs)
            custom_model['type'] = 'custom'
            models.append(custom_model)
        ###ID the best score
        best_score=max([i['score'] for i in models])
        best_model=[i for i in models if i['score']==best_score][0]
        logger.info('Selected the {} Model, with the score of {}'.format(best_model['type'], best_model['score']))
        return best_model
    else:
        ###Otherwise, just run the custom model
        import custom_model as cust
        logger.info('\n\n######CREATE CUSTOM MODEL {}######\n\n'.format(cust.__name__))
        custom_model = cust.my_custom_model(y, X, X_hdrs)
        custom_model['type'] = 'custom'
        models.append(custom_model)
        best_score=max([i['score'] for i in models])
        best_model=[i for i in models if i['score']==best_score][0]
        logger.info('Custom model selected')
        return best_model

def intersection(lst1, lst2):
    '''
    Intersection method from https://www.geeksforgeeks.org/python-intersection-two-lists/
    :param lst1:
    :param lst2:
    :return:
    '''
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

####The actual match funciton
def match_fun(arg):
    '''
    This function runs the match.  
    arg is a dictionary.  
    target: the block we are targeting  
    '''
    ###Sometimes, the arg will have a logging flag if we are 10% of the way through
    try:
        start = time.time()
        ###setup the to_return
        to_return = {}
        if arg['model_name']!='False':
            logger.info('starting block {}, using model {}'.format(arg['target'], arg['model_name']))
        else:
            logger.info('starting block {}'.format(arg['target']))
        db=get_db_connection(CONFIG, timeout=1)
        ###The query we are going to use
        ####Note the select * here, means we can have a 'global' filter variable
        input_qry = """select cast(a.id as text) {data1_name}_id, cast(b.id as text) {data2_name}_id from
                (select *, {data1_block_name} as block{data1_selection_statement_variable} from {data1_name} where {data1_block_name} = '{target}' {selection_statement}) a
                inner join
                (select *, {data2_block_name} as block{data2_selection_statement_variable} from {data2_name} where {data2_block_name} = '{target}' {selection_statement}) b
                on a.block = b.block
                {filter_statement}"""
        ##if we aren't doing a chunked block
        if arg['block_info']['chunk_size']==-1:
            if ast.literal_eval(CONFIG['custom_selection_statement']) == False:
                input_qry_args = {'data1_name': CONFIG['data1_name'], 'data2_name': CONFIG['data2_name'],
                                  'data1_block_name': arg['block_info'][CONFIG['data1_name']],
                                  'data2_block_name': arg['block_info'][CONFIG['data2_name']],
                                  'target': arg['target'],
                                  'selection_statement': ''}
            else:
                input_qry_args = {'data1_name': CONFIG['data1_name'], 'data2_name': CONFIG['data2_name'],
                                  'data1_block_name': arg['block_info'][CONFIG['data1_name']],
                                  'data2_block_name': arg['block_info'][CONFIG['data2_name']],
                                  'target': arg['target'],
                                  'selection_statement': 'and ' + CONFIG['custom_selection_statement'].replace("'",'')}
        else:
            ###if we are chunking out the block, this identifies the block using the appropriate target element (remember, arg['target'] in this case is a tuple)
            if ast.literal_eval(CONFIG['custom_selection_statement']) == False:
                input_qry_args = {'data1_name': CONFIG['data1_name'], 'data2_name': CONFIG['data2_name'],
                                  'data1_block_name': arg['block_info'][CONFIG['data1_name']],
                                  'data2_block_name': arg['block_info'][CONFIG['data2_name']],
                                  'target': arg['target'][0],
                                  'selection_statement': ''}
            else:
                input_qry_args = {'data1_name': CONFIG['data1_name'], 'data2_name': CONFIG['data2_name'],
                                  'data1_block_name': arg['block_info'][CONFIG['data1_name']],
                                  'data2_block_name': arg['block_info'][CONFIG['data2_name']],
                                  'target': arg['target'][0],
                                  'selection_statement': 'and ' + CONFIG['custom_selection_statement'].replace("'",
                                                                                                               '')}        ###get the data
        ###Now fix the filter statement
        ###If we are in deduplication mode and we have a global filter statement:
        if CONFIG['mode']=='deduplication' and ast.literal_eval(CONFIG['global_filter_statement'])!=False:
            input_qry_args['filter_statement'] = 'where a.id < b.id and {}'.format(CONFIG['global_filter_statement'])
        ###if we are in deduplication mode and we do not have a global filter statement
        elif CONFIG['mode']=='deduplication' and ast.literal_eval(CONFIG['global_filter_statement'])==False:
            input_qry_args['filter_statement'] = 'where a.id < b.id'
        ###If we are ignoring duplicate IDs but aren't in deduplication mode and have a global filter
        elif ast.literal_eval(CONFIG['ignore_duplicate_ids'])==True and CONFIG['mode']!='deduplication' and ast.literal_eval(CONFIG['global_filter_statement'])!=False:
            ###So if we are ignoring duplicate IDs
            input_qry_args['filter_statement'] = 'where a.id!=b.id and {}'.format(CONFIG['global_filter_statement'])
        ###if we are ignorning duplciate IDS but aren't in deduplication mode but DO NOT have a global filter
        elif ast.literal_eval(CONFIG['ignore_duplicate_ids'])==True and CONFIG['mode']!='deduplication' and ast.literal_eval(CONFIG['global_filter_statement'])==False:
            ###So if we are ignoring duplicate IDs
            input_qry_args['filter_statement'] = 'where a.id!=b.id and {}'.format(CONFIG['global_filter_statement'])
        ###So in all other circumstances this is its own thing
        else:
            if ast.literal_eval(CONFIG['global_filter_statement'])!=False:
                input_qry_args['filter_statement'] = 'where {}'.format(CONFIG['global_filter_statement'])
            else:
                ###otherwise a blank statement
                input_qry_args['filter_statement'] = ''
        ###Ok.  Now add in the variable filter args if present
        if arg['block_info']['variable_filter_info']!=-1:
            ###this means we are using a variable filter
            ###three possible cases:
            # 1.  using a fuzzy or geometric filter, so we ignore
            if arg['block_info']['variable_filter_info']['match_type'] !='exact':
                input_qry_args['data1_selection_statement_variable'] = ''
                input_qry_args['data2_selection_statement_variable'] = ''
                ###we don't make any change to the filter statement
            # 2.  using json to define
            else:
                input_qry_args['data1_selection_statement_variable'] = ',{} as a_filter_target'.format(arg['block_info']['variable_filter_info'][CONFIG['data1_name']])
                input_qry_args['data2_selection_statement_variable'] = ',{} as b_filter_target'.format(arg['block_info']['variable_filter_info'][CONFIG['data2_name']])
                ###if there's already a statement, add to it
                if ast.literal_eval(CONFIG['ignore_duplicate_ids']) == True:
                    input_qry_args['filter_statement'] = input_qry_args['filter_statement'] + ' and a_filter_target {} b_filter_target'.format(arg['block_info']['variable_filter_info']['test']).replace("==","=")
                else:
                    input_qry_args['filter_statement'] = 'where a_filter_target {} b_filter_target'.format(arg['block_info']['variable_filter_info']['test']).replace("==","=")
        else:
            input_qry_args['data1_selection_statement_variable'] = ''
            input_qry_args['data2_selection_statement_variable'] = ''
        ###retrieve the data
        input_data = pd.DataFrame(get_table_noconn(input_qry.format(**input_qry_args), db))
    except Exception:
        logger.info('Error generating input data for block {}, error {}'.format(arg['target'], traceback.format_exc()))
        return 'fail'
        ####Nif the length, skip
    if len(input_data)==0:
        end = time.time() - start
        logger.info('There were no valid matches to attempt for block {}'.format(arg['target']))
        stats_dat={'batch_id':arg['batch_id'],
                                 'block_level':arg['block_info']['block_name'],
                                 'block_id': str(arg['target']),
                                 'block_time': end,
                                 'block_size': 0,
                                 'block_matches': 0,
                                 'block_matches_avg_score': 0,
                                 'block_non_matches': 0,
                                 'block_non_matches_avg_score': 0}
        stats_out = write_to_db(stats_dat, 'batch_statistics')
    else:
        try:
            orig_len = len(input_data)
            if arg['block_info']['variable_filter_info']!=-1 and arg['block_info']['variable_filter_info']['match_type'] in ['fuzzy','geo_distance']:
                logger.info('Filtering for block {}'.format(arg['target']))
                ###if we are filtering, return the input data that meets the criteria
                input_data = filter_data(input_data)
            if type(input_data)==str == True:
                if input_data=='fail':
                    return 'fail'
            elif len(input_data)==0 and type(input_data)!=str:
                end = time.time() - start
                logger.info('After filtering, there were no valid matches to attempt for block {}'.format(arg['target']))
                stats_dat={'batch_id':arg['batch_id'],
                                 'block_level':arg['block_info']['block_name'],
                                 'block_id': str(arg['target']),
                                 'block_time': end,
                                 'block_size': 0,
                                 'block_matches': 0,
                                 'block_matches_avg_score': 0,
                                 'block_non_matches': 0,
                                 'block_non_matches_avg_score': 0,
                                 'match_pairs_removed_filter':orig_len-len(input_data)}
                stats_out = write_to_db(stats_dat, 'batch_statistics')
                if stats_out:
                    logger.info('Unable to write batch statistics for batch {}, continuing'.format(arg['target']))
            else:
                logger.info('Creating Scores for block {}'.format(arg['target']))
                X, X_hdrs = create_all_scores(input_data, 'prediction', arg['X_hdrs'])
                ####Now write to the DB
                if len(input_data) > 0:
                    cur=db.cursor()
                    if ast.literal_eval(CONFIG['prediction']) == True:
                        ###Mean Center
                        if arg['model']['type'] == 'Logistic Regression':
                            X = pd.DataFrame(X, columns=X_hdrs) - arg['model']['means']
                            X = np.array(X)
                        logger.info('Predicting for block {}'.format(arg['target']))
                        if ast.literal_eval(CONFIG['use_custom_model'])==True:
                            input_data['predicted_probability'] = arg['model']['model'].predict_proba(X, X_hdrs)
                        else:
                            input_data['predicted_probability'] = probsfunc(arg['model']['model'].predict_proba(X), arg['model']['type'])
                        ####Now add in any matches
                        #####If the predicted probability is a string (so someone returned a JSON)
                        if input_data['predicted_probability'].dtype == 'O':
                            ####So we have an object that isn't meant to be filtered. Log it
                            ####Return predicted probability
                            input_data['match_info'] = input_data['predicted_probability']
                            input_data['predicted_probability'] = input_data.match_info.str['predicted_probability'].astype(float)
                        ##Now to the match threshold
                        stats_dat = copy.deepcopy(input_data)
                        matches = input_data[input_data['predicted_probability'] >= float(CONFIG['match_threshold'])]
                        if len(matches) > 0:
                            ###ranks
                            #####Note here that if you have an object, this will rank them alphabetically. Feel free to ignore them!
                            matches['{}_rank'.format(CONFIG['data1_name'])] = \
                            matches.groupby('{}_id'.format(CONFIG['data1_name']))['predicted_probability'].rank('dense')
                            matches['{}_rank'.format(CONFIG['data2_name'])] = \
                            matches.groupby('{}_id'.format(CONFIG['data2_name']))[
                                'predicted_probability'].rank('dense')
                            ###add the batch_id
                            matches['batch_id']=arg['batch_id']
                            ###convert the IDs into integers
                            for col in ['left_id','right_id','{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]:
                                if col in matches.columns:
                                    matches[col] = matches[col].astype(int)
                            ##convert back to dict
                            matches = matches.to_dict('records')
                            ###write to DB
                            write_out = write_to_db(matches, 'matched_pairs')
                            ###if the write to db has returned anything (i.e. it failed), then return the matches
                            if write_out:
                                to_return['matches'] = matches
                            if ast.literal_eval(CONFIG['chatty_logger']) == True:
                                logger.info('''{} matches added in block {}'''.format(len(matches), arg['target']))
                    ###otherwise, make sure stats dat has a zero
                    else:
                        stats_dat = copy.deepcopy(input_data)
                        stats_dat['predicted_probability'] = 0
                    ###try to write to db, if not return and then we will dump later
                    if ast.literal_eval(CONFIG['clerical_review_candidates']) == True:
                        ###If we are predicting, we will ONLY use the predicted probability from the model
                        if ast.literal_eval(CONFIG['prediction']) == False:
                            target_column = [k for k in range(len(X_hdrs)) if X_hdrs[k]==CONFIG['clerical_review_threshold']['variable'].lower()][0]
                            ####find where this is true
                            clerical_values = np.where(X[:,target_column] >= float(CONFIG['clerical_review_threshold']['value']), True, False)
                            input_data['threshold_value'] = X[:,target_column]
                            clerical_candidates = input_data[clerical_values==True]
                        else:
                            clerical_values = np.where(input_data['predicted_probability'] >= float(CONFIG['clerical_review_threshold']['value']), True, False)
                            clerical_candidates = input_data[clerical_values==True]
                        ###add in the batch id to each row
                        clerical_candidates['batch_id']=arg['batch_id']
                        ###if there are more than 10, select 10% of them
                        if len(clerical_candidates) >= 10:
                            clerical_review_dict = dcpy(random.sample(clerical_candidates.to_dict('records'),int(np.round(.1*len(clerical_candidates),0))))
                        else:
                            clerical_review_dict = dcpy(clerical_candidates.to_dict('records'))
                        if len(clerical_review_dict) > 0:
                            clerical_out = write_to_db(clerical_review_dict, 'clerical_review_candidates')
                            if clerical_out:
                                to_return['clerical_review_candidates']=clerical_out
                        ###if it's a chatty logger, log.
                        if ast.literal_eval(CONFIG['chatty_logger']) == True:
                            logger.info('''{} clerical review candidates added from block {}'''.format(len(clerical_review_dict), arg['target']))
                    cur.close()
                    db.close()
                    ###do the block statistics
                    end = time.time() - start
                    stats_dat = {'batch_id': arg['batch_id'],
                                               'block_level': str(arg['block_info']['block_name']),
                                               'block_id': str(arg['target']),
                                               'block_time': end,
                                               'block_size': len(stats_dat),
                                               'block_matches': int(np.sum(np.where(stats_dat['predicted_probability'] >= float(CONFIG['match_threshold']),1,0))),
                                               'block_matches_avg_score': np.nanmean(np.where(stats_dat['predicted_probability'] >= float(CONFIG['match_threshold']),stats_dat['predicted_probability'],np.nan)),
                                               'block_non_matches': int(np.sum(np.where(stats_dat['predicted_probability'] < float(CONFIG['match_threshold']),1,0))),
                                               'block_non_matches_avg_score': np.nanmean(np.where(stats_dat['predicted_probability'] < float(CONFIG['match_threshold']),stats_dat['predicted_probability'],np.nan)),
                                               'match_pairs_removed_filter':orig_len-len(input_data)}
                logger.info('main match complete for block {}'.format(arg['target']))
        except Exception:
            logger.info('Error generating scores for block {}, error {}'.format(arg['target'], traceback.format_exc()))
            return 'fail'
        try:
            stats_out = write_to_db(stats_dat, 'batch_statistics')
            if stats_out:
                logger.info('Unable to write batch statistics for batch {}, continuing'.format(arg['target']))
            if arg['logging_flag']!=-1:
                logger.info('{}% complete with block'.format(arg['logging_flag']))
            if 'clerical_review_candidates' in to_return.keys() or 'matches' in to_return.keys():
                return to_return
            else:
                return None
        except Exception:
            logger.info('Error for return stats dat for block {}, error {}'.format(arg['target'], traceback.format_exc()))
            return 'fail'

####The block function
def run_block(block, model, batch_id):
    '''
    This function creates the list of blocks to use for comparison and then runs the matches
    :param block: the dict of the block we are running
    :param model: the model we are using
    :param batch_id: for whatever reason, the CONFIG['batch_id'] call is inconsistent, so putting that directly here
    :return:
    '''
    ####If we are running the 'full' block
    db = get_db_connection(CONFIG)
    data1_blocks = get_table_noconn('''select distinct {} as block from {}'''.format(block[CONFIG['data1_name']], CONFIG['data1_table_name']), db)
    data1_blocks = [i['block'] for i in data1_blocks]
    data2_blocks = get_table_noconn('''select distinct {} as block from {}'''.format(block[CONFIG['data2_name']], CONFIG['data2_table_name']), db)
    data2_blocks = [i['block'] for i in data2_blocks]
    if block['chunk_size']==-1:
        ###So if we ARE NOT doing a chunked out block, the block list is the intersection (ie all the blocks that match)
        block_list=intersection(data1_blocks, data2_blocks)
    else:
        ###otherwise it's a list of tuples of all of the possible combinations of the blocks
        block_list=[(x, y) for x in data1_blocks for y in data2_blocks]
    if CONFIG.get('debug_block', None) is not None:
        block_list = [CONFIG['debug_block']]
    logger.info('We have {} blocks to run for {}'.format(len(block_list), block['block_name']))
    ###load any mapped models
    mapped_models = get_table_noconn('''select * from block_model_mapping where batch_id={}'''.format(batch_id), db)
    ###Create list to kick to workers
    arg_list=[]
    for k in range(len(block_list)):
        my_arg={}
        if len(mapped_models) > 0 and str(block_list[k]) in [b['block_id'] for b in mapped_models]:
            ###get the target filename
            target_mapping = [b for b in mapped_models if b['block_id']==str(block_list[k])][0]
            target_model = load_model(CONFIG,target_mapping['model_name'])
            my_arg['model'] = target_model
            my_arg['X_hdrs'] = target_model['variable_headers']
            my_arg['model_name'] = target_mapping['model_name']
            logger.info('Block {} will use model {}'.format(block_list[k],target_mapping['model_name']))
        else:
            ###Otherwise, use the default model
            my_arg['model']=model
            my_arg['X_hdrs'] = model['variable_headers']
            my_arg['model_name'] = CONFIG['saved_model']
        my_arg['target']=block_list[k]
        my_arg['block_info']=block
        my_arg['batch_id']=batch_id
        if len(block_list) > 10:
            if k % (len(block_list)/10)==0:
                my_arg['logging_flag']=int(np.round(100*(k/len(block_list)),0))
            else:
                my_arg['logging_flag'] = -1
        else:
            my_arg['logging_flag']=int(k)
        arg_list.append(my_arg)
    logger.info('STARTING TO MATCH FOR {}'.format(block['block_name']))
    pool=Pool(numWorkers)
    out=pool.map(match_fun, arg_list)
    ###check for any failures
    for o in range(len(out)):
        if out[0]=='fail':
            logger.info('Failed for {}, block {}, see log for details'.format(block['block_name'], arg_list[o]['target']))
            os._exit(0)
    ###Once that is done, need to
        ##push the remaining items in out to the db
    db=get_db_connection(CONFIG)
    cur=db.cursor()
    logger.info('Dumping remaining matches to DB for block {}'.format(block['block_name']))
    for i in out:
        if i!=None:
            if 'clerical_review_candidates' in i.keys():
                write_to_db(i['clerical_review_candidates'],'clerical_review_candidates')
            if 'matches' in i.keys():
                write_to_db(i['matches'], 'matched_pairs')
    db.commit()
    ##then change the matched flags on the data tables to 1 where it's been matched
    ###updating the matched flags
    if CONFIG['mode']!='deduplication':
        cur.execute('''update {data1_name} set matched=1 where id in (select distinct {data1_name}_id from matched_pairs)'''.format(data1_name=CONFIG['data1_name']))
        cur.execute('''update {data2_name} set matched=1 where id in (select distinct {data2_name}_id from matched_pairs)'''.format(data2_name=CONFIG['data2_name']))
    else:
        cur.execute('''update processing_per set matched=1 where id in (select distinct right_id from matched_pairs)'''.format(data1_name=CONFIG['data1_name']))
        cur.execute('''update processing_per set matched=1 where id in (select distinct left_id from matched_pairs)'''.format(data2_name=CONFIG['data2_name']))
    ##commit and close the DB
    db.commit()
    db.close()
    logger.info('Block {} Complete'.format(block['block_name']))

if __name__=='__main__':
    print('why did you do this?')
    os._exit(0)
    
    