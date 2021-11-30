###########
###CHUNK: THE RANDOM FOREST FUNCTIONS
############
from __future__ import division
import pandasql as ps
import copy
import os
from copy import deepcopy as dcpy
import numpy as np
from scipy.stats import randint as sp_randint
from programs.global_vars import *
from programs.logger_setup import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
import warnings
from sklearn.model_selection import StratifiedKFold
import programs.custom_scoring_methods as cust_scoring
from inspect import getmembers, isfunction

logger = logger_setup(CONFIG['log_file_name'])
from sklearn.ensemble import RandomForestClassifier
import random


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
    return (vif)


###Function to grab the predicted probabilities that are retruened from predict_proba as a tuple
def probsfunc(x):
    '''
    This function generates a single arrary of predicted probabilities  of a match
    inputs:
    probs: an array of predicted probabilites form an RF Classifier
    '''
    ith = np.zeros(shape=(len(x), 1))
    for i in range(len(x)):
        ith[i] = x[i][1]
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
    for i in range(len(values)):
        if header_types[i] == 'fuzzy':
            if np.isnan(values[i]).all() == True:
                values[i] = -1
            else:
                values[i] = np.round(values[i], 1)
        elif header_types[i] == 'numeric':
            if np.isnan(values[i]).all() == True:
                my_dims = nominal_info[header_names[i]]
                ##sub with 10 times the largest possible difference for the variable
                values[i] = 10 * (my_dims['max'] - my_dims['min'])
    return values


def create_scores(input_data, score_type, varlist, headers, method='full'):
    '''
    this function produces the list of dictionaries for all of the scores for the fuzzy variables.
    each dict will have two arguments: the array and a list of names
    :param input_data: Either a block to query the database with OR a dataframe of variable types
    :param score_type;: the type of score to calculate
    :param varlist: the variable list
    :param headers: the list of headers
    :param method: eithe r'training' or 'full' (the default).  if 'training', pull from the _training database tables
    NOTE FOR FUTURE SELF: IN NORMAL FUNCTIONS (MODEL_TRAINING=FALSE) THIS WILL ALREADY BE RUN IN A SUB-PROCESS
    SO YOU HAVE TO SINGLE-THREAD THE CALCULATIONS, SO THE WHOLE THING IS SINGLE-THREADED
    return: an array of values for each pair to match/variable and a list of headers
    '''
    ###If we are using truth/training data, use those table names.  Otherwise use the main tables
    if method == 'truth':
        table_1_name = '{}_training'.format(CONFIG['data1_name'])
        table_2_name = '{}_training'.format(CONFIG['data2_name'])
    else:
        table_1_name = CONFIG['data1_name']
        table_2_name = CONFIG['data2_name']
    db = get_db_connection(CONFIG)
    if score_type == 'fuzzy':
        ##first, let's see if we need to hit the data at all
        ####now for each pair, get the value for each fuzzy variable
        fuzzy_var_list = [i['variable_name'] for i in varlist]
        ####assemble the to_run list
        method_names = [i.__name__ for i in methods]
        to_run = []
        out_headers = []
        #####if we aren't using all variables
        if headers != 'all':
            for header in headers:
                if '_' in header:
                    my_name = header.split('_')
                    ###see if the name is in the method names
                    if my_name[1] in method_names:
                        to_run.append({'variable_name': my_name[0],
                                       'function': [meth for meth in methods if meth.__name__ == my_name[1]][0]})
                        out_headers.append('{}_{}'.format(my_name[0], my_name[1]))
        else:
            to_run = [{'variable_name': i['variable_name'], 'function': m} for i in varlist for m in methods]
            out_headers = ['{}_{}'.format(i['variable_name'], i['function'].__name__) for i in to_run]
        ###get the datai
        data1_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
        data2_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
        ###create an indexed list of the id pairs to serve as the core of our dictionary
        input_data = dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict = input_data[
            ['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]].to_dict('records')
        ###now get the values from the database
        data1_names = ','.join(['{} as {}'.format(i[CONFIG['data1_name']], i['variable_name']) for i in varlist])
        data2_names = ','.join(['{} as {}'.format(i[CONFIG['data2_name']], i['variable_name']) for i in varlist])
        ###get the values
        data1_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names,
                                                                                      table_name=table_1_name,
                                                                                      id_list=data1_ids), db)
        data2_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names,
                                                                                      table_name=table_2_name,
                                                                                      id_list=data2_ids), db)
        ###give the data values the name for each searching
        data1_values = {str(item['id']): item for item in data1_values}
        data2_values = {str(item['id']): item for item in data2_values}
        ###now create the dictionary with the list of names and the array
        out_arr = np.zeros(shape=(len(core_dict), len(to_run)))
        for i in range(len(core_dict)):
            i_scores = []
            for k in to_run:
                ####get the fuzzy var info
                fuzzy_var_info = [var for var in varlist if var['variable_name'] == k['variable_name']][0]
                if data1_values[core_dict[i]['{}_id'.format(CONFIG['data1_name'])]][k['variable_name']] != 'NULL' and \
                        data2_values[core_dict[i]['{}_id'.format(CONFIG['data2_name'])]][k['variable_name']] != 'NULL':
                    ####Two null match methods: either they get a 0 or the median value for the chunk
                    try:
                        i_scores.append(k['function'](data1_values[core_dict[i]['{}_id'.format(CONFIG['data1_name'])]][
                                                          fuzzy_var_info['variable_name']],
                                                      data2_values[core_dict[i]['{}_id'.format(CONFIG['data2_name'])]][
                                                          fuzzy_var_info['variable_name']]))
                    except:
                        i_scores.append(np.nan)
                    ###other method can go here if we think of one
                else:
                    i_scores.append(np.nan)
            out_arr[i,] = i_scores
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': out_headers}
    elif score_type == 'numeric_dist':
        ###numeric distance variables
        data1_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
        data2_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
        ###now get the values from the database
        data1_names = ','.join(['{} as {}'.format(i[CONFIG['data1_name']], i['variable_name']) for i in varlist])
        data2_names = ','.join(['{} as {}'.format(i[CONFIG['data2_name']], i['variable_name']) for i in varlist])
        ###get the values
        data1_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names,
                                                                                      table_name=table_1_name,
                                                                                      id_list=data1_ids), db)
        data2_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names,
                                                                                      table_name=table_2_name,
                                                                                      id_list=data2_ids), db)
        ###give the data values the name for each searching
        data1_values = {str(item['id']): item for item in data1_values}
        data2_values = {str(item['id']): item for item in data2_values}
        ####now for each pair, get the value for each variable
        ###create an indexed list of the id pairs to serve as the core of our dictionary
        input_data = dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict = input_data[
            ['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]].to_dict('records')
        numeric_distance_vars = [i['variable_name'] for i in varlist]
        ##convert fuzzy vars to a list
        ###now create the dictionary with the list of names and the array
        out_arr = np.zeros(shape=(len(core_dict), len(numeric_distance_vars)))
        for i in range(len(core_dict)):
            ###
            i_scores = []
            for j in range(len(numeric_distance_vars)):
                if data1_values[core_dict[i]['{}_id'.format(CONFIG['data1_name'])]][numeric_distance_vars[j]] and \
                        data2_values[core_dict[i]['{}_id'.format(CONFIG['data2_name'])]][numeric_distance_vars[j]]:
                    i_scores.append(
                        data1_values[core_dict[i]['{}_id'.format(CONFIG['data1_name'])]][numeric_distance_vars[j]] -
                        data2_values[core_dict[i]['{}_id'.format(CONFIG['data2_name'])]][numeric_distance_vars[j]])
                else:
                    i_scores.append(np.nan)
            out_arr[i,] = i_scores
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': numeric_distance_vars}
    elif score_type == 'exact':
        ###If we are running a training data model
        data1_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
        data2_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
        ###now get the values from the database
        data1_names = ','.join(['{} as {}'.format(i[CONFIG['data1_name']], i['variable_name']) for i in varlist])
        data2_names = ','.join(['{} as {}'.format(i[CONFIG['data2_name']], i['variable_name']) for i in varlist])
        ###get the values
        data1_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names,
                                                                                      table_name=table_1_name,
                                                                                      id_list=data1_ids), db)
        data2_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names,
                                                                                      table_name=table_2_name,
                                                                                      id_list=data2_ids), db)
        ###give the data values the name for each searching
        data1_values = {str(item['id']): item for item in data1_values}
        data2_values = {str(item['id']): item for item in data2_values}
        ####now for each pair, get the value for each variable
        ###create an indexed list of the id pairs to serve as the core of our dictionary
        input_data = dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict = input_data[
            ['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]].to_dict('records')
        exact_vars = [i['variable_name'] for i in varlist]
        ##convert fuzzy vars to a list
        ###now create the dictionary with the list of names and the array
        out_arr = np.zeros(shape=(len(core_dict), len(exact_vars)))
        for i in range(len(core_dict)):
            i_scores = []
            for j in range(len(exact_vars)):
                if data1_values[core_dict[i]['{}_id'.format(CONFIG['data1_name'])]][exact_vars[j]] and \
                        data2_values[core_dict[i]['{}_id'.format(CONFIG['data2_name'])]][exact_vars[j]]:
                    if str(data1_values[core_dict[i]['{}_id'.format(CONFIG['data1_name'])]][
                               exact_vars[j]]).upper() == str(
                            data2_values[core_dict[i]['{}_id'.format(CONFIG['data2_name'])]][exact_vars[j]]).upper():
                        i_scores.append(1)
                    else:
                        i_scores.append(0)
                else:
                    i_scores.append(-1)
            out_arr[i,] = i_scores
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': exact_vars}
    elif score_type == 'geo_distance':
        data1_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
        data2_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
        ###now get the values from the database
        ###get the values
        data1_values = get_table_noconn(
            '''select id, latitude, longitude from {table_name} where id in ({id_list})'''.format(
                table_name=table_1_name, id_list=data1_ids), db)
        data2_values = get_table_noconn(
            '''select id, latitude, longitude from {table_name} where id in ({id_list})'''.format(
                table_name=table_2_name, id_list=data2_ids), db)
        input_data = dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict = input_data[
            ['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]].to_dict('records')
        out_arr = np.zeros(shape=(len(core_dict), 1))
        for i in range(len(core_dict)):
            data1_target = \
            [k for k in data1_values if str(k['id']) == str(core_dict[i]['{}_id'.format(CONFIG['data1_name'])])][0]
            data2_target = \
            [k for k in data2_values if str(k['id']) == str(core_dict[i]['{}_id'.format(CONFIG['data2_name'])])][0]
            if data1_target['latitude'] is not None and data1_target['longitude'] is not None and data2_target[
                'latitude'] is not None and data2_target['longitude'] is not None:
                out_arr[i] = haversine(tuple([data1_target['latitude'], data1_target['longitude']]),
                                       tuple([data2_target['latitude'], data2_target['longitude']]))
            else:
                ####if it's missing, make it the entire diameter of the earth away (but negative to ensure the model can differentiate)
                out_arr[i] = - 12742
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': ['geo_distance']}
    elif score_type == 'date':
        ###If we are running a training data model
        data1_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
        data2_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
        ###now get the values from the database
        data1_names = ','.join(['{} as {}'.format(i[CONFIG['data1_name']], i['variable_name']) for i in varlist])
        data2_names = ','.join(['{} as {}'.format(i[CONFIG['data2_name']], i['variable_name']) for i in varlist])
        ###get the values
        data1_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names,
                                                                                      table_name=table_1_name,
                                                                                      id_list=data1_ids), db)
        data2_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names,
                                                                                      table_name=table_2_name,
                                                                                      id_list=data2_ids), db)
        ###give the data values the name for each searching
        data1_values = {str(item['id']): item for item in data1_values}
        data2_values = {str(item['id']): item for item in data2_values}
        ####now for each pair, get the value for each variable
        ###create an indexed list of the id pairs to serve as the core of our dictionary
        input_data = dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict = input_data[
            ['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]].to_dict('records')
        date_vars = [i['variable_name'] for i in varlist]
        ##convert fuzzy vars to a list
        ###now create the dictionary with the list of names and the array
        out_arr = np.zeros(shape=(len(core_dict), len(date_vars)))
        for i in range(len(core_dict)):
            i_scores = []
            for j in range(len(date_vars)):
                if data1_values[core_dict[i]['{}_id'.format(CONFIG['data1_name'])]][date_vars[j]] and \
                        data2_values[core_dict[i]['{}_id'.format(CONFIG['data2_name'])]][date_vars[j]]:
                    i_scores.append(
                        feb.editdist(data1_values[core_dict[i]['{}_id'.format(CONFIG['data1_name'])]][date_vars[j]],
                                     data2_values[core_dict[i]['{}_id'.format(CONFIG['data2_name'])]][date_vars[j]]))
                else:
                    i_scores.append(0)
            out_arr[i,] = i_scores
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': date_vars}
    elif score_type == 'custom':
        ###If we are running a training data model
        data1_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
        data2_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
        ###now get the values from the database
        data1_names = ','.join(['''(case when {}='NULL' then NULL else {} end) as {}'''.format(i[CONFIG['data1_name']],
                                                                                               i[CONFIG['data1_name']],
                                                                                               i['variable_name']) for i
                                in varlist])
        data2_names = ','.join(['''(case when {}='NULL' then NULL else {} end) as {}'''.format(i[CONFIG['data2_name']],
                                                                                               i[CONFIG['data2_name']],
                                                                                               i['variable_name']) for i
                                in varlist])
        ###get the values
        data1_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names,
                                                                                      table_name=table_1_name,
                                                                                      id_list=data1_ids), db)
        data2_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names,
                                                                                      table_name=table_2_name,
                                                                                      id_list=data2_ids), db)
        ###give the data values the name for each searching
        data1_values = {str(item['id']): item for item in data1_values}
        data2_values = {str(item['id']): item for item in data2_values}
        ####now for each pair, get the value for each variable
        ###create an indexed list of the id pairs to serve as the core of our dictionary
        input_data = dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict = input_data[
            ['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]].to_dict('records')
        custom_vars = [i['variable_name'] for i in varlist]
        ##convert fuzzy vars to a list
        ###now create the dictionary with the list of names and the array
        ####get the right arrangement of custom functions and targets
        custom_scoring_functions = [{'name': i[0], 'function': i[1]} for i in getmembers(cust_scoring, isfunction)]
        ###get the corresponding function attached to the var list
        for i in varlist:
            m = [m for m in getmembers(cust_scoring, isfunction) if i['custom_variable_name'] == m[0]][0]
            if m:
                i['function'] = m[1]
        out_arr = np.zeros(shape=(len(core_dict), len(custom_vars)))
        for i in range(len(core_dict)):
            i_scores = []
            for j in range(len(varlist)):
                #####NOTE HERE.  YOU MUST FIGURE OUT HOW TO RETURN NULL VALUES IN YOUR FUNCTION
                #####I know this seems lazy on my part, BUT there's too many alternatives (impute? just 0?)
                #####that vary with what you're trying to do. (Also yes, it's lazy on my part)
                i_scores.append(varlist[j]['function'](
                    data1_values[core_dict[i]['{}_id'.format(CONFIG['data1_name'])]][varlist[j]['variable_name']],
                    data2_values[core_dict[i]['{}_id'.format(CONFIG['data2_name'])]][varlist[j]['variable_name']]))
            out_arr[i,] = i_scores
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': custom_vars}


def create_all_scores(input_data, method, headers='all'):
    '''
    This function takes input data and creates all the scores needed
    :param input_data: the data we want to generate scores for
    :param method: either "prediction" or "truth".  If prediction, just return an array of X values.  If "Truth" return X, and X_hdrs
    :return: y, X_imputed, and X_hdrs
    '''
    ###get the varible types
    var_rec = pd.read_csv('mamba_variable_types.csv', keep_default_na=False).replace({'': None}).to_dict('records')
    ###add possible headers
    for v in var_rec:
        if v['match_type'] == 'fuzzy':
            v['possible_headers'] = ['{}_{}'.format(v['variable_name'], meth.__name__) for meth in methods]
    ###We have three types of variables.
    # 1) First, identify all the variables we are going to need.  Fuzzy, exact, numeric_dist
    if headers == 'all':
        fuzzy_vars = [i for i in var_rec if i['match_type'] == 'fuzzy']
        numeric_dist_vars = [i for i in var_rec if i['match_type'] == 'num_distance']
        exact_match_vars = [i for i in var_rec if i['match_type'] == 'exact']
        geo_distance = [i for i in var_rec if i['match_type'] == 'geom_distance']
        date_vars = [i for i in var_rec if i['match_type'] == 'date']
        custom_vars = [i for i in var_rec if i['custom_variable_name'] if i['custom_variable_name'] is not None]
    else:
        ###If we are running with recursive feature elimination, just find the features we are going to use
        fuzzy_vars = [i for i in var_rec if
                      i['match_type'] == 'fuzzy' and any(item in headers for item in i['possible_headers']) == True]
        numeric_dist_vars = [i for i in var_rec if i['match_type'] == 'num_distance' and i['variable_name'] in headers]
        exact_match_vars = [i for i in var_rec if i['match_type'] == 'exact' and i['variable_name'] in headers]
        geo_distance = [i for i in var_rec if i['match_type'] == 'geom_distance' and 'geo_distance' in headers]
        date_vars = [i for i in var_rec if i['match_type'] == 'date' and i['variable_name'] in headers]
        custom_vars = [i for i in var_rec if i['custom_variable_name'] if
                       i['custom_variable_name'] is not None and i['variable_name'] in headers]
    ###Are we running any fuzzy_vars?
    if len(fuzzy_vars) > 0:
        fuzzy_values = create_scores(input_data, 'fuzzy', fuzzy_vars, headers, method)
        X = fuzzy_values['output']
        X_hdrs = copy.deepcopy(fuzzy_values['names'])
    # 2) num_distance: get the numeric distance between the two values, with a score of -9999 if missing
    if len(numeric_dist_vars) > 0:
        numeric_dist_values = create_scores(input_data, 'numeric_dist', numeric_dist_vars, headers, method)
        X = np.hstack((X, numeric_dist_values['output']))
        X_hdrs.extend(numeric_dist_values['names'])
    # 3) exact: if they match, 1, else 0
    if len(exact_match_vars) > 0:
        exact_match_values = create_scores(input_data, 'exact', exact_match_vars, headers, method)
        X = np.hstack((X, exact_match_values['output']))
        X_hdrs.extend(exact_match_values['names'])
    ###Geo Distance
    if len(geo_distance) > 0:
        geo_distance_values = create_scores(input_data, 'geo_distance', 'lat', headers, method)
        if type(geo_distance_values['output']) != string:
            X = np.hstack((X, geo_distance_values['output']))
            X_hdrs.extend(geo_distance_values['names'])
    ####date vars
    if len(date_vars) > 0:
        date_values = create_scores(input_data, 'date', date_vars, headers, method)
        if type(date_values['output']) != string:
            X = np.hstack((X, date_values['output']))
            X_hdrs.extend(date_values['names'])
    ###custom values
    if len(custom_vars) > 0:
        custom_values = create_scores(input_data, 'custom', custom_vars, headers, method)
        if type(custom_values['output']) != string:
            X = np.hstack((X, custom_values['output']))
            X_hdrs.extend(custom_values['names'])
    ##Impute the missing data
    ####Options for missing data: Impute, or Interval:
    ##Imputation: Just impute the scores based on an iterative imputer
    if CONFIG['imputation_method'] == 'Imputer':
        imp = IterativeImputer(max_iter=10, random_state=0)
        ###fit the imputation
        imp.fit(X)
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
        nominal_boundaries = {}
        if len(numeric_dist_vars) > 0:
            for var in numeric_dist_vars:
                ###get the values from the database
                my_values = get_table_noconn('''select min(min_1, min_2) min, max(max_1, max_2) max
                                                 from 
                                                 (select '0' id, min({data1_var}) min_1, max({data1_var}) max_1 from {table1_name})
                                                 left outer join 
                                                 (select '0' id, min({data2_var}) min_2, max({data2_var}) max_2 from {table2_name})'''.format(
                    table1_name=CONFIG['data1_name'],
                    table2_name=CONFIG['data2_name'],
                    data1_var=var[CONFIG['data1_name']],
                    data2_var=var[CONFIG['data2_name']]), db)
                ###save as a dictionary entry
                nominal_boundaries[var['variable_name']] = my_values[0]
        ###first, get the min and max values for the possible nominal variables
        header_types = []
        for k in range(len(X_hdrs)):
            if X_hdrs[k] in fuzzy_values['names']:
                header_types.append('fuzzy')
            elif X_hdrs[k] in [j['variable_name'] for j in numeric_dist_vars]:
                header_types.append('numeric')
            else:
                header_types.append('other')
        ###Apply the nominal imputation function to each row
        X = np.apply_along_axis(nominal_impute, 1, X, header_names=X_hdrs, header_types=header_types,
                                nominal_info=nominal_boundaries)
        ###making the dependent variable array
        if method == 'truth':
            y = input_data['match'].values
            return y, X, X_hdrs, 'Nominal'
        else:
            return X, X_hdrs


def filter_data(data):
    '''
    This function takes a customized filter and applies to to the input data, basically limited the number
    of observations we want to consider
    :param data: the input_data ids
    :return:
    '''
    ####first get the data
    db = get_db_connection(CONFIG)
    data1_ids = ','.join(str(v) for v in data['{}_id'.format(CONFIG['data1_name'])].drop_duplicates().tolist())
    data2_ids = ','.join(str(v) for v in data['{}_id'.format(CONFIG['data2_name'])].drop_duplicates().tolist())
    ###create an indexed list of the id pairs to serve as the core of our dictionary
    data.reset_index(inplace=True, drop=False)
    data = data[['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]]
    ###get the values
    myvar = \
    [i for i in var_types if i['variable_name'].lower() == CONFIG['variable_filter_info']['variable_name'].lower()][0]
    if myvar['match_type'] != 'geo_distance':
        data1_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as character) id, {var_name} data1_target from {table_name} where id in ({id_list})'''.format(
                var_name=myvar[CONFIG['data1_name']],
                table_name=CONFIG['data1_name'],
                id_list=data1_ids), db))
        data2_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as character) id, {var_name} 
            data2_target from {table_name} where id in ({id_list})'''.format(var_name=myvar[CONFIG['data2_name']],
                                                                             table_name=CONFIG['data2_name'],
                                                                             id_list=data2_ids), db))
    else:
        data1_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as character) id, 
            latitude data1_latitude,
             longitude data1_longitude from {table_name} where id in ({id_list})'''.format(
                table_name=CONFIG['data1_name'],
                id_list=data1_ids), db))
        data2_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as character) id, 
            latitude data2_latitude,
             longitude data2_longitude from {table_name} where id in ({id_list})'''.format(
                table_name=CONFIG['data2_name'],
                id_list=data2_ids), db))
    ###merge the three dataframes
    data = data.merge(data1_values, left_on='{}_id'.format(CONFIG['data1_name']), right_on='id')
    data = data.merge(data2_values, left_on='{}_id'.format(CONFIG['data2_name']), right_on='id')
    ###identify the function we need to run:
    if myvar['match_type'] == 'fuzzy':
        target_fun = [i for i in methods if i.__name__ == CONFIG['variable_filter_info']['fuzzy_name']][0]
        data['score'] = data.apply(
            lambda x: target_fun(x['data1_target'], x['data2_target']) if x['data1_target'] != 'NULL' and x[
                'data2_target'] != 'NULL'
            else 0, axis=1)
    ###Now the exact matches##
    elif myvar['match_type'] == 'exact':
        data['score'] = np.where(data['data1_target'] == data['data2_target'], 1, 0)
    elif myvar['match_type'] == 'num_distance':
        data['score'] = data['data1_target'] - data['data2_target']
    elif myvar['match_type'] == 'geo_distance':
        data['score'] = data.apply(lambda x: haversine((x['data1_latitude'], x['data1_longitude']),
                                                       (x['data2_latitude'], x['data2_longitude'])), axis=1)
    elif myvar['match_type'] == 'date':
        data['score'] = data.apply(lambda x: feb.editdist(x['data1_target'], x['data2_target']), axis=1)
    elif myvar['match_type'] == 'custom':
        ####get the right arrangement of custom functions and targets
        custom_scoring_functions = [{'name': i[0], 'function': i[1]} for i in getmembers(cust_scoring, isfunction)]
        ###get the corresponding function attached to the var list
        my_function = [k for k in custom_scoring_functions if k['name'].lower() == myvar['variable_name'].lower()][0]
        data['score'] = data.apply(
            lambda x: my_function['function'](x['data1_target'], x['data2_target']) if x['data1_target'] != 'NULL' and
                                                                                       x['data2_target'] != 'NULL'
            else 0, axis=1)
    ####now return the columns needed with just the rows that meet the criteria
    qry = '''score {} {}'''.format(CONFIG['variable_filter_info']['test'],
                                   CONFIG['variable_filter_info']['filter_value'])
    data = data.query(qry)
    if len(data) > 0:
        return data[['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]]
    else:
        return pd.DataFrame(
            columns=['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])])


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
    if ast.literal_eval(CONFIG['debugmode']) == True:
        niter = 5
    else:
        niter = 200
    ##Mean-Center
    X = pd.DataFrame(X)
    ##Save the means
    X.columns = X_hdrs
    X_means = X.mean().to_dict()
    X = X - X.mean()
    if ast.literal_eval(CONFIG['feature_elimination_mode']) == False:
        myparams = {
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        mod = LogisticRegression(random_state=0, solver='liblinear')
        cv_rfc = RandomizedSearchCV(estimator=mod, param_distributions=myparams, cv=10, scoring=scoringcriteria,
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
            return {'type': 'Logistic Regression', 'score': score, 'model': mod, 'means': X_means,
                    'variable_headers': X_hdrs}

        except Exception as error:
            logger.warning(
                "Warning.  Unable to calculate VIF, error {}.  Restart MAMBA with logit disabled. Proceeding by ignoring Logit".format(
                    error))
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
            logger.warning(
                "Warning.  Unable to calculate VIF, error {}.  Restart MAMBA with logit disabled. Proceeding by ignoring Logit".format(
                    error))
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
        return {'type': 'Logistic Regression', 'score': score, 'model': mod, 'variable_headers': new_X_hdrs,
                'means': X_means}


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
    if ast.literal_eval(CONFIG['debugmode']) == True:
        niter = 1
    else:
        niter = 200
    if ast.literal_eval(CONFIG['feature_elimination_mode']) == False:
        cv_rfc = RandomizedSearchCV(estimator=rf, param_distributions=myparams, cv=5, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        ##Save the parameters
        trees = cv_rfc.best_params_['n_estimators']
        features_per_tree = cv_rfc.best_params_['max_features']
        score = cv_rfc.best_score_
        max_depth = cv_rfc.best_params_['max_depth']
        ###No obvious need for MAMBALITE here
        ##In future could just remove the worst performaning variable using a relimp measure then run the full model
        # if mambalite == False:
        logger.info('Random Forest Completed.  Score {}'.format(score))
        rf_mod = runRFClassifier(y, X, trees, features_per_tree, max_depth)
        return {'type': 'Random Forest', 'score': score, 'model': rf_mod, 'variable_headers': X_hdrs}
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
        rf_mod = runRFClassifier(y, X, cv_rfc.best_estimator_.get_params()['estimator__n_estimators'],
                                 cv_rfc.best_estimator_.get_params()['estimator__max_features'],
                                 cv_rfc.best_estimator_.get_params()['estimator__max_depth'])
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
    if ast.literal_eval(CONFIG['debugmode']) == True:
        niter = 5
    else:
        niter = int(np.round(len(X) / float(2), 0))
    features_per_tree = ['sqrt', 'log2', 10, 15]
    myparams = {
        'n_estimators': sp_randint(1, 25),
        'algorithm': ['SAMME', 'SAMME.R']}
    if ast.literal_eval(CONFIG['feature_elimination_mode']) == False:
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
        return {'type': 'AdaBoost', 'score': score, 'model': ada_mod, 'variable_headers': X_hdrs}
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
        ada_mod = AdaBoostClassifier(algorithm=cv_rfc.best_estimator_.get_params()['estimator__algorithm'],
                                     n_estimators=cv_rfc.best_estimator_.get_params()['estimator__n_estimators'])
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
        ada_mod.fit(new_X, y)
        score = np.round(ada_mod.score(new_X, y), 5)
        ###limit X to just what we need
        logger.info('AdaBoost Complete.  Score={}'.format(score))
        return {'type': 'AdaBoost', 'score': score, 'model': ada_mod, 'variable_headers': new_X_hdrs}


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
    if ast.literal_eval(CONFIG['debugmode']) == True:
        niter = 5
    else:
        niter = int(np.round(len(X) / float(2), 0))
    myparams = {
        'kernel': ['linear', 'poly', 'rbf']
    }
    if ast.literal_eval(CONFIG['feature_elimination_mode']) == False:
        ##First, get the scores
        svn_rfc = RandomizedSearchCV(estimator=svc, param_distributions=myparams, cv=5, scoring=scoringcriteria)
        svn_rfc.fit(X, y)
        ##Save the parameters
        kernel = svn_rfc.best_params_['kernel']
        score = svn_rfc.best_score_
        logger.info('SVM Complete.  Max Score={}'.format(score))
        svc = svm.SVC(gamma=gamma, degree=degree, kernel=kernel)
        ###Note for when you return--you need to change the predict function to do cross_val_predict
        return {'type': 'SVM', 'score': score, 'model': svc, 'variable_headers': X_hdrs}
    else:
        ##First, get the scores
        myparams = {
            'estimator__kernel': ['linear', 'poly', 'rbf']}
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
        svn_mod.fit(new_X, y)
        score = np.round(svn_mod.score(new_X, y), 5)
        ###limit X to just what we need
        logger.info('Support Vector Machine Complete.  Score={}'.format(score))
        return {'type': 'SVN', 'score': score, 'model': svn_mod, 'variable_headers': new_X_hdrs}


def choose_model(truthdat):
    '''
    This function generates a random forest, svn, and adaboost classifier for the training data, then returns the
    :param truthdat:
    :return:
    '''
    models = []
    y, X, X_hdrs, imputer = create_all_scores(truthdat, method='truth')
    if ast.literal_eval(CONFIG['use_logit']) == True:
        logit = generate_logit(y, X, X_hdrs)
        if logit != 'fail':
            logit = {'score': 0}
    else:
        logit = {'score': 0}
    models.append(logit)
    rf = generate_rf_mod(y, X, X_hdrs)
    models.append(rf)
    # svn=generate_svn_mod(truthdat)
    svn = {'score': 0}
    models.append(svn)
    ada = generate_ada_boost(y, X, X_hdrs)
    models.append(ada)
    ###Are we using the custom model?
    if ast.literal_eval(CONFIG['use_custom_model']) == True:
        import programs.custom_model as cust
        logger.info('\n\n######CREATE CUSTOM MODEL {}######\n\n'.format(cust.my_model_name))
        custom_model = cust.my_custom_model(y, X, X_hdrs)
        models.append(custom_model)
    ###ID the best score
    best_score = max([i['score'] for i in models])
    best_model = [i for i in models if i['score'] == best_score][0]
    logger.info('Selected the {} Model, with the score of {}'.format(best_model['type'], best_model['score']))
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
    var_rec: the variable records, showing what type of match to perform on each variable
    '''
    ###Sometimes, the arg will have a logging flag if we are 10% of the way through
    start = time.time()
    if arg['logging_flag'] != -1:
        logger.info('{}% complete with block'.format(arg['logging_flag']))
    db = get_db_connection(CONFIG, timeout=1)
    ###get the two dataframes
    data1 = get_table_noconn('''select id from {} where {}='{}' and matched=0'''.format(CONFIG['data1_name'],
                                                                                        arg['block_info'][
                                                                                            CONFIG['data1_name']],
                                                                                        arg['target']), db)
    data2 = get_table_noconn('''select id from {} where {}='{}' and matched=0'''.format(CONFIG['data2_name'],
                                                                                        arg['block_info'][
                                                                                            CONFIG['data2_name']],
                                                                                        arg['target']), db)
    ###get the data
    ###If we are running in deduplication mode, take only the input records that the IDs don't match
    if ast.literal_eval(CONFIG['ignore_duplicate_ids']) == True:
        input_data = pd.DataFrame(
            [{'{}_id'.format(CONFIG['data1_name']): str(k['id']), '{}_id'.format(CONFIG['data2_name']): str(y['id'])}
             for k in data1 for y in data2 if str(k['id']) != str(y['id'])])
    else:
        input_data = pd.DataFrame(
            [{'{}_id'.format(CONFIG['data1_name']): str(k['id']), '{}_id'.format(CONFIG['data2_name']): str(y['id'])}
             for k in data1 for y in data2])
    ####Now get the data organized, by creating an array of the different types of variables
    if len(input_data) == 0:
        end = time.time() - start
        logger.info('There were no valid matches to attempt for block {}'.format(arg['target']))
        stats_dat = {'batch_id': CONFIG['batch_id'],
                     'block_level': arg['block_info']['block_name'],
                     'block_id': arg['target'],
                     'block_time': end,
                     'block_size': 0,
                     'block_matches': 0,
                     'block_matches_avg_score': 0,
                     'block_non_matches': 0,
                     'block_non_matches_avg_score': 0}
    else:
        if ast.literal_eval(CONFIG['use_variable_filter']) == True:
            ###if we are filtering, return the input data that meets the criteria
            orig_len = len(input_data)
            input_data = filter_data(input_data)
            filtered_len = orig_len - len(input_data)
        if len(input_data) == 0:
            end = time.time() - start
            logger.info('After filtering, there were no valid matches to attempt for block {}'.format(arg['target']))
            stats_dat = {'batch_id': CONFIG['batch_id'],
                         'block_level': arg['block_info']['block_name'],
                         'block_id': arg['target'],
                         'block_time': end,
                         'block_size': 0,
                         'block_matches': 0,
                         'block_matches_avg_score': 0,
                         'block_non_matches': 0,
                         'block_non_matches_avg_score': 0,
                         'match_pairs_removed_filter': orig_len - len(input_data)}
        else:
            X, X_hdrs = create_all_scores(input_data, 'prediction', arg['model']['variable_headers'])
            ###Mean Center
            if arg['model']['type'] == 'Logistic Regression':
                X = pd.DataFrame(X, columns=X_hdrs) - arg['model']['means']
                X = np.array(X)
            myprediction = probsfunc(arg['model']['model'].predict_proba(X))
            ####don't need the scores anymore
            del X
            del data1
            del data2
            input_data['predicted_probability'] = myprediction
            ###keep only the data above the certain thresholds
            if ast.literal_eval(CONFIG['clerical_review_candidates']) == True:
                keep_thresh = min(float(CONFIG['clerical_review_threshold']), float(CONFIG['match_threshold']))
            else:
                keep_thresh = float(CONFIG['match_threshold'])
            stats_dat = copy.deepcopy(input_data)
            input_data = input_data.loc[input_data['predicted_probability'] >= keep_thresh].to_dict('records')
            ####Now write to the DB
            if len(input_data) > 0:
                ###try to write to db, if not return and then we will dump later
                cur = db.cursor()
                ####if we are looking for clerical review candidates, push either all the records OR a 10% sample, depending on which is larger
                clerical_review_sql = '''insert into clerical_review_candidates({}_id, {}_id, predicted_probability) values (?,?,?) '''.format(
                    CONFIG['data1_name'], CONFIG['data2_name'])
                match_sql = '''insert into matched_pairs({data1}_id, {data2}_id, predicted_probability, {data1}_rank, {data2}_rank) values (?,?,?,?,?) '''.format(
                    data1=CONFIG['data1_name'], data2=CONFIG['data2_name'])
                ###setup the to_return
                to_return = {}
                if ast.literal_eval(CONFIG['clerical_review_candidates']) == True:
                    if len(input_data) >= 10:
                        clerical_review_dict = dcpy(random.sample(input_data, int(np.round(.3 * len(input_data), 0))))
                    else:
                        clerical_review_dict = dcpy(input_data)
                    columns = clerical_review_dict[0].keys()
                    vals = [tuple(i[column] for column in columns) for i in clerical_review_dict]
                    try:
                        cur.executemany(clerical_review_sql, vals)
                        db.commit()
                    except Exception:
                        cur.close()
                        cur = db.cursor()
                        to_return['clerical_review_candidates'] = input_data
                    ###if it's a chatty logger, log.
                    if ast.literal_eval(CONFIG['chatty_logger']) == True:
                        logger.info('''{} clerical review candidates added'''.format(len(clerical_review_dict)))
                ####Now add in any matches
                matches = pd.DataFrame(
                    [i for i in input_data if i['predicted_probability'] >= float(CONFIG['match_threshold'])])
                if len(matches) > 0:
                    ###ranks
                    matches['{}_rank'.format(CONFIG['data1_name'])] = \
                    matches.groupby('{}_id'.format(CONFIG['data1_name']))[
                        'predicted_probability'].rank('dense')
                    matches['{}_rank'.format(CONFIG['data2_name'])] = \
                    matches.groupby('{}_id'.format(CONFIG['data2_name']))[
                        'predicted_probability'].rank('dense')
                    ##convert back to dict
                    matches = matches.to_dict('records')
                    columns = matches[0].keys()
                    vals = [tuple(i[column] for column in columns) for i in matches]
                    try:
                        cur.executemany(match_sql, vals)
                        db.commit()
                        if ast.literal_eval(CONFIG['chatty_logger']) == True:
                            logger.info('''{} matches added in block'''.format(len(matches)))
                    except Exception:
                        to_return['matches'] = matches
                cur.close()
                db.close()
                if 'clerical_review_candidates' in to_return.keys() or 'matches' in to_return.keys():
                    return to_return
                else:
                    return None
            ###do the block statistics
            end = time.time() - start
            stats_dat = {'batch_id': CONFIG['batch_id'],
                         'block_level': str(arg['block_info']['block_name']),
                         'block_id': str(arg['target']),
                         'block_time': end,
                         'block_size': len(stats_dat),
                         'block_matches': np.sum(
                             np.where(stats_dat['predicted_probability'] >= float(CONFIG['match_threshold']), 1, 0)),
                         'block_matches_avg_score': np.nanmean(
                             np.where(stats_dat['predicted_probability'] >= float(CONFIG['match_threshold']),
                                      stats_dat['predicted_probability'], np.nan)),
                         'block_non_matches': np.sum(
                             np.where(stats_dat['predicted_probability'] < float(CONFIG['match_threshold']), 1, 0)),
                         'block_non_matches_avg_score': np.nanmean(
                             np.where(stats_dat['predicted_probability'] < float(CONFIG['match_threshold']),
                                      stats_dat['predicted_probability'], np.nan)),
                         'match_pairs_removed_filter': filtered_len}
    db = get_db_connection(CONFIG)
    cur = db.cursor()
    columns_list = str(tuple([str(j) for j in stats_dat.keys()])).replace("'", '')
    values = tuple(stats_dat[column] for column in stats_dat.keys())
    vals_len = ','.join(['?' for _ in range(len(stats_dat.keys()))])
    insert_statement = '''insert into batch_statistics {} VALUES({})'''.format(columns_list, vals_len)
    cur.execute(insert_statement, values)
    db.close()


####The block function
def run_block(block, model):
    '''
    This function creates the list of blocks to use for comparison and then runs the matches
    :param block: the dict of the block we are running
    :param rf_mod; the random forest model we are using
    :return:
    '''
    ####First get the blocks that appear in both
    db = get_db_connection(CONFIG)
    data1_blocks = get_table_noconn(
        '''select distinct {} as block from {}'''.format(block[CONFIG['data1_name']], CONFIG['data1_name']), db)
    data1_blocks = [i['block'] for i in data1_blocks]
    data2_blocks = get_table_noconn(
        '''select distinct {} as block from {}'''.format(block[CONFIG['data2_name']], CONFIG['data2_name']), db)
    data2_blocks = [i['block'] for i in data2_blocks]
    block_list = intersection(data1_blocks, data2_blocks)
    logger.info('We have {} blocks to run for {}'.format(len(block_list), block['block_name']))
    ###get the list of variables we are trying to match
    var_rec = pd.read_csv('mamba_variable_types.csv').to_dict('records')
    ###Create and kick off the runner
    arg_list = []
    for k in range(len(block_list)):
        my_arg = {}
        my_arg['var_rec'] = var_rec
        my_arg['model'] = model
        my_arg['X_hdrs'] = model['variable_headers']
        my_arg['target'] = block_list[k]
        my_arg['block_info'] = block
        if len(block_list) > 10:
            if k % (len(block_list) / 10) == 0:
                my_arg['logging_flag'] = int(np.round(100 * (k / len(block_list)), 0))
            else:
                my_arg['logging_flag'] = -1
        else:
            my_arg['logging_flag'] = int(k)
        arg_list.append(my_arg)
    logger.info('STARTING TO MATCH  FOR {}'.format(block['block_name']))
    pool = Pool(numWorkers)
    out = pool.map(match_fun, arg_list)
    ###Once that is done, need to
    ##push the remaining items in out to the db
    db = get_db_connection(CONFIG)
    cur = db.cursor()
    logger.info('Dumping remaining matches to DB for block {}'.format(block['block_name']))
    clerical_review_sql = '''insert into clerical_review_candidates({}_id, {}_id, predicted_probability) values (?,?,?) '''.format(
        CONFIG['data1_name'], CONFIG['data2_name'])
    match_sql = '''insert into matched_pairs({data1}_id, {data2}_id, predicted_probability, {data1}_rank, {data2}_rank) values (?,?,?,?,?) '''.format(
        data1=CONFIG['data1_name'], data2=CONFIG['data2_name'])
    for i in out:
        if i != None:
            if 'clerical_review_candidates' in i.keys():
                columns = i['clerical_review_candidates'][0].keys()
                vals = [tuple(j[column] for column in columns) for j in i['clerical_review_candidates']]
                cur.executemany(clerical_review_sql, vals)
            if 'matches' in i.keys():
                columns = i['matches'][0].keys()
                vals = [tuple(j[column] for column in columns) for j in i['matches']]
                cur.executemany(match_sql, vals)
    db.commit()
    ##then change the matched flags on the data tables to 1 where it's been matched
    ###updating the matched flags
    cur.execute(
        '''update {data1_name} set matched=1 where id in (select distinct {data1_name}_id from matched_pairs)'''.format(
            data1_name=CONFIG['data1_name']))
    cur.execute(
        '''update {data2_name} set matched=1 where id in (select distinct {data2_name}_id from matched_pairs)'''.format(
            data2_name=CONFIG['data2_name']))
    db.commit()
    db.close()
    logger.info('Block {} Complete'.format(block['block_name']))


if __name__ == '__main__':
    print('why did you do this?')
    os._exit(0)

