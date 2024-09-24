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
from programs.create_db_helpers import update_batch_summary
from programs.create_db_helpers import get_db_connection, get_table_noconn
from programs.model_load_save_helpers import load_model
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
import itertools
from programs.model_generators import *
from programs.score_functions import *
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


#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer

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
        out_arr = np.stack([np.apply_along_axis(numeric_distance, 1, [(data1_values[x['data1_id']][y['variable_name']],data2_values[x['data2_id']][y['variable_name']]) for x in core_dict]) for y in varlist], axis=1)
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
            ####creating the headers.
            ###NOTE: You need to flag any method that uses the fuzzy variables
            if 'is_fuzzy' in i['custom_variable_kwargs'].keys() and ast.literal_eval(i['custom_variable_kwargs']['is_fuzzy']) == True:
                i['custom_variable_kwargs']['headers'] = [k.replace('{}_'.format(i['variable_name']),'') for k in i['possible_headers']]
                ###now we remove the 'is_fuzzy' argument to not mess with the keywords
                i['custom_variable_kwargs'].pop('is_fuzzy')
            else:
                i['custom_variable_kwargs']['headers'] = [i['variable_name']]
        function_vals = [create_custom_scores(core_dict, data1_values, data2_values, x) for x in varlist]
        ##figuring out the length
        column_len = 0
        for i in function_vals:
            if isinstance(i[0], list):
                column_len += len(i[0])
            else:
                column_len += 1
        ###create an array of zeros
        out_arr = np.zeros((len(function_vals[0]), column_len))
        columns_used = 0
        for i in range(len(function_vals)):
            if isinstance(function_vals[i][0], list):
                out_arr[:, columns_used:columns_used + len(function_vals[i][0])] = function_vals[i]
                columns_used += len(function_vals[i][0])
            else:
                out_arr[:, columns_used] = function_vals[i]
                columns_used += 1
        ####Now return a dictionary of the input array and the names (note the headers are lists of lists so we need to unpack them
        return {'output': out_arr, 'names': list(itertools.chain.from_iterable([i['possible_headers'] for i in varlist]))}
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
    var_rec = copy.deepcopy([v for v in var_types if v['filter_only']=={}])
    for v in var_rec:
        if v['match_type']=='fuzzy':
            v['possible_headers'] = ['{}_{}'.format(v['variable_name'], meth.__name__) for meth in methods]
        if v['match_type']=='custom':
            if 'is_fuzzy' in v['custom_variable_kwargs'].keys() and v['custom_variable_kwargs']['is_fuzzy']=='True':
                if headers=='all':
                    v['possible_headers'] = ['{}_{}'.format(v['variable_name'], meth.__name__) for meth in methods]
                else:
                    v['possible_headers'] = ['{}_{}'.format(v['variable_name'], meth.__name__) for meth in methods if '{}_{}'.format(v['variable_name'], meth.__name__) in headers]
            else:
                v['possible_headers'] = [v['variable_name']]
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
        custom_vars = [i for i in var_rec if i['custom_variable_name'] is not None and any(item.lower() in target_headers for item in i['possible_headers'])==True]
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
                X = np.vstack((X, custom_values['output'].T))
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
            return y, X.T, X_hdrs, 'None'
        else:
            return X.T, X_hdrs

def filter_variable_data(data, v_arg):
    '''
    This function takes a customized filter and applies to to the input data, basically limited the number
    of observations we want to consider FOR VARIABLE LEVEL FILTERS ONLY
    :param data: the input_data ids
    :param arg: the json dictionary used to build the argument
    :param filter_type: either 'block', in which case we assume arg is block-level information, or variable, in which case we assume it's just a variable filter
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
    data1_format_dict={'var_name':v_arg[CONFIG['data1_name']],'table_name':CONFIG['data1_table_name'],
                                                                                      'id_list':data1_ids}
    data2_format_dict={'var_name':v_arg[CONFIG['data2_name']],'table_name':CONFIG['data2_table_name'],
                                                                                      'id_list':data2_ids}
    data1_values = pd.DataFrame(get_table_noconn(
        '''select cast(id as text), {var_name} as data1_target from {table_name} where id in ({id_list})'''.format(**data1_format_dict), db))
    data2_values = pd.DataFrame(get_table_noconn(
        '''select cast(id as text), {var_name} as data2_target from {table_name} where id in ({id_list})'''.format(**data2_format_dict), db))
    ###merge the three dataframes
    data = data.merge(data1_values, left_on='{}_id'.format(CONFIG['data1_name']), right_on='id')
    data = data.merge(data2_values, left_on='{}_id'.format(CONFIG['data2_name']), right_on='id')
    ###identify the function we need to run:
    if v_arg['match_type']=='fuzzy':
       target_fun=[i for i in methods if i.__name__==v_arg['filter_only']['fuzzy_name']][0]
       data['score'] = data.apply(lambda x: target_fun(x['data1_target'].lower(), x['data2_target'].lower()) if x['data1_target']!='NULL' and x['data2_target']!='NULL'
                                                     else 0, axis=1)
    ###Now the exact matches##
    elif v_arg['match_type']=='exact':
        data['score'] = np.where(data['data1_target']==data['data2_target'], 1, 0)
        v_arg['filter_value'] = 1
    elif v_arg['match_type']=='num_distance':
        data['score'] = data['data1_target'] - data['data2_target']
    elif v_arg['match_type']=='geo_distance':
        data['score'] = data.apply(lambda x: haversine((x['data1_latitude'], x['data1_longitude']),(x['data2_latitude'],x['data2_longitude'])), axis=1)
    elif v_arg['match_type']=='date':
        data['score'] = data.apply(lambda x: feb.editdist(x['data1_target'], x['data2_target']), axis=1)
    elif v_arg['match_type']=='custom':
        ####get the right arrangement of custom functions and targets
        ###get the corresponding function attached to the var list
        my_function = [{'name':k[0], 'function':k[1]} for k in getmembers(cust_scoring) if k[0].lower()==v_arg['custom_variable_name'].lower()][0]
        data['score'] = data.apply(lambda x: my_function['function']((x['data1_target'],x['data2_target'])), axis=1)
    ####now return the columns needed with just the rows that meet the criteria
    qry = '''score {} {}'''.format( v_arg['filter_only']['test'],v_arg['filter_only']['value'])
    data=data.query(qry)
    if len(data) > 0:
        return data[['index','{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]]
    else:
        return pd.DataFrame(columns=['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])])

def apply_filter_variables(func_data, variable_list):
    '''
    This is the wrapper function for filter_variable data
    :param data:
    :param variable_list:
    :return:
    '''

    for v in variable_list:
        if len(func_data) > 0:
            if ast.literal_eval(CONFIG['chatty_logger'])==True:
                logger.info('Using variable filtering for variable {}'.format(v['variable_name']))
            func_data = filter_variable_data(func_data, v)
    if len(func_data) > 0:
        return func_data[['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]]
    else:
        return pd.DataFrame(
            columns=['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])])

def filter_block_data(data, arg):
    '''
    This function takes a customized filter and applies to to the input data, basically limited the number
    of observations we want to consider
    :param data: the input_data ids
    :param arg: the json dictionary used to build the argument
    :param filter_type: either 'block', in which case we assume arg is block-level information, or variable, in which case we assume it's just a variable filter
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
        ###see if we need to find anything for data 1
        if CONFIG['data1_name'] in arg['block_info']['variable_filter_info']:
            data1_format_dict={'var_name':arg['block_info']['variable_filter_info'][CONFIG['data1_name']],'table_name':CONFIG['data1_table_name'],'id_list':data1_ids}
        else:
            data1_format_dict={'var_name':'1','table_name':CONFIG['data1_table_name'],'id_list':data1_ids}
        ###ditto for table 2
        if CONFIG['data2_name'] in arg['block_info']['variable_filter_info']:
            data2_format_dict = {'var_name': arg['block_info']['variable_filter_info'][CONFIG['data2_name']],
                                 'table_name': CONFIG['data2_table_name'], 'id_list': data2_ids}
        else:
            data2_format_dict = {'var_name': '1', 'table_name': CONFIG['data2_table_name'], 'id_list': data2_ids}
        data1_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as text), {var_name} as data1_target from {table_name} where id in ({id_list})'''.format(**data1_format_dict), db))
        data2_values = pd.DataFrame(get_table_noconn(
            '''select cast(id as text), {var_name} as data2_target from {table_name} where id in ({id_list})'''.format(**data2_format_dict), db))

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
       data['score'] = data.apply(lambda x: target_fun(x['data1_target'].lower(), x['data2_target'].lower()) if x['data1_target']!='NULL' and x['data2_target']!='NULL'
                                                     else 0, axis=1)
    ###Now the exact matches##
    elif arg['block_info']['variable_filter_info']['match_type']=='exact':
        data['score'] = np.where(data['data1_target']==data['data2_target'], 1, 0)
        arg['block_info']['variable_filter_info']['filter_value'] = 1
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
        data['score'] = data.apply(lambda x: my_function['function'](x), axis=1)
    ####now return the columns needed with just the rows that meet the criteria
    qry = '''score {} {}'''.format( arg['block_info']['variable_filter_info']['test'],arg['block_info']['variable_filter_info']['filter_value'])
    data=data.query(qry)
    if len(data) > 0:
        return data[['index','{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]]
    else:
        return pd.DataFrame(columns=['index', '{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])])

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
        input_qry = """select cast(a.id as text) {data1_name}_id, cast(b.id as text) {data2_name}_id from
                (select id, {data1_block_var_list}{data1_selection_statement_variable} from {data1_name} where {data1_block_target_statement} {data1_selection_statement}) a
                inner join
                (select id, {data2_block_var_list}{data2_selection_statement_variable} from {data2_name} where {data2_block_target_statement} {data2_selection_statement}) b
                {block_merge_statement}
                {filter_statement}"""
        ##first, get the correct block statements
        if isinstance(arg['target'], dict):
            data1_block_target_statement = ' and '.join([''' {} = '{}' '''.format(key, arg['target'][key]) for key in arg['target'].keys()])
            data2_block_target_statement = ' and '.join([''' {} = '{}' '''.format(key, arg['target'][key]) for key in arg['target'].keys()])
            ###now the block var list
            if isinstance(arg['block_info'][CONFIG['data1_name']], str):
                data1_block_var_list = arg['block_info'][CONFIG['data1_name']]
                data2_block_var_list = arg['block_info'][CONFIG['data2_name']]
            else:
                data1_block_var_list = ','.join(arg['block_info'][CONFIG['data1_name']])
                data2_block_var_list = ','.join(arg['block_info'][CONFIG['data2_name']])
            ##finally the block merge statement
            block_merge_statement = 'on '+' and '.join(['a.{} = b.{}'.format(x,x) for x in arg['target'].keys()])
        else:
            data1_block_target_statement = '''{} = '{}' '''.format(arg['block_info'][CONFIG['data1_name']], arg['target'])
            data2_block_target_statement = '''{} = '{}' '''.format(arg['block_info'][CONFIG['data2_name']], arg['target'])
            data1_block_var_list = '{} as block'.format(arg['block_info'][CONFIG['data1_name']])
            data2_block_var_list = '{} as block'.format(arg['block_info'][CONFIG['data2_name']])
            block_merge_statement = 'on a.block=b.block'
        ##if we aren't doing a chunked block
        if arg['block_info']['chunk_size']==-1:
            if CONFIG['custom_selection_statement'] == 'False':
                input_qry_args = {'data1_name': CONFIG['data1_name'], 'data2_name': CONFIG['data2_name'],
                                  'data1_block_target_statement': data1_block_target_statement,
                                  'data2_block_target_statement': data2_block_target_statement,
                                  'data1_block_var_list':data1_block_var_list,
                                  'data2_block_var_list': data2_block_var_list,
                                  'block_merge_statement': block_merge_statement,
                                  'data1_selection_statement': '','data2_selection_statement': ''}
            else:
                if CONFIG['data1_name'] in CONFIG['custom_selection_statement'].keys():
                    data1_selection_statement = 'and ' + CONFIG['custom_selection_statement'][CONFIG['data1_name']]
                else:
                    data1_selection_statement = ''
                if CONFIG['data2_name'] in CONFIG['custom_selection_statement'].keys():
                    data2_selection_statement = 'and ' + CONFIG['custom_selection_statement'][CONFIG['data2_name']]
                else:
                    data2_selection_statement = ''
                input_qry_args = {'data1_name': CONFIG['data1_name'], 'data2_name': CONFIG['data2_name'],
                                  'data1_block_target_statement': data1_block_target_statement,
                                  'data2_block_target_statement': data2_block_target_statement,
                                    'data1_block_var_list':data1_block_var_list,
                                    'data2_block_var_list': data2_block_var_list,
                                  'block_merge_statement': block_merge_statement,
                                  'data1_selection_statement': data1_selection_statement,'data2_selection_statement': data2_selection_statement}
        else:
            ###if we are chunking out the block, this identifies the block using the appropriate target element (remember, arg['target'] in this case is a tuple)
            if ast.literal_eval(CONFIG['custom_selection_statement']) == False:
                input_qry_args = {'data1_name': CONFIG['data1_name'], 'data2_name': CONFIG['data2_name'],
                                  'data1_block_target_statement': data1_block_target_statement,
                                  'data2_block_target_statement': data2_block_target_statement,
                                    'data1_block_var_list':data1_block_var_list,
                                    'data2_block_var_list': data2_block_var_list,
                                  'block_merge_statement': block_merge_statement,
                                  'data1_selection_statement': '','data2_selection_statement': ''}
            else:
                if CONFIG['data1_name'] in CONFIG['custom_selection_statement'].keys():
                    data1_selection_statement = 'and ' + CONFIG['custom_selection_statement'][CONFIG['data1_name']]
                else:
                    data1_selection_statement = ''
                if CONFIG['data2_name'] in CONFIG['custom_selection_statement'].keys():
                    data2_selection_statement = 'and ' + CONFIG['custom_selection_statement'][CONFIG['data2_name']]
                else:
                    data2_selection_statement = ''
                input_qry_args = {'data1_name': CONFIG['data1_name'], 'data2_name': CONFIG['data2_name'],
                                  'data1_block_target_statement': data1_block_target_statement,
                                  'data2_block_target_statement': data2_block_target_statement,
                                  'data1_block_var_list':data1_block_var_list,
                                'data2_block_var_list': data2_block_var_list,
                                  'block_merge_statement': block_merge_statement,
                                  'data1_selection_statement': data1_selection_statement,'data2_selection_statement': data2_selection_statement}
        ###Now fix the filter statement
        ###If we are in deduplication mode and we have a global filter statement:
        if CONFIG['mode']=='deduplication' and CONFIG['global_filter_statement'] != 'False':
            input_qry_args['filter_statement'] = 'where a.id < b.id and {}'.format(CONFIG['global_filter_statement'])
        ###if we are in deduplication mode and we do not have a global filter statement
        elif CONFIG['mode']=='deduplication' and CONFIG['global_filter_statement']=='False':
            input_qry_args['filter_statement'] = 'where a.id < b.id'
        ###If we are ignoring duplicate IDs but aren't in deduplication mode and have a global filter
        elif ast.literal_eval(CONFIG['ignore_duplicate_ids'])==True and CONFIG['mode']!='deduplication' and CONFIG['global_filter_statement'] != 'False':
            ###So if we are ignoring duplicate IDs
            input_qry_args['filter_statement'] = 'where a.id!=b.id and {}'.format(CONFIG['global_filter_statement'])
        ###if we are ignorning duplciate IDS but aren't in deduplication mode but DO NOT have a global filter
        elif ast.literal_eval(CONFIG['ignore_duplicate_ids'])==True and CONFIG['mode']!='deduplication' and ast.literal_eval(CONFIG['global_filter_statement'])==False:
            ###So if we are ignoring duplicate IDs
            input_qry_args['filter_statement'] = 'where a.id!=b.id'
        ###So in all other circumstances this is its own thing
        else:
            if CONFIG['global_filter_statement'] != 'False':
                input_qry_args['filter_statement'] = 'where {}'.format(CONFIG['global_filter_statement'])
            else:
                ###otherwise a blank statement
                input_qry_args['filter_statement'] = ''
        ###Ok.  Now add in the variable filter args if present
        if arg['block_info']['variable_filter_info']!=-1:
            ###this means we are using a variable filter
            ### possible cases:
            # 1.  using a fuzzy or geometric filter, so we ignore
            if arg['block_info']['variable_filter_info']['match_type'] not in ['exact','custom_selection']:
                input_qry_args['data1_selection_statement_variable'] = ''
                input_qry_args['data2_selection_statement_variable'] = ''
                ###we don't make any change to the filter statement
            # 2.  using json to define variables that get added
            else:
                input_qry_args['data1_selection_statement_variable'] = ',{} as a_filter_target'.format(arg['block_info']['variable_filter_info'][CONFIG['data1_name']])
                input_qry_args['data2_selection_statement_variable'] = ',{} as b_filter_target'.format(arg['block_info']['variable_filter_info'][CONFIG['data2_name']])
                ###so here is where we will add in the test statement.
                ###RULE: if the test is something other than the list, then we need to use the custom test statement.
                if arg['block_info']['variable_filter_info']['test']=='custom':
                    filter_statement = arg['block_info']['variable_filter_info']['custom_test']
                else:
                    filter_statement = 'a_filter_target {} b_filter_target'.format(arg['block_info']['variable_filter_info']['test']).replace("==","=")
                if ast.literal_eval(CONFIG['ignore_duplicate_ids']) == True:
                    input_qry_args['filter_statement'] = input_qry_args['filter_statement'] + ' and ' + filter_statement
                else:
                    input_qry_args['filter_statement'] = 'where '+filter_statement
        else:
            input_qry_args['data1_selection_statement_variable'] = ''
            input_qry_args['data2_selection_statement_variable'] = ''
        ###finally, if in deduplication mode, replace data1 and data2 names with target table
        if CONFIG['mode']=='deduplication':
            input_qry = input_qry.replace("from {data1_name}", "from {}".format(CONFIG['target_table']))
            input_qry = input_qry.replace("from {data2_name}", "from {}".format(CONFIG['target_table']))
        ###retrieve the data
        input_data = pd.DataFrame(get_table_noconn(input_qry.format(**input_qry_args), db))
        ####Now remove ANY rows that have already been matched for this batch if we are doing the second block
        ####Also only done if we have any records to match
        if int(arg['block_info']['order']) > 1 and len(input_data) > 0:
            ###different queries if we are dealing with
            matched_data = pd.DataFrame(get_table_noconn('''select {}_id::text, {}_id::text from {} 
                            where batch_id={} '''.format(CONFIG['data1_name'], CONFIG['data2_name'],
                                                         CONFIG['matched_pairs_table_name'], CONFIG['batch_id']), db))
            ###merge, keep only the input records NOT in matched_data if this is the second block
            if len(matched_data) > 0 and int(arg['block_info']['order']) > 1:
                input_data = input_data.merge(matched_data[['{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]],
                                              on=['{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])],
                                              indicator=True, how='outer')
                input_data = input_data.loc[input_data._merge=='left_only']
                input_data.drop('_merge', axis=1, inplace=True)
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
                input_data = filter_block_data(input_data, arg)
            if type(input_data)==str == True:
                if input_data=='fail':
                    return 'fail'
            elif len(input_data)==0 and type(input_data)!=str:
                end = time.time() - start
                logger.info('After block-level filtering, there were no valid matches to attempt for block {}'.format(arg['target']))
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
                ####Now do the variable only filters
                block_orig_len = len(input_data)
                if len([v for v in var_types if v['filter_only']!={}]) > 0:
                    input_data = apply_filter_variables(input_data, [v for v in var_types if v['filter_only']!={}])
                if len(input_data) == 0 and type(input_data) != str:
                    end = time.time() - start
                    logger.info(
                        'After variable filtering, there were no valid matches to attempt for block {}'.format(arg['target']))
                    stats_dat = {'batch_id': arg['batch_id'],
                                 'block_level': arg['block_info']['block_name'],
                                 'block_id': str(arg['target']),
                                 'block_time': end,
                                 'block_size': 0,
                                 'block_matches': 0,
                                 'block_matches_avg_score': 0,
                                 'block_non_matches': 0,
                                 'block_non_matches_avg_score': 0,
                                 'match_pairs_removed_filter': block_orig_len - len(input_data)}
                    stats_out = write_to_db(stats_dat, 'batch_statistics')
                    if stats_out:
                        logger.info('Unable to write batch statistics for batch {}, continuing'.format(arg['target']))
                else:
                    logger.info('Creating Scores for block {}'.format(arg['target']))
                    X, X_hdrs = create_all_scores(input_data, 'prediction', arg['X_hdrs'])
                    ####Now write to the DB
                    if len(input_data) > 0:
                        ###need to reset the index after all that filtering.
                        input_data.reset_index(inplace=True, drop=True)
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
                            ###first, if we don't have prediction then our data will already be a dataframe.  This just confirms this
                            if isinstance(X, pd.DataFrame) == False:
                                X = pd.DataFrame(X, columns=X_hdrs)
                                X['predicted_probability'] = input_data['predicted_probability']
                            ####Now add in any matches
                            #####If the predicted probability is a string (so someone returned a JSON)
                            if input_data['predicted_probability'].dtype == 'O':
                                ####So we have an object that isn't meant to be filtered. Log it
                                ####Return predicted probability
                                input_data['match_info'] = input_data['predicted_probability']
                                input_data['predicted_probability'] = input_data.match_info.str['predicted_probability'].astype(float)
                            ##Now to the match thresholds
                            stats_dat = copy.deepcopy(input_data)
                            matches = input_data[input_data['predicted_probability'] >= float(CONFIG['match_threshold'])]
                            matches.sort_values('predicted_probability', ascending=False, inplace=True)
                            if len(matches) > 0:
                                ###ranks
                                #####Note here that if you have an object, this will rank them alphabetically. Feel free to ignore them!
                                matches['{}_rank'.format(CONFIG['data1_name'])] = matches.groupby('{}_id'.format(CONFIG['data1_name']))['predicted_probability'].rank(method='dense', ascending=False)
                                matches['{}_rank'.format(CONFIG['data2_name'])] = matches.groupby('{}_id'.format(CONFIG['data2_name']))['predicted_probability'].rank(method='dense', ascending=False)
                                ###add the batch_id
                                matches['batch_id']=arg['batch_id']
                                ###add the block level into the json
                                matches['match_pair_info'] = json.dumps({'block_name':arg['block_info']['block_name'], 'block_value':arg['target']})
                                ###convert the IDs into integers
                                for col in ['left_id','right_id','{}_id'.format(CONFIG['data1_name']), '{}_id'.format(CONFIG['data2_name'])]:
                                    if col in matches.columns:
                                        matches[col] = matches[col].astype(int)
                                ##convert back to dict
                                matches = matches.to_dict('records')
                                ###write to DB
                                write_out = write_to_db(matches, CONFIG['matched_pairs_table_name'])
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
                            ###Two flavors of clerical review candidates. 1) with multiple conditions or 2) with a single condition
                            ###sure I could make 2) an element of 1) here but I don't want to break peoples pre-defined working properties files.
                            ##start by declaring an empty list
                            clerical_candidates = []
                            ####now use pandas sql to get the candidates
                            if isinstance(X, pd.DataFrame) == False:
                                X = pd.DataFrame(X, columns=X_hdrs)
                            if ast.literal_eval(CONFIG['prediction'])==True:
                                ####assign the predicted probabilyt for the records in case that is how we are picking candidates
                                X['predicted_probability'] = input_data['predicted_probability']
                            if 'listed_variable_names' in CONFIG['clerical_review_threshold'].keys():
                                ###first get the statements
                                sql_qry = ' {} '.format(CONFIG['clerical_review_threshold']['query_logic']).join(['{} >= {}'.format(k['variable'], k['value']) for k in CONFIG['clerical_review_threshold']['listed_variable_names']])
                                threshold_values = X.query(sql_qry)
                                if len(threshold_values) > 0:
                                    threshold_values['threshold_values'] = threshold_values.apply(lambda x: json.dumps({k['variable']:x[k['variable']] for k in CONFIG['clerical_review_threshold']['listed_variable_names']}), 1)
                                    clerical_candidates = input_data[input_data.index.isin(threshold_values.index.tolist())]
                                    clerical_candidates['threshold_values'] = threshold_values.apply(lambda x: json.dumps({k['variable']:x[k['variable']] for k in CONFIG['clerical_review_threshold']['listed_variable_names']}), 1)
                            else:
                                clerical_values = np.where(input_data[CONFIG['clerical_review_threshold']['variable']] >= float(CONFIG['clerical_review_threshold']['value']), True, False)
                                clerical_candidates = input_data[(clerical_values==True)]
                                if len(clerical_candidates) > 0:
                                    clerical_candidates['threshold_values'] = clerical_candidates.apply(lambda x: json.dumps({CONFIG['clerical_review_threshold']['variable']:x[CONFIG['clerical_review_threshold']['variable']]}),1)
                            ###add in the batch id to each row
                            ###Only do the following if we have clerical candidates
                            if len(clerical_candidates) > 0:
                                if ast.literal_eval(CONFIG['chatty_logger']) == True:
                                    logger.info('Before sampling, have {} records for block {}'.format(len(clerical_candidates), arg['target']))
                                clerical_candidates['batch_id']=arg['batch_id']
                                ###if there are more than 10, select 10% of them
                                if len(clerical_candidates) >= 10:
                                    clerical_review_dict = dcpy(random.sample(clerical_candidates.to_dict('records'),int(np.round(.1*len(clerical_candidates),0))))
                                else:
                                    clerical_review_dict = dcpy(clerical_candidates.to_dict('records'))
                                if len(clerical_review_dict) > 0:
                                    if ast.literal_eval(CONFIG['chatty_logger']) == True:
                                        logger.info('Writing {} candidates for block {}'.format(len(clerical_review_dict), arg['target']))
                                    if len([k for k in clerical_review_dict if k['{}_id'.format(CONFIG['data1_name'])] is None])>0:
                                        logger.info('Block {} may have an issue'.format(arg['block']['target']))
                                    clerical_out = write_to_db(clerical_review_dict, CONFIG['clerical_review_candidates_table_name'])
                                    if clerical_out:
                                        to_return['clerical_review_candidates']=clerical_out
                                ###if it's a chatty logger, log.
                                if ast.literal_eval(CONFIG['chatty_logger']) == True:
                                    logger.info('''{} clerical review candidates added from block {}'''.format(len(clerical_review_dict), arg['target']))
                            else:
                                if ast.literal_eval(CONFIG['chatty_logger']) == True:
                                    logger.info('''Block {} did not have any eligible clerical review candidates'''.format(arg['target']))
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
                                                   'block_matches_avg_score': np.where(len(stats_dat) > 0,np.nanmean(np.where(stats_dat['predicted_probability'] >= float(CONFIG['match_threshold']),stats_dat['predicted_probability'],np.nan)), np.nan).tolist(),
                                                   'block_non_matches': int(np.sum(np.where(stats_dat['predicted_probability'] < float(CONFIG['match_threshold']),1,0))),
                                                   'block_non_matches_avg_score': np.where(len(stats_dat) > 0,np.nanmean(np.where(stats_dat['predicted_probability'] < float(CONFIG['match_threshold']),stats_dat['predicted_probability'],np.nan)), np.nan).tolist(),
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
def run_block(block, model, batch_id, batch_summary):
    '''
    This function creates the list of blocks to use for comparison and then runs the matches
    :param block: the dict of the block we are running
    :param model: the model we are using
    :param batch_id: for whatever reason, the CONFIG['batch_id'] call is inconsistent, so putting that directly here
    :return:
    '''
    ####If we are running the 'full' block
    db = get_db_connection(CONFIG)
    ###if we have a debug or debug block list, we can skip the DB call
    if CONFIG.get('debug_block', None) is not None:
        if len(blocks) > 1:
            logger.warning("You have specified a single debug block but have multiple blocking passes.  The debug block may not work correctly.  Please use 'debug_block_list' and add a block_order for your debug blocks.")
        block_list = [CONFIG['debug_block']]
    elif CONFIG.get('debug_block_list',None) is not None:
        block_list = [k for k in CONFIG['debug_block_list'] if 'block_order' in k.keys() and k['block_order'] == block['order']]
        if len(block_list) == 0:
            '''This would happen when the block order is not specified'''
            logger.warning('No debug blocks identified for block {}, exiting.  Make sure you have specified block_order correctly.'.format(block['block_name']))
            batch_summary['batch_status'] = 'failed'
            batch_summary['batch_completed'] = dt.datetime.now()
            batch_summary['failure_message'] = 'No debug blocks identified for block {}, exiting.  Make sure you have specified block_order correctly.'.format(block['block_name'])
            update_batch_summary(batch_summary)
            db.close()
            os._exit(0)
        ###remove block order
        for b in block_list:
            del b['block_order']
    else:
        data1_selection_statement = ''
        data2_selection_statement = ''
        # global filer statements
        if CONFIG['custom_selection_statement']!='False':
            if CONFIG['data1_name'] in CONFIG['custom_selection_statement'].keys():
                data1_selection_statement = 'where {}'.format(CONFIG['custom_selection_statement'][CONFIG['data1_name']])
            if CONFIG['data2_name'] in CONFIG['custom_selection_statement'].keys():
                data2_selection_statement = 'where {}'.format(CONFIG['custom_selection_statement'][CONFIG['data2_name']])
        # ###now get the blocks
        ###First, check if we are splitting a list and create the block target statement
        if isinstance(block[CONFIG['data1_name']],list):
            data1_target_statement = ''','''.join([v for v in block[CONFIG['data1_name']]])
            data1_blocks = get_table_noconn('''select distinct {} from {} {}'''.format(data1_target_statement, CONFIG['data1_table_name'],data1_selection_statement), db)
            ##now data2
            data2_target_statement = ''','''.join([v for v in block[CONFIG['data2_name']]])
            data2_blocks = get_table_noconn('''select distinct {} from {} {}'''.format(data2_target_statement, CONFIG['data2_table_name'],data2_selection_statement), db)
        else:
            ##otherwise we are dealing with as ingle variable, so just proceed as normal
            data1_target_statement = block[CONFIG['data1_name']]
            data1_blocks = get_table_noconn('''select distinct {} as block from {} {}'''.format(data1_target_statement, CONFIG['data1_table_name'],data1_selection_statement), db)
            data1_blocks = [i['block'] for i in data1_blocks]
            ##now data 2
            data2_target_statement = block[CONFIG['data2_name']]
            data2_blocks = get_table_noconn('''select distinct {} as block from {} {}'''.format(data2_target_statement, CONFIG['data2_table_name'],data2_selection_statement), db)
            data2_blocks = [i['block'] for i in data2_blocks]
        ### the block list is the intersection (ie all the blocks that match)
        logger.info('Have Blocks.  Getting intersection list')
        ###so here we aren't using the debug blocks
        if block['chunk_size']==-1:
            if isinstance(block[CONFIG['data1_name']], list):
                block_list=pd.DataFrame(data1_blocks).merge(pd.DataFrame(data2_blocks), left_on=block[CONFIG['data1_name']], right_on=block[CONFIG['data2_name']], how='inner').to_dict('records')
            else:
                block_list = intersection(data1_blocks,data2_blocks)
        ###otherwise we are using no block and it's just the combination.
        else:
            block_list = [(x, y) for x in data1_blocks for y in data2_blocks]
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
    pool.close()
    ###check for any failures
    for o in range(len(out)):
        if out[0]=='fail':
            logger.info('Failed for {}, block {}, see log for details'.format(block['block_name'], arg_list[o]['target']))
            os._exit(0)
    ###Once that is done, need to
        ##push the remaining items in out to the db
    try:
        db=get_db_connection(CONFIG)
        cur=db.cursor()
        logger.info('Dumping remaining matches to DB for block {}'.format(block['block_name']))
        for i in out:
            if i!=None and i.__class__==dict:
                if 'clerical_review_candidates' in i.keys():
                    write_to_db(i['clerical_review_candidates'],CONFIG['clerical_review_candidates_table_name'])
                if 'matches' in i.keys() and i.__class__==dict:
                    write_to_db(i['matches'], CONFIG['matched_pairs_table_name'])
        db.commit()
        logger.info('Remaining matches and candidates dumped')
    except Exception as error:
        logger.info('Error Dumping Remaining Matches for block {}. Error {}'.format(block, error))
    ##then change the matched flags on the data tables to 1 where it's been matched
    ###updating the matched flags
    if CONFIG.get('debug_block', None) is not None or CONFIG.get('debug_block_list', None) is not None:
        logger.info('This was just a debug run, so we are done.')
        ##commit and close the DB
        db.commit()
        db.close()
        logger.info('Block {} Complete'.format(block['block_name']))
    else:
        # if ast.literal_eval(CONFIG['prediction'])==True:
        #     if CONFIG['mode']!='deduplication':
        #         cur.execute('''update {data1_name} set matched=1 where id in (select distinct {data1_name}_id from matched_pairs)'''.format(data1_name=CONFIG['data1_name']))
        #         cur.execute('''update {data2_name} set matched=1 where id in (select distinct {data2_name}_id from matched_pairs)'''.format(data2_name=CONFIG['data2_name']))
        #     else:
        #         cur.execute(
        #             '''update {table_name} set matched=1 where id in (select distinct right_id from matched_pairs)'''.format(
        #                 table_name=CONFIG['target_table']))
        #         cur.execute(
        #             '''update {table_name} set matched=1 where id in (select distinct left_id from matched_pairs)'''.format(
        #                 table_name=CONFIG['target_table']))
        ##commit and close the DB
        db.commit()
        db.close()
        logger.info('Block {} Complete'.format(block['block_name']))

if __name__=='__main__':
    print('why did you do this?')
    os._exit(0)
    
    