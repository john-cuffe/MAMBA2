###########
###CHUNK: THE RANDOM FOREST FUNCTIONS
############
from __future__ import division
import os
from programs.global_vars import *
from copy import deepcopy as dcpy
import copy
from programs.logger_setup import *
logger=logger_setup(os.environ['log_file_name'])

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

def featSelRFImpurity(y,X,hdr,k,nT, nE,mD):
    """
    Description: Perform random forest decreased impurity feature selection
    Parameters:
        y - array of 1/0 truth for UM
        X - array of features
        hdr - column names for features in X
        k - number of features to select
        nE - nubmer of estimators
        mD - max depth
    Returns: cols, indexed column positions for features selected
    """
    tfunc = time.time()
    #print('#'*10+str(round((time.time() - t0)/60, 2))+' Minutes: Feature Selection (Impurity Method)'+'#'*10)
    # fit requires 1-d array not column vector
    y = y.ravel()
    # fit feature selection model
    model= RandomForestClassifier(n_estimators=nT,max_features=nE,max_depth=mD)
    model.fit(X, y)
    # extract column scores
    rndscores=[round(i, 2) for i in model.feature_importances_]
    colScore = sorted(zip(range(X.shape[1]),rndscores),key=lambda tup: tup[1],reverse=True)[0:k]
    # scores with headers
    hdrScore = sorted(zip(hdr,rndscores),key=lambda tup: tup[1],reverse=True)[0:k]
    # vector of columns selected
    cols=[x[0] for x in colScore]
    #print('Top +'10 Features: {0}'.format(hdrScore))
    #print('Selected Features: {0}'.format(hdrScore))
    #print('Function Time (Min): {0}'.format((time.time()-tfunc)/60))
    return hdrScore

###Run the Predictions
def predict(model, X):
    """
    Description: Executes predict function for a giinput model and set of features
    Parameters:
        model - model object of classifier (e.g. RandomForestClassifier)
        X - array of features
    Returns: predicted, array of 1/0 predictions
    """
    tfunc = time.time()
    predicted = model.predict(X)
    return predicted


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


def create_scores(input_data, score_type, varlist):
    '''
    this function produces the list of dictionaries for all of the scores for the fuzzy variables.
    each dict will have two arguments: the array and a list of names
    :param: db : the database connection
    :param input_data: Either a block to query the database with OR a dataframe of variable types
    :param fuzzy_vars: a list of dictionaries of fuzzy variables to target
    :param model_training: Are we training a model or running an acutal match (False)
    NOTE FOR FUTURE SELF: IN NORMAL FUNCTIONS (MODEL_TRAINING=FALSE) THIS WILL ALREADY BE RUN IN A SUB-PROCESS
    SO YOU HAVE TO SINGLE-THREAD THE CALCULATIONS, SO THE WHOLE THING IS SINGLE-THREADED
    return: an array of values for each pair to match/variable and a list of headers
    '''
    db=get_connection_sqlite(os.environ['db_name'])
    if score_type=='fuzzy':
        data1_ids=','.join(str(v) for v in input_data['{}_id'.format(os.environ['data1_name'])].drop_duplicates().tolist())
        data2_ids=','.join(str(v) for v in input_data['{}_id'.format(os.environ['data2_name'])].drop_duplicates().tolist())
        ###create an indexed list of the id pairs to serve as the core of our dictionary
        input_data=dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict=input_data[['index','{}_id'.format(os.environ['data1_name']),'{}_id'.format(os.environ['data2_name'])]].to_dict('record')
        ###now get the values from the database
        data1_names=','.join(['{} as {}'.format(i[os.environ['data1_name']], i['variable_name']) for i in varlist])
        data2_names=','.join(['{} as {}'.format(i[os.environ['data2_name']], i['variable_name']) for i in varlist])
        ###get the values
        data1_values=get_table_noconn('''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names, table_name=os.environ['data1_name'], id_list=data1_ids), db)
        data2_values=get_table_noconn('''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names, table_name=os.environ['data2_name'], id_list=data2_ids), db)
        ###give the data values the name for each searching
        data1_values = {str(item['id']):item for item in data1_values}
        data2_values = {str(item['id']):item for item in data2_values}
        ####now for each pair, get the value for each fuzzy variable
        fuzzy_var_list=[i['variable_name'] for i in varlist]
        ##convert fuzzy vars to a list
        ###now create the dictionary with the list of names and the array
        out_arr = np.zeros(shape=(len(core_dict), len(fuzzy_var_list)*len(methods)))
        for i in range(len(core_dict)):
            i_scores=[]
            for j in range(len(fuzzy_var_list)):
                for k in methods:
                    ####Two null match methods: either they get a 0 or the median value for the chunk
                    try:
                        i_scores.append(k(data1_values[core_dict[i]['{}_id'.format(os.environ['data1_name'])]][fuzzy_var_list[j]], data2_values[core_dict[i]['{}_id'.format(os.environ['data2_name'])]][fuzzy_var_list[j]]))
                    except:
                        if os.environ['null_match_method']=='zero':
                            i_scores.append(0)
                        ###other method can go here if we think of one
            out_arr[i,]=i_scores
        ####Now return a dictionary of the input array and the names
        return {'output':out_arr, 'names':['{}_{}'.format(i,j.__name__) for i in fuzzy_var_list for j in methods]}
    elif score_type=='numeric_dist':
        ###numeric distance variables
        data1_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(os.environ['data1_name'])].drop_duplicates().to_list())
        data2_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(os.environ['data2_name'])].drop_duplicates().to_list())
        ###now get the values from the database
        data1_names = ','.join(['{} as {}'.format(i[os.environ['data1_name']], i['variable_name']) for i in varlist])
        data2_names = ','.join(['{} as {}'.format(i[os.environ['data2_name']], i['variable_name']) for i in varlist])
        ###get the values
        data1_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names,table_name=os.environ['data1_name'],id_list=data1_ids), db)
        data2_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names,table_name=os.environ['data2_name'],id_list=data2_ids), db)
        ###give the data values the name for each searching
        data1_values = {str(item['id']): item for item in data1_values}
        data2_values = {str(item['id']): item for item in data2_values}
        ####now for each pair, get the value for each variable
        ###create an indexed list of the id pairs to serve as the core of our dictionary
        input_data = dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict = input_data[['index', '{}_id'.format(os.environ['data1_name']), '{}_id'.format(os.environ['data2_name'])]].to_dict('record')
        numeric_distance_vars = [i['variable_name'] for i in varlist]
        ##convert fuzzy vars to a list
        ###now create the dictionary with the list of names and the array
        out_arr = np.zeros(shape=(len(core_dict), len(numeric_distance_vars)))
        for i in range(len(core_dict)):
            i_scores = []
            for j in range(len(numeric_distance_vars)):
                if data1_values[core_dict[i]['{}_id'.format(os.environ['data1_name'])]][numeric_distance_vars[j]] and data2_values[core_dict[i]['{}_id'.format(os.environ['data2_name'])]][numeric_distance_vars[j]]:
                    i_scores.append(data1_values[core_dict[i]['{}_id'.format(os.environ['data1_name'])]][numeric_distance_vars[j]]-data2_values[core_dict[i]['{}_id'.format(os.environ['data2_name'])]][numeric_distance_vars[j]])
                else:
                    i_scores.append(-99999)
            out_arr[i,] = i_scores
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': numeric_distance_vars}
    elif score_type=='exact':
        ###If we are running a training data model
        data1_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(os.environ['data1_name'])].drop_duplicates().to_list())
        data2_ids = ','.join(
            str(v) for v in input_data['{}_id'.format(os.environ['data2_name'])].drop_duplicates().to_list())
        ###now get the values from the database
        data1_names = ','.join(['{} as {}'.format(i[os.environ['data1_name']], i['variable_name']) for i in varlist])
        data2_names = ','.join(['{} as {}'.format(i[os.environ['data2_name']], i['variable_name']) for i in varlist])
        ###get the values
        data1_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data1_names,table_name=os.environ['data1_name'],id_list=data1_ids), db)
        data2_values = get_table_noconn(
            '''select id, {names} from {table_name} where id in ({id_list})'''.format(names=data2_names,table_name=os.environ['data2_name'],id_list=data2_ids), db)
        ###give the data values the name for each searching
        data1_values = {str(item['id']): item for item in data1_values}
        data2_values = {str(item['id']): item for item in data2_values}
        ####now for each pair, get the value for each variable
        ###create an indexed list of the id pairs to serve as the core of our dictionary
        input_data = dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict = input_data[['index', '{}_id'.format(os.environ['data1_name']), '{}_id'.format(os.environ['data2_name'])]].to_dict('record')
        exact_vars = [i['variable_name'] for i in varlist]
        ##convert fuzzy vars to a list
        ###now create the dictionary with the list of names and the array
        out_arr = np.zeros(shape=(len(core_dict), len(exact_vars)))
        for i in range(len(core_dict)):
            i_scores = []
            for j in range(len(exact_vars)):
                if data1_values[core_dict[i]['{}_id'.format(os.environ['data1_name'])]][exact_vars[j]] and data2_values[core_dict[i]['{}_id'.format(os.environ['data2_name'])]][exact_vars[j]] and data1_values[core_dict[i]['{}_id'.format(os.environ['data1_name'])]][exact_vars[j]].upper()==data2_values[core_dict[i]['{}_id'.format(os.environ['data2_name'])]][exact_vars[j]].upper():
                    i_scores.append(1)
                else:
                    i_scores.append(0)
            out_arr[i,] = i_scores
        ####Now return a dictionary of the input array and the names
        return {'output': out_arr, 'names': exact_vars}
    elif score_type=='geo_distance':
        data1_ids = ','.join(str(v) for v in input_data['{}_id'.format(os.environ['data1_name'])].drop_duplicates().to_list())
        data2_ids = ','.join(str(v) for v in input_data['{}_id'.format(os.environ['data2_name'])].drop_duplicates().to_list())
        ###now get the values from the database
        ###get the values
        data1_values = get_table_noconn('''select id, latitude, longitude from {table_name} where id in ({id_list})'''.format(table_name=os.environ['data1_name'], id_list=data1_ids), db)
        data2_values = get_table_noconn('''select id, latitude, longitude from {table_name} where id in ({id_list})'''.format(table_name=os.environ['data2_name'],id_list=data2_ids), db)
        input_data = dcpy(input_data)
        input_data.reset_index(inplace=True, drop=False)
        core_dict = input_data[['index', '{}_id'.format(os.environ['data1_name']), '{}_id'.format(os.environ['data2_name'])]].to_dict('record')
        out_arr = np.zeros(shape=(len(core_dict), 1))
        null_count=0
        for i in range(len(core_dict)):
            data1_target=[k for k in data1_values if str(k['id'])==str(core_dict[i]['{}_id'.format(os.environ['data1_name'])])][0]
            data2_target=[k for k in data2_values if str(k['id'])==str(core_dict[i]['{}_id'.format(os.environ['data2_name'])])][0]
            if data1_target['latitude'] is not None and data1_target['longitude'] is not None and data2_target['latitude'] is not None and data2_target['longitude'] is not None:
                    out_arr[i]=haversine(tuple([data1_target['latitude'],data1_target['longitude']]),
                                  tuple([data2_target['latitude'], data2_target['longitude']]))
            else:
                null_count+=1
        if null_count==0:
            return {'output': out_arr, 'names': 'geo_distance'}
        else:
            logger.info('MAMBA detected missing lat/long coordinates.  Lat/Long distances will not be used')
            return {'output':'fail', 'names':'fail'}


def generate_rf_mod(truthdat):
    '''
    Generate the random forest model we are going to use for the matching
    inputs: truthdat: the truth data.  Just 3 columns--data1 id, data2 id, and if they match
    '''
    logger.info('\n\n######CREATE RANDOM FOREST MODEL ######\n\n')
    ###get the varible types
    var_rec=pd.read_csv('mamba_variable_types.csv').to_dict('record')
    ###We have three types of variables.
    #1) Fuzzy: Create all the fuzzy values from febrl
    fuzzy_vars=[i for i in var_rec if i['match_type']=='fuzzy']
    if len(fuzzy_vars) > 0:
        fuzzy_values=create_scores(truthdat, 'fuzzy', fuzzy_vars)
        X=fuzzy_values['output']
        X_hdrs=fuzzy_values['names']
    #2) num_distance: get the numeric distance between the two values, with a score of -9999 if missing
    numeric_dist_vars=[i for i in var_rec if i['match_type']=='num_distance']
    if len(numeric_dist_vars) > 0:
        numeric_dist_values=create_scores(truthdat, 'numeric_dist', numeric_dist_vars)
        X=np.hstack((X, numeric_dist_values['output']))
        X_hdrs.extend(numeric_dist_values['names'])
    #3) exact: if they match, 1, else 0
    exact_match_vars=[i for i in var_rec if i['match_type']=='exact']
    if len(exact_match_vars) > 0:
        exact_match_values=create_scores(truthdat, 'exact', exact_match_vars)
        X=np.hstack((X, exact_match_values['output']))
        X_hdrs.extend(exact_match_values['names'])
    geo_distance = [i for i in var_rec if i['match_type'] == 'geo_distance']
    if len(geo_distance) > 0:
        geo_distance_values = create_scores(truthdat, 'geo_distance', 'lat')
        if geo_distance_values['output']!='fail':
            X = np.hstack((X, geo_distance_values['output']))
            X_hdrs.extend(geo_distance_values['names'])
    ###making the dependent variable array
    y = truthdat['match'].values
    ###Generate the Grid Search to find the ideal values
    features_per_tree = ['sqrt', 'log2', 10, 15]
    rf = RandomForestClassifier(n_jobs=int(os.environ['rf_jobs']), max_depth=10, max_features='sqrt', n_estimators=10)
    myparams = {
        'n_estimators': sp_randint(1, 25),
        'max_features': features_per_tree,
        'max_depth': sp_randint(5, 25)}
    ##Run the Grid Search
    if debug:
        niter = 5
    else:
        niter = 200
    cv_rfc = RandomizedSearchCV(estimator=rf, param_distributions=myparams, cv=10, scoring=scoringcriteria,n_iter=niter)
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
    return {'type':'Random Forest', 'score':score, 'model':rf_mod}

def generate_ada_boost(truthdat):
    '''
    This function generates the adaboosted model
    '''
    logger.info('\n\n######CREATE ADABoost MODEL ######\n\n')
    from sklearn.ensemble import AdaBoostClassifier
    ###get the varible types
    var_rec = pd.read_csv('mamba_variable_types.csv').to_dict('record')
    ###We have three types of variables.
    # 1) Fuzzy: Create all the fuzzy values from febrl
    fuzzy_vars = [i for i in var_rec if i['match_type'] == 'fuzzy']
    if len(fuzzy_vars) > 0:
        fuzzy_values = create_scores(truthdat, 'fuzzy', fuzzy_vars)
        X = fuzzy_values['output']
        X_hdrs = fuzzy_values['names']
    # 2) num_distance: get the numeric distance between the two values, with a score of -9999 if missing
    numeric_dist_vars = [i for i in var_rec if i['match_type'] == 'num_distance']
    if len(numeric_dist_vars) > 0:
        numeric_dist_values = create_scores(truthdat, 'numeric_dist', numeric_dist_vars)
        X = np.hstack((X, numeric_dist_values['output']))
        X_hdrs.extend(numeric_dist_values['names'])
    # 3) exact: if they match, 1, else 0
    exact_match_vars = [i for i in var_rec if i['match_type'] == 'exact']
    if len(exact_match_vars) > 0:
        exact_match_values = create_scores(truthdat, 'exact', exact_match_vars)
        X = np.hstack((X, exact_match_values['output']))
        X_hdrs.extend(exact_match_values['names'])
    ###making the dependent variable array
    y = truthdat['match'].values
    ###Generate the Grid Search to find the ideal values
    ##setup the SVM
    ada = AdaBoostClassifier(
                         algorithm="SAMME",
                         n_estimators=200)
    if debug:
        niter = 5
    else:
        niter = int(np.round(len(X)/float(2),0))
    features_per_tree = ['sqrt', 'log2', 10, 15]
    myparams = {
        'n_estimators': sp_randint(1, 25),
        'algorithm':['SAMME','SAMME.R']}
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
    ###Note for when you return--you need to change the predict function to do cross_val_predict
    return {'type': 'AdaBoost', 'score': score, 'model': ada_mod}

def generate_svn_mod(truthdat):
    '''
    Generate the SVM model we are going to use for the matching
    inputs: truthdat: the truth data.  Just 3 columns--data1 id, data2 id, and if they match
    '''
    logger.info('\n\n######CREATE SVN MODEL ######\n\n')
    ###get the varible types
    var_rec=pd.read_csv('mamba_variable_types.csv').to_dict('record')
    ###We have three types of variables.
    #1) Fuzzy: Create all the fuzzy values from febrl
    fuzzy_vars=[i for i in var_rec if i['match_type']=='fuzzy']
    if len(fuzzy_vars) > 0:
        fuzzy_values=create_scores(truthdat, 'fuzzy', fuzzy_vars)
        X=fuzzy_values['output']
        X_hdrs=fuzzy_values['names']
    #2) num_distance: get the numeric distance between the two values, with a score of -9999 if missing
    numeric_dist_vars=[i for i in var_rec if i['match_type']=='num_distance']
    if len(numeric_dist_vars) > 0:
        numeric_dist_values=create_scores(truthdat, 'numeric_dist', numeric_dist_vars)
        X=np.hstack((X, numeric_dist_values['output']))
        X_hdrs.extend(numeric_dist_values['names'])
    #3) exact: if they match, 1, else 0
    exact_match_vars=[i for i in var_rec if i['match_type']=='exact']
    if len(exact_match_vars) > 0:
        exact_match_values=create_scores(truthdat, 'exact', exact_match_vars)
        X=np.hstack((X, exact_match_values['output']))
        X_hdrs.extend(exact_match_values['names'])
    ###making the dependent variable array
    y = truthdat['match'].values
    ###Generate the Grid Search to find the ideal values
    from sklearn import svm
    ##setup the SVM
    svc = svm.SVC(class_weight='balanced', gamma='scale')
    if debug:
        niter = 5
    else:
        niter = int(np.round(len(X)/float(2),0))
    from sklearn.model_selection import cross_val_score, cross_val_predict
    myparams={
        'kernel':['linear','poly','rbf'],
        'degree':[1,3,5],
        'gamma':['scale','auto']
    }
    ##First, get the scores
    svn_rfc = RandomizedSearchCV(estimator=svc, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                n_iter=niter)
    svn_rfc.fit(X, y)
    ##Save the parameters
    kernel = svn_rfc.best_params_['kernel']
    degree = svn_rfc.best_params_['degree']
    gamma = svn_rfc.best_params_['gamma']
    score = svn_rfc.best_score_
    logger.info('SVM Complete.  Max Score={}'.format(score))
    svc=svm.SVC(gamma=gamma, degree=degree, kernel=kernel)
    ###Note for when you return--you need to change the predict function to do cross_val_predict
    return {'type':'SVM', 'score':score, 'model':svc}


def choose_model(truthdat):
    '''
    This function generates a random forest, svn, and adaboost classifier for the training data, then returns the
    :param truthdat:
    :return:
    '''
    rf=generate_rf_mod(truthdat)
    svn=generate_svn_mod(truthdat)
    ada=generate_ada_boost(truthdat)
    ###find the max score
    best_score=max([i['score'] for i in [rf, svn, ada]])
    best_model=[i for i in [rf, svn,ada] if i['score']==i]
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
    if arg['logging_flag']:
        logger.info('{}% complete with block'.format(arg['logging_flag']))
    db=get_connection_sqlite(os.environ['db_name'], timeout=1)
    ###get the two dataframes
    data1=get_table_noconn('''select id from {} where {}='{}' and matched=0'''.format(os.environ['data1_name'], arg['block_info'][os.environ['data1_name']], arg['target']), db)
    data2=get_table_noconn('''select id from {} where {}='{}' and matched=0'''.format(os.environ['data2_name'], arg['block_info'][os.environ['data1_name']], arg['target']), db)
    ###get the data
    input_data=pd.DataFrame([{'{}_id'.format(os.environ['data1_name']):str(k['id']), '{}_id'.format(os.environ['data2_name']):str(y['id'])} for k in data1 for y in data2])
    ####Now get the data organized, by creating an array of the different types of variables
    # 1) Fuzzy: Create all the fuzzy values from febrl
    fuzzy_vars = [i for i in arg['var_rec'] if i['match_type'] == 'fuzzy']
    if len(fuzzy_vars) > 0:
        fuzzy_values = create_scores(input_data, 'fuzzy', fuzzy_vars)
        X = fuzzy_values['output']
        X_hdrs = fuzzy_values['names']
    # 2) num_distance: get the numeric distance between the two values, with a score of -9999 if missing
    numeric_dist_vars = [i for i in arg['var_rec'] if i['match_type'] == 'num_distance']
    if len(numeric_dist_vars) > 0:
        numeric_dist_values = create_scores(input_data, 'numeric_dist', numeric_dist_vars)
        X = np.hstack((X, numeric_dist_values['output']))
        X_hdrs.extend(numeric_dist_values['names'])
    # 3) exact: if they match, 1, else 0
    exact_match_vars = [i for i in arg['var_rec'] if i['match_type'] == 'exact']
    if len(exact_match_vars) > 0:
        exact_match_values = create_scores(input_data, 'exact', exact_match_vars)
        X = np.hstack((X, exact_match_values['output']))
        X_hdrs.extend(exact_match_values['names'])
    ###Now predict
    myprediction=probsfunc(arg['model'].predict_proba(X))
    ####don't need the scores anymore
    del X
    del data1
    del data2
    input_data['predicted_probability']=myprediction
    ###keep only the data above the certain thresholds
    if ast.literal_eval(os.environ['clerical_review_candidates'])==True:
        keep_thresh=min(float(os.environ['clerical_review_threshold']),float(os.environ['match_threshold']))
    else:
        keep_thresh=float(os.environ['match_threshold'])
    input_data=input_data.loc[input_data['predicted_probability'] >= keep_thresh].to_dict('record')
    if len(input_data) > 0:
        ###try to write to db, if not return and then we will dump later
        cur=db.cursor()
         ####if we are looking for clerical review candidates, push either all the records OR a 10% sample, depending on which is larger
        clerical_review_sql = '''insert into clerical_review_candidates({}_id, {}_id, predicted_probability) values (?,?,?) '''.format(
            os.environ['data1_name'], os.environ['data2_name'])
        match_sql = '''insert into matched_pairs({data1}_id, {data2}_id, predicted_probability, {data1}_rank, {data2}_rank) values (?,?,?,?,?) '''.format(
            data1=os.environ['data1_name'], data2=os.environ['data2_name'])
        ###setup the to_return
        to_return={}
        if ast.literal_eval(os.environ['clerical_review_candidates']) == True:
            if len(input_data) >= 10:
                clerical_review_dict = dcpy(input_data.sample(frac=.1))
            else:
                clerical_review_dict = dcpy(input_data)
            columns = clerical_review_dict[0].keys()
            vals = [tuple(i[column] for column in columns) for i in clerical_review_dict]
            try:
                cur.executemany(clerical_review_sql, vals)
                db.commit()
            except Exception:
                cur.close()
                cur=db.cursor()
                to_return['clerical_review_candidates']=input_data
            ###if it's a chatty logger, log.
            if ast.literal_eval(os.environ['chatty_logger']) == True:
                logger.info('''{} clerical review candidates added'''.format(len(clerical_review_dict)))
        ####Now add in any matches
        matches = pd.DataFrame(
            [i for i in input_data if i['predicted_probability'] >= float(os.environ['match_threshold'])])
        ###ranks
        matches['{}_rank'.format(os.environ['data1_name'])] = matches.groupby('{}_id'.format(os.environ['data1_name']))[
            'predicted_probability'].rank('dense')
        matches['{}_rank'.format(os.environ['data2_name'])] = matches.groupby('{}_id'.format(os.environ['data2_name']))[
            'predicted_probability'].rank('dense')
        ##convert back to dict
        matches = matches.to_dict('record')
        columns=matches[0].keys()
        vals = [tuple(i[column] for column in columns) for i in matches]
        try:
            cur.executemany(match_sql, vals)
            db.commit()
        except Exception:
            to_return['matches']=matches
        cur.close()
        db.close()
        if ast.literal_eval(os.environ['chatty_logger']) == True:
                    logger.info('''{} matches added in block'''.format(len(matches)))
        if 'clerical_review_candidates'in to_return.keys() or 'matches' in to_return.keys():
            return to_return
        else:
            return None

####The block function
def run_block(block, rf_mod):
    '''
    This function creates the list of blocks to use for comparison and then runs the matches
    :param block: the dict of the block we are running
    :param rf_mod; the random forest model we are using
    :return:
    '''
    ####First get the blocks that appear in both
    db=get_connection_sqlite(os.environ['db_name'])
    data1_blocks=get_table_noconn('''select distinct {} as block from {}'''.format(block[os.environ['data1_name']], os.environ['data1_name']), db)
    data1_blocks=[i['block'] for i in data1_blocks]
    data2_blocks=get_table_noconn('''select distinct {} as block from {}'''.format(block[os.environ['data2_name']], os.environ['data2_name']), db)
    data2_blocks=[i['block'] for i in data2_blocks]
    block_list=intersection(data1_blocks, data2_blocks)
    logger.info('We have {} blocks to run for {}'.format(len(block_list), block['block_name']))
    ###get the list of variables we are trying to match
    var_rec = pd.read_csv('mamba_variable_types.csv').to_dict('record')
    ###Create and kick off the runner
    arg_list=[]
    for k in range(len(block_list)):
        my_arg={}
        my_arg['var_rec']=var_rec
        my_arg['model']=rf_mod
        my_arg['target']=block_list[k]
        my_arg['block_info']=block
        if k % (len(block_list)/10)==0:
            my_arg['logging_flag']=int(np.round(100*(k/len(block_list)),0))
        arg_list.append(my_arg)
    logger.info('STARTING TO MATCH  FOR {}'.format(block['block_name']))
    pool=Pool(numWorkers)
    out=pool.map(match_fun, arg_list)
    ###Once that is done, need to
    ##push the remaining items in out to the db
    db=get_connection_sqlite(os.environ['dbname'])
    cur=db.cursor()
    logger.info('Dumping remaining matches to DB')
    clerical_review_sql = '''insert into clerical_review_candidates({}_id, {}_id, predicted_probability) values (?,?,?) '''.format(
        os.environ['data1_name'], os.environ['data2_name'])
    match_sql = '''insert into matched_pairs({data1}_id, {data2}_id, predicted_probability, {data1}_rank, {data2}_rank) values (?,?,?,?,?) '''.format(
        data1=os.environ['data1_name'], data2=os.environ['data2_name'])
    for i in out:
        if i!=None:
            if 'clerical_review_candidates' in i.keys():
                columns = k['clerical_review_candidates'][0].keys()
                vals = [tuple(i[column] for column in columns) for i in k['clerical_review_candidates']]
                cur.executemany(clerical_review_sql, vals)
            if 'matches' in i.keys():
                columns = k['matches'][0].keys()
                vals = [tuple(i[column] for column in columns) for i in k['matches']]
                cur.executemany(match_sql, vals)
    db.commit()
    ##then change the matched flags on the data tables to 1 where it's been matched
    ###updating the matched flags
    cur.execute('''update {data1_name} set matched=1 where id in (select distinct {data1_name}_id from matched_pairs)'''.format(data1_name=os.environ['data1_name']))
    cur.execute('''update {data2_name} set matched=1 where id in (select distinct {data2_name}_id from matched_pairs)'''.format(data2_name=os.environ['data2_name']))
    db.commit()
    db.close()
    logger.info('Block {} Complete'.format(block['block_name']))

if __name__=='__main__':
    print('why did you do this?')
    
    
    