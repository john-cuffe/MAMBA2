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
    '''
    db=get_connection_sqlite(os.environ['db_name'])
    if score_type=='fuzzy':
        data1_ids=','.join(str(v) for v in input_data['{}_id'.format(os.environ['data1_name'])].drop_duplicates().to_list())
        data2_ids=','.join(str(v) for v in input_data['{}_id'.format(os.environ['data2_name'])].drop_duplicates().to_list())
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
    ###making the dependent variable array
    y = truthdat['match'].values
    ###Generate the Grid Search to find the ideal values
    features_per_tree = ['sqrt', 'log2', 10, 15]
    rf = RandomForestClassifier(n_jobs=int(os.environ['rf_jobs']), max_depth=10, max_features='sqrt', n_estimators=10)
    myparams = {
        'n_estimators': sp_randint(1, 10),
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
    logger.info('\n\n######{0} MINUTES: THE BEST FITTING MODEL FOR YOUR DATA HAS: ######\n\n'.format(
        str(round((time.time() - start) / 60, 2))))
    logger.info('{0} trees, {1} features per tree'.format(str(trees), str(features_per_tree)))
    logger.info('and attains a {0} score of {1}'.format(scoringcriteria, str(round(score, 3))))
    logger.info('\n\n######{0} MINUTES: FIT BEST RANDOM FOREST MODEL ######\n\n'.format(
        str(round((time.time() - start) / 60, 2))))
    rf_mod = runRFClassifier(y, X, trees, features_per_tree, max_depth)
    return rf_mod, X_hdrs
    ###Otherwise, take the best 25%, or 4, features, whichever is greater
    # else:
    #     if len(X_hdrs) < 16:
    #         mambalite_len=4
    #     else:
    #         mambalite_len=int(np.round(len(X_hdrs)/4))
    #     outls = featSelRFImpurity(y, X, X_hdrs, mambalite_len, cv_rfc.best_params_['n_estimators'], cv_rfc.best_params_['max_features'],cv_rfc.best_params_['max_depth'])
    #     bestfeats = [i[0] for i in outls]
    #     logger.info('\n\n######{} MINUTES: FOR MAMBA LITE, WE WILL USE: {}######\n\n'.format(
    #         str(round((time.time() - start) / 60, 2)), str(", ".join(bestfeats))))
    #     ####Now let's re-run the model
    #     ###first we need the adequate features
    #     lite_X = np.zeros(shape=(len(truthdat), 0))
    #     for i in [fuzzy_values, numeric_dist_values, exact_match_values]:
    #         keeplist=[]
    #         for k in range(len(i['names'])):
    #             if i['names'][k] in bestfeats:
    #                 keeplist.append(k)
    #         if len(keeplist) > 0:
    #             lite_X=np.hstack((lite_X,i['output'][:,keeplist]))
    #     features_per_tree = ['sqrt', 'log2', 5, 8]
    #     ###setting the parameters
    #     myparams = {
    #         'n_estimators': sp_randint(10, 100),
    #         'max_features': features_per_tree,
    #         'max_depth': sp_randint(5, 16)}
    #     rf = RandomForestClassifier(n_jobs=int(os.environ['rf_jobs']), max_depth=10, max_features='sqrt', n_estimators=10)
    #     if debug:
    #         niter = 5
    #     else:
    #         niter = 200
    #     cv_rfc = RandomizedSearchCV(estimator=rf, param_distributions=myparams, cv=10, scoring=scoringcriteria,
    #                                 n_iter=niter)
    #     cv_rfc.fit(lite_X, y)
    #     ##Save the parameters
    #     trees = cv_rfc.best_params_['n_estimators']
    #     features_per_tree = cv_rfc.best_params_['max_features']
    #     score = cv_rfc.best_score_
    #     max_depth = cv_rfc.best_params_['max_depth']
    #     logger.info('\n\n######{0} MINUTES: THE BEST FITTING LITE MODEL FOR YOUR DATA HAS: ######\n\n'.format(
    #         str(round((time.time() - start) / 60, 2))))
    #     logger.info('{0} trees, {1} features per tree'.format(str(trees), str(features_per_tree)))
    #     logger.info('and attains a {0} score of {1}'.format(scoringcriteria, str(round(score, 3))))
    #     logger.info('\n\n######{0} MINUTES: FIT BEST RANDOM FOREST MODEL ######\n\n'.format(
    #         str(round((time.time() - start) / 60, 2))))
    #     rf_mod = runRFClassifier(y, lite_X, trees, features_per_tree, max_depth)
    #     mamba_lite_feats = bestfeats
    #     return rf_mod, mamba_lite_feats
    #

############
###CHUNK: CREATE RUNNER OBJECT
############
class Runner(object):
    '''
    # Class to hold shared attributes shared across workers
    #Description: For each geographic block, creates a container object with block-specific information for storage of local variables
    ##Inputs
      ###date--timestamp for logging purposes
      ###numWorkers--number of workers who will match the data--2 fewer than CPUs with the single worker
      ###lock--an object of the multiproccessing.lock() class that ensures queue safety
      ###database--the database to connect
      ###model--the random forest model to use for predictions
      ###block--what is the geographic block we are matching on?
    ###returns
    #None, but spits out .csv files for the matches for each block into the allmatch and bestmatch .csv files created in the main() statement
    '''

    ############
    ###Runner Step: define common traits across workers (referred to as self.trait   )
    ############
    def __init__(self, numWorkers, lock, model, block, var_rec, blocklen, db_name, data1_name, data2_name, logger):
        self.numWorkers = numWorkers  ###how many workers will we have?
        self.m = multiprocessing.Manager()  ###create the object "m" which is a multiprocessing manager
        self.outQueue = self.m.Queue()  ###out queue.  this queue will be filled with arrays to be predicted
        self.inQueue = self.m.Queue()  ###in queue.  This queue is the list of blocks to process
        self.writeQueue = self.m.Queue()  ###The write queue.  This queue is a single process, exclusively designed to write and avoid conention
        self.t0 = time.time()  ###initialization time
        self.lock = lock  ##the lock, that stores the counters that will run across workers
        self.chunksPulled = multiprocessing.Value('i', 0)  ###running counter of the number of chunks loaded
        self.chunksMatched = multiprocessing.Value('i', 0)  ##runnign counter of chunks matched
        self.chunksProcessed = multiprocessing.Value('i', 0)  ###running counter of processed records
        self.recordsPulled = multiprocessing.Value('i', 0)  ###running counter of records pulled from data
        self.chunksWritten = multiprocessing.Value('i', 0)  ###running counter of number of chunks written
        self.endOfStatementFile = multiprocessing.Value('i', False)  ###
        self.model = model
        self.block = copy.deepcopy(block)  ###The name of the block (e.g zip5)
        self.environ=dict(os.environ)
        self.var_rec=copy.deepcopy(var_rec) ###the variables we are matching with
        self.db_name=db_name
        self.data1_name=data1_name
        self.data2_name=data2_name
        ###Logging flag--if we are in debug mode, just log every 50 obs.  otherwise log every 10% of the chunks
        self.loggingflag = np.floor(blocklen / 10)  ##The logging flag.  Logs every 10% (approximately) of chunks matched
        self.logger = logger

    ############
    ###Define the Log time, which will give us the minutes since the initiation of the Runner
    ############
    # def logTime(self):  ###getting the log time functions, which compares the current time with the start time
    #     t = round((time.time() - self.t0) / 60, 2)
    #     logger.info('Block {0} Minutes lapsed: {1}'.format(self.block, t))
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    ############
    ###CREATING THE 'worker'
    ############
    def worker(self):
        '''
        Description: a worker is an object that runs the predictions.  It gets a
        Inputs: Self (the runner object)
        Outputs (in the /output directory):
        '''
        myconnt = get_connection_sqlite(self.db_name)
        churn = True
        # if matchingtype=="NameOnly":
        # tfidfdat=pd.read_sql_query('select '+self.matchinglevelvar+', stnnamebr, minyear, maxyear, tfidf_mean from br', myconnt)
        ###load the tfidf data once to save time
        ###Add in the tfidf for the BR values
        while churn:
            if not self.inQueue.empty():
                try:
                    target_block = self.inQueue.get(timeout=240)
                except:
                    time.sleep(10)
                    continue
                ###run the match
                self.runner_match(target_block)
                with self.lock:
                    self.chunksMatched.value += 1
                continue
            ###If the above condition isn't met, we can convert the worker to a worker
            ####if the inqueue isn't empty and the outqueue is empty and the block isn't the all block, run a match for the geographic blocking
            else:
                logger.info('####### Worker Breaking #######')
                break


    ############
    ###CREATING THE WRITER OBJECT
    ##########
    def writer(self):
        '''
        This object is a special process only decidated to grabbing items from the writeQueue and then pushing them to the DB.
        If the writequeue is empty, it becomes a worker
        '''
        ##set up a db connection
        ####conn string
        myconnt = get_connection_sqlite(os.environ['db_name'])
        cur=myconnt.cursor()
        churn = True
        ###sleep for two minutes to get things started
        # logger.info('Writer Sleeping')
        # time.sleep(10)
        # if matchingtype=="NameOnly":
        # tfidfdat=pd.read_sql_query('select '+self.matchinglevelvar+', stnnamebr, minyear, maxyear, tfidf_mean from br', myconnt)
        ###load the tfidf data once to save time
        ###Add in the tfidf for the BR values
        while churn:
            #####If we haven't pulled all the chunks and the out queue has any chunks of matches waiting, run a prediction
            if not self.writeQueue.empty():
                self.dumpOutQueue()
                ###add chunk written value
                with self.lock:
                    self.chunksWritten.value += 1
            ###If the above condition isn't met, we can convert the writer to a worker
            ####if the inqueue isn't empty and the outqueue is empty and the block isn't the all block, run a match for the geographic blocking
            elif not self.inQueue.empty() and self.writeQueue.empty():
                try:
                    target_block = self.inQueue.get(timeout=240)
                except:
                    time.sleep(10)
                    continue
                self.runner_match(block)
                continue
            ####if the inqueue isn't empty and the outqueue is empty and the block is all, run an allmatch (which has the loop for BR
            elif self.inQueue.empty() and self.outQueue.empty():
                ###sleep for 60 seconds
                time.sleep(60)
                if self.inQueue.empty() and self.outQueue.empty():
                    logger.info('####### Writer Breaking #######')
                    break
                else:
                    continue

    ####The actual match funciton
    def runner_match(self, target_block):
        '''
        This function runs the match.  block is the target block
        '''
        ###get the two dataframes
        data1=get_table_noconn('''select id from {} where {}='{}' and matched=0'''.format(os.environ['data1_name'], self.data1_name, target_block), myconnt)
        data2=get_table_noconn('''select id from {} where {}='{}' and matched=0'''.format(os.environ['data2_name'], self.data2_name, target_block), myconnt)
        ###get the data
        input_data=pd.DataFrame([{'{}_id'.format(os.environ['data1_name']):str(k['id']), '{}_id'.format(os.environ['data2_name']):str(y['id'])} for k in data1 for y in data2])
        ####Now get the data organized, by creating an array of the different types of variables
        # 1) Fuzzy: Create all the fuzzy values from febrl
        fuzzy_vars = [i for i in self.var_rec if i['match_type'] == 'fuzzy']
        if len(fuzzy_vars) > 0:
            fuzzy_values = create_scores(input_data, 'fuzzy', fuzzy_vars)
            X = fuzzy_values['output']
            X_hdrs = fuzzy_values['names']
        # 2) num_distance: get the numeric distance between the two values, with a score of -9999 if missing
        numeric_dist_vars = [i for i in self.var_rec if i['match_type'] == 'num_distance']
        if len(numeric_dist_vars) > 0:
            numeric_dist_values = create_scores(input_data, 'numeric_dist', numeric_dist_vars)
            X = np.hstack((X, numeric_dist_values['output']))
            X_hdrs.extend(numeric_dist_values['names'])
        # 3) exact: if they match, 1, else 0
        exact_match_vars = [i for i in self.var_rec if i['match_type'] == 'exact']
        if len(exact_match_vars) > 0:
            exact_match_values = create_scores(input_data, 'exact', exact_match_vars)
            X = np.hstack((X, exact_match_values['output']))
            X_hdrs.extend(exact_match_values['names'])
        ###Now predict
        myprediction=probsfunc(self.model.predict_proba(X))
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
        input_data=input_data.loc[input_data['predicted_probability'] >= keep_thresh]
        if len(input_data) > 0:
            outdict={'output':input_data.to_dict('record'), 'block':target_block}
            self.writeQueue.put(outdict)
    ####DumpQueue functions
    ####Dump the outQueue to write Queue
    def dumpOutQueue(self):
        '''
        This function dumps the outQueue into the write Queue for any remaining blocks
        '''
        mydict = self.writeQueue.get()
        ####if we are looking for clerical review candidates, push either all the records OR a 10% sample, depending on which is larger
        clerical_review_sql = '''insert into clerical_review_candidates({}_id, {}_id, predicted_probability) values (?,?,?) '''.format(
            os.environ['data1_name'], os.environ['data2_name'])
        match_sql = '''insert into matched_pairs({data1}_id, {data2}_id, predicted_probability, {data1}_rank, {data2}_rank) values (?,?,?,?,?) '''.format(
            data1=os.environ['data1_name'], data2=os.environ['data2_name'])
        if ast.literal_eval(os.environ['clerical_review_candidates']) == True:
            if len(mydict['output']) >= 10:
                clerical_review_dict = dcpy(mydict['output'].sample(frac=.1))
            else:
                clerical_review_dict = dcpy(mydict['output'])
            columns = clerical_review_dict[0].keys()
            vals = [tuple(i[column] for column in columns) for i in clerical_review_dict]
            cur.executemany(clerical_review_sql, vals)
            myconnt.commit()
            if ast.literal_eval(os.environ['chatty_logger']) == True:
                logger.info('''{} clerical review candidates added in block {}'''.format(len(clerical_review_dict),
                                                                                         mydict['block']))
        ####Now add in any matches
        matches = pd.DataFrame(
            [i for i in mydict['output'] if i['predicted_probability'] >= float(os.environ['match_threshold'])])
        ###ranks
        matches['{}_rank'.format(os.environ['data1_name'])] = matches.groupby('{}_id'.format(os.environ['data1_name']))[
            'predicted_probability'].rank('dense')
        matches['{}_rank'.format(os.environ['data2_name'])] = matches.groupby('{}_id'.format(os.environ['data2_name']))[
            'predicted_probability'].rank('dense')
        ##convert back to dict
        matches = matches.to_dict('record')
        vals = [tuple(i[column] for column in columns) for i in matches]
        cur.executemany(match_sql, vals)
        myconnt.commit()
        if ast.literal_eval(os.environ['chatty_logger']) == True:
                    logger.info('''{} matches added in block {}'''.format(len(matches), mydict['block']))

    ###main dumpQueue function
    def dumpQueues(self):
        '''
        This function dumps the queues at the end of the run to ensure we don't miss anything/avoid hung sessions
        '''
        logger.info('#### Beginning outQueue dump: {0} blocks ####'.format(self.writeQueue.qsize()))
        while self.writeQueue.qsize() > 0:
            self.dumpOutQueue()
        logger.info('##### writeQueue dump complete ####'.format(self.writeQueue.qsize()))



def intersection(lst1, lst2):
    '''
    Intersection method from https://www.geeksforgeeks.org/python-intersection-two-lists/
    :param lst1:
    :param lst2:
    :return:
    '''
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

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
    lock=multiprocessing.Lock()
    runner = Runner(numWorkers,lock, rf_mod, copy.deepcopy(block), copy.deepcopy(var_rec), len(block_list), os.environ['db_name'], os.environ['data1_name'], os.environ['data2_name'], logger)
    # Start the Input Queue, a Q of input blocks.  Put each one in there
    logger.info( 'FILLING BLOCK QUEUE FOR {}'.format(block['block_name']))
    for i in range(len(block_list)):
        # if i % (len(runner.chunks))/float(10)==0:
        # logger.info('{} Chunks inserted into queue.  Queue size = {}'.format(i, sys.getsizeof(runner.inQueue)/float(1000000000)))
        runner.inQueue.put(block_list[i])
    logger.info('STARTING TO MATCH  FOR {}'.format(block['block_name']))
    # Create and kick off workers
    processes = []
    ##set up the workers, where l is an indivudal worker in the range from 1:numWorkers
    for l in range(int(os.environ['numWorkers'])):
        l = multiprocessing.Process(target=runner.worker)
        l.start()
        processes.append(l)

    ###set up the writer
    w = multiprocessing.Process(target=runner.writer)
    w.start()
    processes.append(w)
    ###Now kick off the workers
    for p in processes:
        p.join()

    runner.dumpQueues()

if __name__=='__main__':
    print('why did you do this?')
    
    
    